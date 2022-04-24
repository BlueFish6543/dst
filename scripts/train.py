from __future__ import annotations

import json
import logging
import pathlib
import sys
import time
from pathlib import Path
from typing import Optional

import click
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from dst.dataset import (
    BatchedTestDataset,
    Vocabulary,
    get_dataloader,
    get_inference_data_loader,
)
from dst.evaluation import get_metrics, log_metrics_to_tb, save_metrics
from dst.inference import run_inference, setup_inference_config
from dst.parser import parse, setup_parser
from dst.scoring_utils import setup_evaluator_inputs
from dst.utils import (
    get_data_version,
    get_datetime,
    load_model,
    load_optimizer_scheduler,
    load_schema,
    save_checkpoint,
    set_seed,
)

LR_LOG_FREQ_CHANGE_LIMIT = 32000
"""LR is logged with freq specified in train args until
this many examples have been seen."""
LR_LOG_FREQ_BATCHES = 2500
"""LR logged when batch number divisible by this after
LR_LOG_FREQ_CHANGE_LIMIT is exceeded."""


try:
    from apex.optimizers import FusedAdam

    AdamW = FusedAdam
except ModuleNotFoundError:
    from torch.optim import AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def compute_dev_lm_loss(args, dataloader, model) -> dict[str, float]:
    logger.info("Computing LM loss...")
    total_tokens, total_seen_tokens, total_unseen_tokens = 0, 0, 0
    loss_total, combined_loss_total, seen_loss_total, unseen_loss_total = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    num_batches = 0
    model.eval()
    start_time = time.time()
    ignore_idx = dataloader.dataset.ignore_token_id
    loss_fct = CrossEntropyLoss(ignore_index=ignore_idx, reduction="none")
    for batch in tqdm(
        dataloader, desc=dataloader.dataset.split, disable=args.verbose.disable_display
    ):
        batch_size = batch["input_ids"].shape[0]
        assert isinstance(batch_size, int)
        num_batches += 1
        labels = batch["label_ids"].to(DEVICE)
        seen_examples_mask = batch["seen"].to(DEVICE)
        labels_mask = labels != ignore_idx
        seen_labels_mask = labels_mask[seen_examples_mask]
        unseen_labels_mask = labels_mask[~seen_examples_mask]
        with torch.no_grad():
            output = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=labels,
            )
            # [B, S, V] -> [B, V, S] where B = batch size, S = max out seq len, V = vocab size
            logits = output.logits.transpose(2, 1)
            # [B, S]
            per_token_losses = loss_fct(logits, labels)
            seen_token_losses = per_token_losses[seen_examples_mask]
            unseen_token_losses = per_token_losses[~seen_examples_mask]
            seen_examples = seen_examples_mask.sum().item()
            unseen_examples = (~seen_examples_mask).sum().item()
            assert seen_examples + unseen_examples == batch_size
            total_seen_tokens += seen_labels_mask.sum().item()
            total_unseen_tokens += unseen_labels_mask.sum().item()
            total_tokens += labels_mask.sum().item()
            assert total_tokens == (total_seen_tokens + total_unseen_tokens)
            combined_loss = per_token_losses.sum().item()
            combined_loss_total += combined_loss
            if seen_examples > 0:
                seen_loss = seen_token_losses.sum().item()
                seen_loss_total += seen_loss
            if unseen_examples > 0:
                unseen_loss = unseen_token_losses.sum().item()
                unseen_loss_total += unseen_loss
        loss_total += output.loss.item()
    logger.info(f"Took {time.time() - start_time:.3f} to evaluate dev likelihood")
    torch.cuda.empty_cache()
    return {
        "dev_avg_token_batch": loss_total / num_batches,
        "dev_seen_per_token": seen_loss_total / total_seen_tokens,
        "dev_unseen_per_token": unseen_loss_total / total_unseen_tokens,
        "dev_per_token": combined_loss_total / total_tokens,
    }


def compute_task_oriented_metrics(
    global_step: int,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    inference_config: DictConfig,
    inference_data_loader: BatchedTestDataset,
) -> dict:
    """Compute the SGD metrics for the current model."""

    this_ckpt_hyp_path = Path(inference_config.hyp_dir).joinpath(
        f"model.{str(global_step)}"
    )
    belief_states = {}
    if this_ckpt_hyp_path.exists():
        if not inference_config.override:
            logger.warning(
                f"Cannot override predictions for {this_ckpt_hyp_path}, skipping decoding. "
                f"Use --override flag to achieve this behaviour."
            )
            belief_states = None
        else:
            logger.warning(f"Overriding predictions for {this_ckpt_hyp_path}")
    else:
        this_ckpt_hyp_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Decoding model after {global_step} seen examples. "
        f"Saving dialogues and belief states to {this_ckpt_hyp_path}"
    )
    if belief_states is not None:
        logger.info(f"Computing task oriented metrics at global step {global_step}")
        belief_states = run_inference(
            inference_config, tokenizer, model, inference_data_loader, DEVICE
        )
        with open(this_ckpt_hyp_path.joinpath("belief_states.json"), "w") as f:
            json.dump(belief_states, f)
        assert isinstance(inference_config.dst_test_path, ListConfig)
        logger.info(f"Parsing model predictions at global step {global_step}")
        parser_inputs = setup_parser(
            inference_config.ref_schema_path,
            inference_config.dst_test_path[0],
            inference_config.template_dir,
            str(this_ckpt_hyp_path),
        )
        parse(
            parser_inputs["schema"],
            belief_states,
            parser_inputs["preprocessed_references"],
            this_ckpt_hyp_path,
            inference_config,
            value_separator=inference_config.preprocessing.value_selection.value_separator,
            recase_categorical_values=inference_config.preprocessing.lowercase_model_targets,
            target_slot_index_separator=inference_config.preprocessing.target_slot_index_separator,
            files_to_parse=inference_data_loader.dialogue_files,
        )
        logger.info(f"Evaluating model predictions at global step {global_step}")
        evaluator_inputs = setup_evaluator_inputs(this_ckpt_hyp_path, inference_config)
        all_metrics_aggregate, _ = get_metrics(
            evaluator_inputs["dataset_ref"],
            evaluator_inputs["dataset_hyp"],
            evaluator_inputs["eval_services"],
            evaluator_inputs["in_domain_services"],
        )
        logger.info(f"Dialogue metrics {str(all_metrics_aggregate['#ALL_SERVICES'])}")
        logger.info(f"Dialogue metrics {str(all_metrics_aggregate['#SEEN_SERVICES'])}")
        logger.info(
            f"Dialogue metrics {str(all_metrics_aggregate['#UNSEEN_SERVICES'])}"
        )
        save_metrics(
            global_step,
            Path(inference_config.metrics_dir),
            this_ckpt_hyp_path,
            evaluator_inputs,
            all_metrics_aggregate,
        )
        return all_metrics_aggregate
    return {}


def optimize_model(
    args,
    tokenizer,
    model,
    train_dataloader,
    dev_dataloader,
    optimizer,
    scheduler,
    initial_step: int = 0,
    inference_config: Optional[DictConfig] = None,
    inference_data_loader: Optional[BatchedTestDataset] = None,
    max_dev_jga: float = 0.0,
    patience: int = 0,
):
    def _log_lr(scheduler: torch.optim.lr_scheduler.LambdaLR):

        nonlocal writer
        log_lr = False
        if scheduler is not None:
            if n_batches * train_args.batch_size > LR_LOG_FREQ_CHANGE_LIMIT:
                if n_batches % LR_LOG_FREQ_BATCHES == 0:
                    log_lr = True
            else:
                if n_batches % train_args.lr_log_freq == 0:
                    log_lr = True
            if log_lr:
                lrs = scheduler.get_last_lr()
                assert len(lrs) == 1
                writer.add_scalar(
                    "lr", lrs[0], global_step=n_batches * train_args.batch_size
                )

    train_dev_args = args
    dev_args, train_args = args.dev, args.train
    max_patience = (dev_args.eval_interval // train_args.batch_size) / (
        dev_args.patience // train_args.batch_size
    )
    # assume that for some reason we want to continue training
    # after max dev jga stopped it
    if initial_step != 0 and patience == max_patience:
        patience = 0
    log_dir = (
        Path()
        .resolve()
        .joinpath("runs", f"{train_args.experiment_name}_version_{args.data.version}")
    )
    writer = SummaryWriter(
        log_dir=str(log_dir),
    )
    logger.info(f"Tensorboard logs saved at: {log_dir}")
    eval_step = dev_args.eval_interval // train_args.batch_size
    n_batches = initial_step // train_args.batch_size
    dev_losses = compute_dev_lm_loss(dev_args, dev_dataloader, model)
    dev_jga = max_dev_jga
    for subset, value in dev_losses.items():
        logger.info(f"Batches {n_batches} | {subset} loss: {value:.8f}")
        if n_batches > 0:
            # We can't actually read the plot if we log that value
            writer.add_scalar(
                f"Loss/{subset}", value, global_step=n_batches * train_args.batch_size
            )
    logger.info("Start training!")
    stop_training = False
    for epoch in range(train_args.epochs):
        # Initialise for each epoch
        start_time = time.time()
        loss_disp = 0
        model.train()
        model.zero_grad()

        iterator = enumerate(
            tqdm(
                train_dataloader,
                desc=f"Epoch {epoch}",
                disable=train_args.verbose.disable_display,
            )
        )
        local_step = 0
        for local_step, batch in iterator:
            output = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["label_ids"].to(DEVICE),
            )
            loss = output.loss.mean()
            loss_disp += output.loss.mean().item()
            n_batches += 1
            # Update model
            if loss.item() != 0:
                loss = loss / train_args.gradient_accumulation_steps
                loss.backward()
            if n_batches % train_args.gradient_accumulation_steps == 0:
                optimizer.step()
                if train_args.use_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            _log_lr(scheduler)

            if n_batches % eval_step == 0:
                task_oriented_metrics = compute_task_oriented_metrics(
                    n_batches * train_args.batch_size,
                    model,
                    tokenizer,
                    inference_config,
                    inference_data_loader,
                )
                log_metrics_to_tb(
                    n_batches * train_args.batch_size, writer, task_oriented_metrics
                )
                if task_oriented_metrics:
                    inference_config.date = get_datetime()
                    OmegaConf.save(
                        args,
                        f=inference_config.hyp_path.joinpath(
                            f"model.{n_batches * train_args.batch_size}",
                            "experiment_config.yaml",
                        ),
                    )
                    dev_jga = task_oriented_metrics["#ALL_SERVICES"][
                        "joint_goal_accuracy"
                    ]
                    if dev_jga > max_dev_jga:
                        max_dev_jga = dev_jga
                        patience = 0
                    else:
                        patience += 1
                    if patience == max_patience:
                        stop_training = True
                        break
                dev_losses = compute_dev_lm_loss(dev_args, dev_dataloader, model)
                for subset, value in dev_losses.items():
                    logger.info(
                        f"Epoch: {epoch} | "
                        f"Batch: {n_batches} | "
                        f"{subset} loss: {value:.8f}"
                    )
                    writer.add_scalar(
                        f"Loss/{subset}",
                        value,
                        global_step=n_batches * train_args.batch_size,
                    )
                dev_jga = (
                    0.0
                    if not task_oriented_metrics
                    else task_oriented_metrics["#ALL_SERVICES"]["joint_goal_accuracy"]
                )
                save_checkpoint(
                    train_dev_args,
                    tokenizer,
                    model,
                    n_batches * train_args.batch_size,
                    optimizer,
                    scheduler,
                    train_args.batch_size * n_batches,
                    dev_jga=dev_jga,
                    patience=patience,
                )
                model.train()
            if n_batches in train_args.global_step_checkpoints:
                save_checkpoint(
                    train_dev_args,
                    tokenizer,
                    model,
                    n_batches * train_args.batch_size,
                    optimizer,
                    scheduler,
                    train_args.batch_size * n_batches,
                    dev_jga=dev_jga,
                    patience=patience,
                )
        loss_disp /= local_step + 1
        logger.info(
            f"Epoch: {epoch} | Batch: {n_batches} | "
            f"Train loss: {loss_disp:.8f} | "
            f"Time: {time.time() - start_time:.3f}"
        )
        writer.add_scalar(
            "Loss/train", loss_disp, global_step=n_batches * train_args.batch_size
        )
        save_checkpoint(
            train_dev_args,
            tokenizer,
            model,
            n_batches * train_args.batch_size,
            optimizer,
            scheduler,
            train_args.batch_size * n_batches,
            dev_jga=dev_jga,
            patience=patience,
        )
        save_checkpoint(
            train_dev_args,
            tokenizer,
            model,
            "last",
            optimizer,
            scheduler,
            train_args.batch_size * n_batches,
            dev_jga=dev_jga,
            patience=patience,
        )
        dev_losses = compute_dev_lm_loss(dev_args, dev_dataloader, model)
        for subset, value in dev_losses.items():
            logger.info(
                f"Epoch: {epoch} | Batch: {n_batches} | {subset} loss: {value:.8f}"
            )
            writer.add_scalar(
                f"Loss/{subset}", value, global_step=n_batches * train_args.batch_size
            )
        if stop_training:
            break


def set_model(args: DictConfig, data_parallel: bool = False):
    # Initiate config, tokenizer and model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, config=config
    )
    vocabulary = Vocabulary()
    vocabulary.add_special_tokens(args.special_tokens)
    tokenizer.add_special_tokens(vocabulary.special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    if data_parallel:
        model = nn.DataParallel(model)
    model.to(DEVICE)
    return config, tokenizer, model


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-t",
    "--train_data",
    "train_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to training datasets.",
    multiple=True,
)
@click.option(
    "-s",
    "--orig_train_schema_path",
    "orig_train_schema_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute original train schema path. Used to determine seen/unseen examples",
)
@click.option(
    "-d",
    "--dev_data",
    "dev_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to dev datasets.",
    multiple=True,
)
@click.option(
    "-a",
    "--args",
    "args_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the file with the training arguments.",
)
@click.option(
    "-r",
    "--restore",
    "ckpt_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the checkpoint folder from where the model is to be loaded.",
)
@click.option(
    "--do_inference",
    is_flag=True,
    default=False,
    help="Evaluate task-oriented performance on dev set.",
)
# decoder args
@click.option(
    "--override",
    is_flag=True,
    default=False,
    help="Override previous predictions found at destination.",
)
@click.option(
    "-hyp",
    "--hyp-dir",
    "hyp_dir",
    type=click.Path(path_type=Path),
    help="Dir where hypothesis files are to be saved. "
    "Auto-suffixed with checkpoint step number.",
)
@click.option(
    "-templates",
    "--template_dir",
    "template_dir",
    type=click.Path(exists=True, path_type=Path),
    help="Absolute to the directory containing blank dialogue files for the dev set.",
)
# necessary for scoring
@click.option(
    "-ref",
    "--ref_dir",
    "ref_dir",
    type=click.Path(exists=True, path_type=Path),
    help="Dir where the references files for task-oriented eval are saved."
    "Necessary to evaluate task-oriented performance during training.",
)
def main(
    args_path: pathlib.Path,
    train_path: tuple[pathlib.Path],
    dev_path: tuple[pathlib.Path],
    log_level: int,
    ckpt_path: pathlib.Path,
    orig_train_schema_path: pathlib.Path,
    do_inference: bool,
    override: bool,
    hyp_dir: pathlib.Path,
    ref_dir: pathlib.Path,
    template_dir: pathlib.Path,
):
    args = OmegaConf.load(args_path)
    set_seed(args.reproduce)
    data_versions = {get_data_version(p) for p in train_path}
    assert (
        len(data_versions) == 1
    ), f"Attempting to train on multiple data versions {data_versions}"
    args.data.version = list(data_versions)[0]
    special_tokens = []
    for p in train_path + dev_path:
        model_input_config = OmegaConf.load(
            f"{p.parent.joinpath('preprocessing_config.yaml')}"
        )
        args.data.metadata[str(p)] = model_input_config.metadata
        args.data.preprocessing[str(p)] = model_input_config.preprocessing
        special_tokens.extend(model_input_config.preprocessing.special_tokens)
    args.train.special_tokens = list(set(special_tokens))
    args.train.dst_train_path = [str(p) for p in train_path]  # type: list[str]
    args.dev.dst_dev_path = [str(p) for p in dev_path]  # type: list[str]
    if len(args.dev.dst_dev_path) > 1:
        args.dev.dst_dev_path = [args.dev.dst_dev_path[0]]
        logger.warning(
            "Decoding multiple sets during inference is not supported. "
            f"Only {args.dev.dst_dev_path[0]} will be decoded..."
        )
    args.train.orig_train_schema_path = str(orig_train_schema_path)
    if do_inference:
        assert ref_dir.joinpath("schema.json").exists()
        args.decode.ref_path = str(ref_dir)
        args.decode.ref_schema_path = str(ref_dir.joinpath("schema.json"))
        args.decode.template_dir = str(template_dir)

    log_dir = (
        Path(args.train.checkpoint_dir)
        .joinpath(args.train.experiment_name, f"version_{args.data.version}", "logs")
        .resolve()
    )
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True, parents=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            "{}.log".format(log_dir.joinpath(Path(__file__).stem)),
            mode="a" if ckpt_path else "w",
        ),
    ]
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)
    if ckpt_path:
        logger.info(f"Restarting training from checkpoint: {ckpt_path}")

    logger.info("Training on: {}".format("GPU" if "cuda" in DEVICE.type else "CPU"))
    orig_train_schema = load_schema(orig_train_schema_path)
    if orig_train_schema is None:
        raise ValueError(
            "None of the paths provided via --train_data contains the original dataset"
        )
    if ckpt_path:
        # Load from checkpoint
        args.train.checkpoint = str(ckpt_path)
        config, tokenizer, model = load_model(
            args.train, device=DEVICE, data_parallel=args.train.data_parallel
        )
    else:
        config, tokenizer, model = set_model(
            args.train, data_parallel=args.train.data_parallel
        )

    if args.train.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.train.learning_rate, eps=args.train.adam_eps
        )
    elif args.train.optimizer == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=args.train.learning_rate,
        )
    elif args.train.optimizer == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=args.train.learning_rate,
            relative_step=args.train.relative_step,
            scale_parameter=args.train.scale_parameter,
        )
    else:
        raise ValueError(f"Unknown optimizer {args.train.optimizer}!")
    logger.info(
        f"Initialised optimizer: {type(optimizer)} from config option {args.train.optimizer}..."
    )
    train_dataloader = get_dataloader(
        args.train,
        tokenizer,
        args.train.dst_train_path,
        orig_train_schema,
        sampler=RandomSampler,
        data_size=args.train.data_size,
    )
    dev_dataloader = get_dataloader(
        args.dev,
        tokenizer,
        args.dev.dst_dev_path,
        orig_train_schema,
        sampler=SequentialSampler,
        data_size=args.dev.data_size,
    )
    scheduler = None
    metrics = {}
    initial_step = 0
    if args.train.use_scheduler:
        if args.train.optimizer in ["adam", "adamw"]:
            t_total = (
                len(train_dataloader)
                // args.train.gradient_accumulation_steps
                * args.train.epochs
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.train.warmup_steps,
                num_training_steps=t_total,
            )
        else:
            try:
                assert not all((args.train.relative_step, args.train.scale_parameter))
            except AssertionError:
                logger.error(
                    "Cannot use scheduler with Adafactor optimizer is automatic LR scheduling is enabled."
                    "Set training arguments `relative_step` and `scale_paramater` to `false`"
                )
                raise AssertionError
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.train.warmup_steps
            )
    if ckpt_path is not None:
        assert (
            ckpt_path.name == "model.last"
        ), "For reproducibility, load model saved at the end of the epoch"
        optimizer, scheduler, metrics, inital_step = load_optimizer_scheduler(
            ckpt_path, optimizer, scheduler
        )

    inference_config, inference_data_loader = None, None
    if do_inference:
        inference_config = setup_inference_config(args, hyp_dir, override)
        logger.info("Inference config...")
        logger.info(OmegaConf.to_yaml(inference_config))
        inference_data_loader = get_inference_data_loader(inference_config, tokenizer)
    logger.info(OmegaConf.to_yaml(args))
    optimize_model(
        args,
        tokenizer,
        model,
        train_dataloader,
        dev_dataloader,
        optimizer,
        scheduler,
        initial_step=initial_step,
        inference_config=inference_config,
        inference_data_loader=inference_data_loader,
        max_dev_jga=metrics.get("dev_jga", 0.0),
        patience=metrics.get("dev_jga_patience", 0),
    )


if __name__ == "__main__":
    main()
