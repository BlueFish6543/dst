from  __future__ import annotations
import logging
import pathlib
import sys
import time
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from src.dst.dataset import (
    TrainDataset,
    Vocabulary
)
from src.dst.utils import (
    set_seed,
    save_checkpoint,
    load_model,
    load_optimizer_scheduler,
    load_schema, Schema, get_data_version,
)
from torch.nn import CrossEntropyLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def get_dataloader(
        args: DictConfig,
        tokenizer,
        data_paths: list[str],
        schema: Schema,
        sampler,
        data_size : int = -1
) -> DataLoader:
    dataset = TrainDataset(args, tokenizer, data_paths, data_size, schema)
    dataloader = DataLoader(
        dataset,
        sampler=sampler(dataset),
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn
    )
    return dataloader


def compute_dev_lm_loss(args, dataloader, model) -> dict[str, float]:
    logger.info("Computing LM loss...")
    total_tokens, total_seen_tokens, total_unseen_tokens = 0, 0, 0
    loss_total, combined_loss_total, seen_loss_total, unseen_loss_total = 0.0, 0.0, 0.0, 0.0
    num_batches = 0
    model.eval()
    start_time = time.time()
    ignore_idx = dataloader.dataset.ignore_token_id
    loss_fct = CrossEntropyLoss(ignore_index=ignore_idx, reduction='none')
    for batch in tqdm(dataloader, desc=dataloader.dataset.split, disable=args.verbose.disable_display):
        batch_size = batch["input_ids"].shape[0]
        assert isinstance(batch_size, int)
        num_batches += 1
        labels = batch['label_ids'].to(DEVICE)
        seen_examples_mask = batch["seen"].to(DEVICE)
        labels_mask = labels != ignore_idx
        seen_labels_mask = labels_mask[seen_examples_mask]
        unseen_labels_mask = labels_mask[~seen_examples_mask]
        with torch.no_grad():
            output = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=labels
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
    return {
        'dev_avg_token_batch': loss_total / num_batches,
        'dev_seen_per_token': seen_loss_total / total_seen_tokens,
        'dev_unseen_per_token': unseen_loss_total / total_unseen_tokens,
        'dev_per_token': combined_loss_total / total_tokens
    }


def train(args, tokenizer, model, train_dataloader, dev_dataloader, optimizer, scheduler, initial_step: int  = 0):
    train_dev_args = args
    dev_args, train_args = args.dev, args.train
    log_dir = Path().resolve().joinpath("runs", f"{train_args.experiment_name}_version_{args.data.version}")
    writer = SummaryWriter(
        log_dir=str(log_dir),
    )
    logger.info(f"Tensorboard logs saved at: {log_dir}")
    eval_step = dev_args.eval_interval // train_args.batch_size
    global_step = initial_step // train_args.batch_size
    dev_losses = compute_dev_lm_loss(dev_args, dev_dataloader, model)
    for subset, value in dev_losses.items():
        logger.info(f"Epoch: {global_step} | {subset} loss: {value:.8f}")
        if global_step > 0:
            # We can't actually read the plot if we log that value
            writer.add_scalar(f'Loss/{subset}', value, global_step=global_step * train_args.batch_size)
    logger.info('Start training!')
    for epoch in range(train_args.epochs):
        # Initialise for each epoch
        start_time = time.time()
        loss_disp = 0
        model.train()
        model.zero_grad()

        iterator = enumerate(
            tqdm(
                train_dataloader,
                desc=f"Epoch {epoch}", disable=train_args.verbose.disable_display)
        )
        local_step = 0
        for local_step, batch in iterator:
            output = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=batch['label_ids'].to(DEVICE),
            )
            loss = output.loss.mean()
            loss_disp += output.loss.mean().item()
            global_step += 1
            # Update model
            if loss.item() != 0:
                loss = loss / train_args.gradient_accumulation_steps
                loss.backward()
            if global_step % train_args.gradient_accumulation_steps == 0:
                optimizer.step()
                if train_args.use_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            if global_step % eval_step == 0:
                dev_losses = compute_dev_lm_loss(dev_args, dev_dataloader, model)
                for subset, value in dev_losses.items():
                    logger.info(
                        f"Epoch: {epoch} | "
                        f"Batch: {global_step} | "
                        f"{subset} loss: {value:.8f}"
                    )
                    writer.add_scalar(
                        f'Loss/{subset}',
                        value,
                        global_step=global_step * train_args.batch_size
                    )
                save_checkpoint(
                    train_dev_args,
                    tokenizer,
                    model,
                    global_step * train_args.batch_size,
                    optimizer,
                    scheduler
                )
                model.train()
            if global_step in train_args.global_step_checkpoints:
                save_checkpoint(
                    train_dev_args,
                    tokenizer,
                    model,
                    global_step * train_args.batch_size,
                    optimizer,
                    scheduler
                )
        loss_disp /= (local_step + 1)
        logger.info(
            f"Epoch: {epoch} | Batch: {global_step} | "
            f"Train loss: {loss_disp:.8f} | "
            f"Time: {time.time() - start_time:.3f}"
        )
        writer.add_scalar('Loss/train', loss_disp, global_step=global_step * train_args.batch_size)
        dev_losses = compute_dev_lm_loss(dev_args, dev_dataloader, model)
        for subset, value in dev_losses.items():
            logger.info(f"Epoch: {epoch} | Batch: {global_step} | {subset} loss: {value:.8f}")
            writer.add_scalar(f'Loss/{subset}', value, global_step=global_step * train_args.batch_size)


def set_model(args: DictConfig, data_parallel: bool = False):
    # Initiate config, tokenizer and model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if 'gpt2' in args.model_name_or_path.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
    elif 't5' in args.model_name_or_path.lower():
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    else:
        raise ValueError("Unsupported model.")
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
def main(
        args_path: pathlib.Path,
        train_path: tuple[pathlib.Path],
        dev_path: tuple[pathlib.Path],
        log_level: int,
        ckpt_path: pathlib.Path,
        orig_train_schema_path: pathlib.Path,
):
    args = OmegaConf.load(args_path)
    set_seed(args.reproduce)
    data_versions = {get_data_version(p) for p in train_path}
    assert len(data_versions) == 1, f"Attempting to train on multiple data versions {data_versions}"
    args.data.version = list(data_versions)[0]
    special_tokens = []
    for p in train_path:
        model_input_config = OmegaConf.load(f"{p.parent.joinpath('preprocessing_config.yaml')}")
        args.data.metadata[str(p)] = model_input_config.metadata
        args.data.preprocessing[str(p)] = model_input_config.preprocessing
        special_tokens.extend(model_input_config.preprocessing.special_tokens)
    args.train.special_tokens = list(set(special_tokens))
    args.train.dst_train_path = [str(p) for p in train_path]  # type: list[str]
    args.dev.dst_dev_path = [str(p) for p in dev_path]  # type: list[str]

    log_dir = Path(args.train.checkpoint_dir).joinpath(
        args.train.experiment_name, f"version_{args.data.version}", 'logs'
    ).resolve()
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True, parents=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            '{}.log'.format(log_dir.joinpath(Path(__file__).stem)),
            mode='a' if ckpt_path else 'w',
        )
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

    logger.info(OmegaConf.to_yaml(args))
    logger.info("Training on: {}".format('GPU' if 'cuda' in DEVICE.type else 'CPU'))
    initial_step = 0 if not ckpt_path else int(ckpt_path.suffix[1:])
    orig_train_schema = load_schema(orig_train_schema_path)
    if orig_train_schema is None:
        raise ValueError(
            "None of the paths provided via --train_data contains the original dataset"
        )
    if ckpt_path:
        # Load from checkpoint
        args.train.checkpoint = str(ckpt_path)
        config, tokenizer, model = load_model(
            args.train,
            device=DEVICE,
            data_parallel=args.train.data_parallel
        )
    else:
        config, tokenizer, model = set_model(
            args.train,
            data_parallel=args.train.data_parallel
        )
    optimizer = AdamW(
        model.parameters(),
        lr=args.train.learning_rate,
        eps=args.train.adam_eps
    )
    train_dataloader = get_dataloader(
        args.train,
        tokenizer,
        args.train.dst_train_path,
        orig_train_schema,
        sampler=RandomSampler,
        data_size=args.train.data_size
    )
    dev_dataloader = get_dataloader(
        args.dev,
        tokenizer,
        args.dev.dst_dev_path,
        orig_train_schema,
        sampler=SequentialSampler,
        data_size=args.dev.data_size
    )
    scheduler = None
    if args.train.use_scheduler:
        t_total = len(train_dataloader) // args.train.gradient_accumulation_steps * args.train.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.train.warmup_steps,
            num_training_steps=t_total
        )
    if ckpt_path:
        optimizer, scheduler = load_optimizer_scheduler(ckpt_path, optimizer, scheduler)
    train(args, tokenizer, model, train_dataloader, dev_dataloader,
          optimizer, scheduler, initial_step=initial_step)


if __name__ == '__main__':
    main()
