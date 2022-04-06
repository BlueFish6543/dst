from __future__ import annotations

import json
import logging
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import click
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from dst.dataset import BatchedTestDataset
from dst.utils import (
    get_datetime,
    infer_data_version_from_path,
    infer_schema_variant_from_path,
    load_model,
    load_schema,
    set_seed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def remove_padding(output_strings: list[str], pad_token: str) -> list(str):
    """When doing batched decoding, shorter sequences are padded. This
    function removes the padding from the output sequences.

    Note:
    ----
    The addition of `pad_token` at the end of each sequence is specific to the T5
    model where the pad symbol is <EOS>. The T5 parser requires an <EOS> symbol at the
    end of the sequence, which is why it is added to every string.
    """
    padding_free = []
    for s in output_strings:
        pad_token_start = s.find(pad_token)
        while pad_token_start != -1:
            s = f"{s[:pad_token_start]}{s[pad_token_start+len(pad_token):].lstrip()}"
            pad_token_start = s.find(pad_token)
        padding_free.append(f"{s} {pad_token}")
    return padding_free


def decode(args, batch, model, tokenizer) -> list[str]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if args.generate_api == "huggingface":
        output_seqs = model.generate(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
            max_length=args.decoder_max_seq_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    else:
        raise ValueError(
            f"Unknown generation API: {args.generate_API}. Only `huggingface' option is valid for batched decoding."
        )
    output_strings = tokenizer.batch_decode(output_seqs)
    return remove_padding(output_strings, tokenizer.pad_token)


def test(args, tokenizer, model):
    train_schema = load_schema(args.orig_train_schema_path)
    dataset = BatchedTestDataset(
        args, tokenizer, args.dst_test_path, args.data_size, train_schema
    )
    sampler = SequentialSampler(dataset)
    test_gen_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=None, collate_fn=dataset.collate_fn
    )
    model.eval()
    collector = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    description = (
        f"Decoding {len(dataset)} batches from split {dataset.split}. "
        f"Schema variants: {dataset.schema_variants}"
    )
    with torch.no_grad():
        iterator = enumerate(
            tqdm(
                test_gen_dataloader,
                desc=description,
                disable=args.verbose.disable_display,
            )
        )
        for step, batch in iterator:
            bs_pred_str_batch = decode(args, batch, model, tokenizer)
            for i, bs_pred_str in enumerate(bs_pred_str_batch):
                schema_variant, dial_turn = batch["example_id"][i].split("_", 1)
                dialogue_id, turn_idx = dial_turn.rsplit("_", 1)
                usr_utterance = batch["user_utterance"][i]
                service = batch["service"][i]
                collector[dialogue_id][turn_idx]["utterance"] = usr_utterance
                collector[dialogue_id][turn_idx][service]["predicted_str"] = bs_pred_str
    return dict(collector)


def decode_checkpoint(
    args, ckpt_path: pathlib.Path, hyp_path: pathlib.Path
) -> Optional[dict]:
    """Runs decoding for a single checkpoint.

    Parameters
    ---------
    args:
        Decoding settings.
    ckpt_path:
        Absolute path to the model binary.
    hyp_path:
        Path of directory where all decoding results are saved.

    Returns
    -------
    belief_states
        A dictionary containing the predicted belief state and the original utterances.
    """
    args.checkpoint = str(ckpt_path)
    # Suffix model name to path so that we can experiment with models
    this_ckpt_hyp_path = hyp_path.joinpath(ckpt_path.name)
    if this_ckpt_hyp_path.exists():
        if not args.override:
            logger.warning(
                f"Cannot override predictions for {this_ckpt_hyp_path}, skipping decoding. "
                f"Use --override flag to achieve this behaviour."
            )
            return
        else:
            logger.warning(f"Overriding predictions for {this_ckpt_hyp_path}")
    else:
        this_ckpt_hyp_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Decoding {str(ckpt_path)}. Saving dialogues and belief states to {hyp_path}"
    )
    _, tokenizer, model = load_model(args, device=DEVICE)
    belief_states = test(args, tokenizer, model)
    with open(this_ckpt_hyp_path.joinpath("belief_states.json"), "w") as f:
        json.dump(belief_states, f, indent=4)
    return belief_states


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-t",
    "--test-data",
    "test_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to testing data.",
)
@click.option(
    "-a",
    "--args",
    "args_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the file with the decoding arguments.",
)
@click.option(
    "-c",
    "--checkpoint",
    "checkpoint",
    type=click.Path(path_type=Path),
    help="Optional. Absolute path to checkpoint. Overrides option under args.decode.checkpoint, at least one must be "
    "specified. See also --all flag.",
)
@click.option(
    "-hyp",
    "--hyp-dir",
    "hyp_dir",
    type=click.Path(path_type=Path),
    help="Dir where hypothesis files are to be saved. "
    "Auto-suffixed with args.decode.experiment_name model checkpoint binary name.",
)
@click.option(
    "--all",
    is_flag=True,
    default=False,
    help="Decode all checkpoints in a folder. -c/--checkpoint must be the dir where all checkpoint folders are stored.",
)
@click.option(
    "-f",
    "--decode-freq",
    "freq",
    type=int,
    default=1,
    help="Subsample the checkpoints to speed up task-oriented evaluation as training progresses.",
)
@click.option(
    "-reverse",
    "--reverse_decode",
    "reverse",
    is_flag=True,
    default=True,
    help="When multiple checkpoints are decoded, it ensures that later models decode first. This option "
    "does not have any effect if the decode_steps is option is specified in the configuration file. "
    "Use that option if you require that checkpoints are decoded in a specific order.",
)
@click.option(
    "--override", is_flag=True, default=False, help="Override previous results."
)
@click.option(
    "-s",
    "--orig_train_schema_path",
    "orig_train_schema_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute original train schema path. Used to determine seen/unseen examples",
)
@click.option("--dev_small", "split", flag_value="dev_small")
@click.option("--dev", "split", flag_value="dev")
@click.option("--test", "split", flag_value="test")
def main(
    args_path: pathlib.Path,
    test_path: pathlib.Path,
    checkpoint: pathlib.Path,
    hyp_dir: pathlib.Path,
    log_level: int,
    all: bool,
    freq: int,
    reverse: bool,
    override: bool,
    orig_train_schema_path: pathlib.Path,
    split: str,
):
    args = OmegaConf.load(args_path)
    set_seed(args.reproduce)
    args = args.decode
    args.orig_train_schema_path = str(orig_train_schema_path)
    args.override = override
    args.reverse = reverse
    experiment = args.experiment_name
    schema_variant_identifier = infer_schema_variant_from_path(str(test_path))
    data_version = infer_data_version_from_path(str(test_path))
    test_data_version = -1
    if data_version:
        test_data_version = int(data_version.split("_")[1])

    try:
        hyp_path = hyp_dir.joinpath(
            experiment,
            schema_variant_identifier,
            split,
            data_version,
        )
    # Retrieve path from args
    except AttributeError:
        if not args.hyp_dir:
            raise ValueError(
                "You must provide a path for hypothesis to be saved via args.decode.hyp_path or -hyp/--hyp-path."
            )
        hyp_path = Path(args.hyp_dir).joinpath(
            experiment,
            schema_variant_identifier,
            split,
            data_version,
        )

    if not hyp_path.exists():
        hyp_path.mkdir(exist_ok=True, parents=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            f"{hyp_path.joinpath(Path(__file__).stem)}.log",
            mode="w" if not all else "a",
        ),
    ]
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)
    # for compatibility - TestDataset inherits DSTDataset which operates on shards
    args.dst_test_path = [str(test_path)]
    logger.info(OmegaConf.to_yaml(args))
    # Decode all checkpoints in a folder sequentially to save the pain on running many commands
    all_checkpoints = []
    if all:
        if not checkpoint or not checkpoint.exists():
            raise ValueError(
                "Please provide absolute path to the checkpoints directory!"
            )
        else:
            all_checkpoints = [
                f
                for f in checkpoint.glob("model*")
                if "yaml" not in str(f) and "log" not in str(f)
            ]
            # decode later checkpoints first
            all_checkpoints.sort(
                key=lambda x: int(str(x).split(".")[-1]), reverse=False
            )
            if args.decode_steps:
                all_checkpoints_steps = [
                    int(str(el).split(".")[-1]) for el in all_checkpoints
                ]
                filtered_checkpoints = []
                for step in args.decode_steps:
                    try:
                        filtered_checkpoints.append(
                            all_checkpoints[all_checkpoints_steps.index(step)]
                        )
                    except ValueError:
                        logger.warning(
                            f"Decoding step {step} specified in configuration file but corresponding checkpoint "
                            f"not found amongst all checkpoints: __{all_checkpoints}__"
                        )
                all_checkpoints = filtered_checkpoints
            else:
                all_checkpoints = all_checkpoints[::freq]
            if not args.decode_steps and reverse:
                all_checkpoints = reversed(all_checkpoints)
    else:
        try:
            if checkpoint.exists():
                assert (
                    "model." in checkpoint.name
                ), "Path to model checkpoint should end in /model.*/"
                all_checkpoints = [checkpoint]
        except AttributeError:
            if not args.checkpoint:
                raise ValueError(
                    "Checkpoint abs path not provided and path not specified in args.decode.checkpoint!"
                )
            elif not Path(args.checkpoint).exists():
                raise ValueError(
                    f"No checkpoint exists at {args.checkpoint}. Make sure you provide absolute path!"
                )
            else:
                checkpoint = Path(args.checkpoint)
                assert (
                    "model." in checkpoint.name
                ), "Path to model checkpoint should end in /model.*/"
                all_checkpoints = [checkpoint]
    logger.info(f"Decoding {len(all_checkpoints)} checkpoints")
    # Decode checkpoints sequentially
    for ckpt_number, checkpoint in enumerate(all_checkpoints):
        logger.info(f"Decoding checkpoint {ckpt_number}...")
        model_config = OmegaConf.load(checkpoint.parent.joinpath("model_config.yaml"))
        config_data_version = model_config.data.version
        inferred_checkpoint_data_version = infer_data_version_from_path(str(checkpoint))
        inferred_checkpoint_data_version = int(
            inferred_checkpoint_data_version.split("_")[1]
        )
        data_versions = [
            inferred_checkpoint_data_version,
            config_data_version,
            test_data_version,
        ]
        if len(set(data_versions)) != 1:
            logger.error(
                f"Test data version: {test_data_version}. "
                f"Checkpoint version: {config_data_version}. "
                f"Checkpoint version inferred from path: {inferred_checkpoint_data_version}."
                f"Decoding aborted!"
            )
            raise ValueError("Incorrect data version!")
        belief_states = decode_checkpoint(args, checkpoint, hyp_path)
        if belief_states:
            decode_config = OmegaConf.create()
            decode_config.decode = args
            decode_config.date = get_datetime()
            OmegaConf.save(
                OmegaConf.merge(decode_config, model_config),
                f=hyp_path.joinpath(checkpoint.name, "experiment_config.yaml"),
            )
    logger.info("Done decoding!")


if __name__ == "__main__":
    main()
