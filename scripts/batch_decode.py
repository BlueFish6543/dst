from __future__ import annotations

import json
import logging
import pathlib
import sys
from pathlib import Path
from typing import Optional

import click
import torch
from omegaconf import OmegaConf

from dst.dataset import get_inference_data_loader
from dst.inference import run_inference
from dst.utils import (
    get_datetime,
    infer_data_version_from_path,
    infer_schema_variant_from_path,
    load_model,
    set_seed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


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
    data_loader = get_inference_data_loader(args, tokenizer)
    belief_states = run_inference(args, tokenizer, model, data_loader, DEVICE)
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
    type=click.Path(path_type=Path, exists=True),
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
    help="When multiple checkpoints are decoded, it ensures that later models decode first. This option "
    "does not have any effect if the decode_steps is option is specified in the configuration file. "
    "Use that option if you require that checkpoints are decoded in a specific order.",
)
@click.option(
    "--override", is_flag=True, default=False, help="Override previous results."
)
@click.option(
    "-compat",
    "--compatible_versions",
    "compatible_versions",
    required=True,
    type=int,
    help="Specify which test data versions are compatible with ",
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
    "-scheme",
    "--ensemble_dirs",
    "ensemble_dirs",
    required=False,
    type=str,
    default="",
    help="Used when inference with the same model is performed with different inputs, to avoid overwriting "
    "inference results.",
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
    compatible_versions: tuple[int],
    orig_train_schema_path: pathlib.Path,
    decoding_scheme: str,
    split: str,
):
    args = OmegaConf.load(args_path)
    set_seed(args.reproduce)
    args = args.decode
    args.orig_train_schema_path = str(orig_train_schema_path)
    args.override = override
    args.reverse = reverse
    if all:
        model_config = OmegaConf.load(checkpoint.joinpath("model_config.yaml"))
    else:
        model_config = OmegaConf.load(checkpoint.parent.joinpath("model_config.yaml"))
    experiment = model_config.train.experiment_name
    schema_variant_identifier = infer_schema_variant_from_path(str(test_path))
    data_version = infer_data_version_from_path(str(test_path))
    test_data_version = -1
    if data_version:
        test_data_version = int(data_version.split("_")[1])

    try:
        hyp_path = hyp_dir.joinpath(
            experiment,
            decoding_scheme,
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
            decoding_scheme,
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
    if all:
        all_checkpoints = [
            f
            for f in checkpoint.glob("model*")
            if "yaml" not in str(f) and "log" not in str(f)
        ]
        # decode later checkpoints first
        all_checkpoints.sort(key=lambda x: int(str(x).split(".")[-1]), reverse=False)
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
            all_checkpoints = list(reversed(all_checkpoints))
    else:
        assert (
            "model." in checkpoint.name
        ), "Path to model checkpoint should end in /model.*/"
        all_checkpoints = [checkpoint]
    logger.info(f"Decoding {len(all_checkpoints)} checkpoints")
    # Decode checkpoints sequentially
    for ckpt_number, checkpoint in enumerate(all_checkpoints):
        logger.info(f"Decoding checkpoint {ckpt_number}...")
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
        try:
            assert len(set(data_versions)) == 1
        except AssertionError:
            if (
                inferred_checkpoint_data_version not in compatible_versions
                and test_data_version not in compatible_versions
            ):
                logger.error(
                    f"Test data version: {test_data_version}. "
                    f"Checkpoint version: {config_data_version}. "
                    f"Checkpoint version inferred from path: {inferred_checkpoint_data_version}."
                    f"Decoding aborted!"
                )
                raise ValueError("Incorrect data version!")
            else:
                logger.warning(
                    f"Decoding on compatible version {test_data_version}. Model was trained on {config_data_version}"
                )
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
