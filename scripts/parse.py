from __future__ import annotations

import json
import logging
import pathlib
import sys
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import Union

import click
from omegaconf import OmegaConf

from dst.parser import parse

logger = logging.getLogger(__name__)


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-b",
    "--belief_path",
    "belief_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute path to the directory containing the belief file to be decoded.",
)
@click.option(
    "-o",
    "--output_dir",
    "output_dir",
    required=False,
    type=click.Path(path_type=Path),
    help="Absolute path to the directory where SGD-formatted dialogues containing predictions"
    "as opposed to annotations are output. If not passed, the dialogues are saved in the "
    "same directory as the parser input, (i.e., -b argument).",
    default=None,
)
@click.option(
    "-s",
    "--schema_path",
    "schema_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute path to the the schema of the data to be parsed.",
)
@click.option(
    "-templates",
    "--dialogue_templates",
    "dialogue_templates",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute to the directory containing blank dialogue files for the split parsed.",
)
@click.option(
    "-t",
    "--test_data",
    "test_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to pre-processed test data for which model predictions are to be parsed."
    "Used to retrieve mappings from indices to slot/intent names which are required to"
    "recover slot names from predicted indices",
)
def main(
    belief_path: pathlib.Path,
    schema_path: pathlib.Path,
    output_dir: Union[pathlib.Path, None],
    dialogue_templates: pathlib.Path,
    test_path: pathlib.Path,
    log_level: int,
):

    if output_dir is None:
        output_dir = belief_path
    else:
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True, parents=True)

    with open(belief_path.joinpath("experiment_config.yaml"), "r") as f:
        experiment_config = OmegaConf.load(f)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            f'{output_dir.joinpath("parse")}.log',
            mode="w",
        ),
    ]
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)
    with open(schema_path, "r") as f:
        schema = json.load(f)
    with open(test_path, "r") as f:
        preprocessed_refs = json.load(f)
    assert belief_path.joinpath(
        "belief_states.json"
    ).exists(), "Could not find belief state files"
    # Copy templates over first
    copy_tree(str(dialogue_templates), str(output_dir))
    logger.info(f"Parsing {belief_path} directory.")
    with open(belief_path.joinpath("belief_states.json"), "r") as f:
        predictions = json.load(f)
    try:
        preprocessed_refs = preprocessed_refs["data"]
    except KeyError:
        pass
    parse(schema, predictions, preprocessed_refs, output_dir, experiment_config)


if __name__ == "__main__":
    main()
