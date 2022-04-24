from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import re
from collections import defaultdict
from datetime import datetime
from operator import methodcaller
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    import importlib_resources
except (ImportError, ModuleNotFoundError):
    pass

logger = logging.getLogger(__name__)

_EXPECTED_SCHEMA_VARIANTS = ["v1", "v2", "v3", "v4", "v5"]
_EXPECTED_SPLITS = ["train", "test", "dev", "dev_small"]
_DATA_PACKAGE = "data.raw"


def set_seed(args):
    # For reproduction
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn.enabled
    torch.backends.cudnn.enabled = args.cudnn.deterministic
    torch.backends.cudnn.benchmark = args.cudnn.benchmark


def save_checkpoint(
    args,
    tokenizer,
    model,
    step_or_identifier: Union[str, int],
    optimizer,
    scheduler,
    global_step,
    dev_jga: float = 0.0,
    patience: int = 0,
):
    ckpt_path = Path(args.train.checkpoint_dir)
    ckpt_path = ckpt_path.joinpath(
        args.train.experiment_name, f"version_{args.data.version}"
    )
    if not ckpt_path.exists():
        ckpt_path.mkdir(exist_ok=True, parents=True)
    save_path = f"{ckpt_path}/model.{step_or_identifier}"
    logger.info(f"Save model in {save_path}!")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    OmegaConf.save(args, f"{ckpt_path}/model_config.yaml")
    state_dict = {"optimizer_state_dict": optimizer.state_dict()}
    state_dict["dev_jga"] = dev_jga
    state_dict["dev_jga_patience"] = patience
    state_dict["global_step"] = global_step
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
        state_dict["scheduler_last_epoch"] = scheduler.last_epoch
    torch.save(state_dict, os.path.join(save_path, "checkpoint.pth"))


def load_model(
    args: DictConfig, device: Union[torch.device, str], data_parallel: bool = False
):
    ckpt_path = args.checkpoint
    logger.info(f"Load model, tokenizer from {ckpt_path}")
    tokenizer = T5Tokenizer.from_pretrained(ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
    if data_parallel:
        model = nn.DataParallel(model)
    model.to(device)
    return model.config, tokenizer, model


def load_optimizer_scheduler(ckpt_path: str, optimizer, scheduler):
    checkpoint = torch.load(os.path.join(ckpt_path, "checkpoint.pth"))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        try:
            scheduler.last_epoch = checkpoint["scheduler_last_epoch"]
        except KeyError:
            logger.warning("Could not find key scheduler_last_epoch in state dict")
    try:
        metrics = {
            "dev_jga": checkpoint["dev_jga"],
            "dev_jga_patience": checkpoint["dev_jga_patience"],
        }
        global_step = checkpoint["global_step"]
    except KeyError:
        metrics = {}
        if "global_step" not in checkpoint:
            logger.warning(
                "Could not find global_step in state dictionary. Defaulting to 0."
            )
        global_step = 0
    return optimizer, scheduler, metrics, global_step


def humanise(name: str, remove_trailing_numbers: bool = False) -> str:
    # Convert a potentially camel or snake case string to a lower case string delimited by spaces
    # Adapted from https://stackoverflow.com/a/1176023
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower().replace("_", " ")
    if remove_trailing_numbers:
        # Remove trailing numbers
        name = re.sub("[0-9]+$", "", name)
    return re.sub(" +", " ", name).strip()


class ServiceSchema(object):
    """A wrapper for schema for a service. Adapted from `here`_.


    .. here
       https://github.com/google-research/google-research/blob/master/schema_guided_dst/schema.py
    """

    def __init__(self, schema_json):
        self._service_name = schema_json["service_name"]
        self._description = schema_json["description"]
        self._schema_json = schema_json

        self._intents = sorted(i["name"] for i in schema_json["intents"])
        self._slots = sorted(s["name"] for s in schema_json["slots"])
        self._categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if s["is_categorical"] and s["name"] in self.state_slots
        )
        self._all_categorical_slots = sorted(
            s["name"] for s in schema_json["slots"] if s["is_categorical"]
        )
        self._non_categorical_slots = sorted(
            s["name"]
            for s in schema_json["slots"]
            if not s["is_categorical"] and s["name"] in self.state_slots
        )
        slot_schemas = {s["name"]: s for s in schema_json["slots"]}
        categorical_slot_values = {}
        for slot in self._categorical_slots:
            slot_schema = slot_schemas[slot]
            values = sorted(slot_schema["possible_values"])
            categorical_slot_values[slot] = values
        self._categorical_slot_values = categorical_slot_values

    @property
    def schema_json(self):
        return self._schema_json

    @property
    def state_slots(self):
        """Set of slots which are permitted to be in the dialogue state."""
        state_slots = set()
        for intent in self._schema_json["intents"]:
            state_slots.update(intent["required_slots"])
            state_slots.update(intent["optional_slots"])
        return state_slots

    @property
    def service_name(self):
        return self._service_name

    @property
    def description(self):
        return self._description

    @property
    def slots(self):
        return self._slots

    @property
    def intents(self):
        return self._intents

    @property
    def categorical_slots(self):
        return self._categorical_slots

    @property
    def all_categorical_slots(self):
        return self._all_categorical_slots

    @property
    def non_categorical_slots(self):
        return self._non_categorical_slots

    def get_categorical_slot_values(self, slot):
        return self._categorical_slot_values[slot]


class Schema(object):
    """Wrapper for schemas for all services in a dataset. Adapted from the original `Google code_`.

    .. Google code
       https://github.com/google-research/google-research/blob/master/schema_guided_dst/schema.py
    """

    def __init__(self, schema_json_path):
        # Load the schema from the json file.
        with open(schema_json_path, "r") as f:
            schemas = json.load(f)
        self._services = sorted(schema["service_name"] for schema in schemas)
        service_schemas = {}
        for schema in schemas:
            service = schema["service_name"]
            service_schemas[service] = ServiceSchema(schema)
        self._service_schemas = service_schemas
        self._schemas = schemas

    def get_service_schema(self, service):
        return self._service_schemas[service]

    @property
    def services(self):
        return self._services

    @property
    def service_descriptions(self):
        return self._service_descriptions

    def save_to_file(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self._schemas, f, indent=4)


def load_schema(data_path: Union[Path, str]) -> Schema:
    return Schema(data_path)


def infer_schema_variant_from_path(path: str) -> str:
    """Extracts the schema version from the data path."""
    match = re.search(r"\bv[1-9]\b", path)  # noqa
    if match is not None:
        schema_version = path[match.start() : match.end()]
        assert schema_version in _EXPECTED_SCHEMA_VARIANTS
    else:
        schema_version = "original"
    return schema_version


def get_data_version(path: pathlib.Path) -> int:
    for elem in path.parts:
        if "version" in elem:
            return int(elem.split("_")[-1])
    return -1


def infer_split_name_from_path(path: str) -> str:
    """Extract the SGD split from the data path."""
    p = Path(path)
    if get_data_version(p) == -1:
        split = p.parent.name
    else:
        split = p.parent.parent.name
    assert split in _EXPECTED_SPLITS
    return split


def infer_data_version_from_path(path: str) -> str:
    match = re.search(r"\bversion_\d+\b", path)  # noqa
    if match is not None:
        version = path[match.start() : match.end()]
    else:
        logger.warning(f"Could not detect data version in path {path}")
        version = ""
    return version


def save_data(
    data: Union[dict, list],
    path: Union[Path, str],
    metadata: Optional[DictConfig] = None,
    version: Optional[int] = None,
    override: bool = False,
):
    """Saves data along with the configuration that created it.

    Args:
        override:
        version:
    """
    path = Path(path)
    if version is None:
        if path.exists():
            existing_version = sorted(
                [
                    int(p.name.split("_")[1])
                    for p in path.iterdir()
                    if "version" in str(p)
                ]
            )  # type: list[int]
            if existing_version:
                version = existing_version[-1] + 1
            else:
                version = 1
        else:
            version = 1
    path = path.joinpath(f"version_{version}")
    if path.exists():
        if not override:
            logger.warning(
                f"Cannot override predictions for {path}, existing data will not be overwritten. "
                f"Use --override flag to achieve this behaviour."
            )
            return
    path.mkdir(parents=True, exist_ok=True)
    if metadata:
        logger.info(
            f"Saving data processing info at path {path.joinpath('preprocessing_config.yaml')}"
        )
        metadata.metadata.version = version
        OmegaConf.save(config=metadata, f=path.joinpath("preprocessing_config.yaml"))
    logger.info(f"Saving data at path {path.joinpath('data.json')}")
    with open(path.joinpath("data.json"), "w") as f:
        json.dump(data, f, indent=4)


def get_datetime() -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def safeget(dct: dict, *keys: Union[tuple[str], list[str]]):
    """Retrieves the value of one nested key represented in `keys`"""
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct


def aggregate_values(
    mapping: dict, agg_fcn: Literal["mean", "prod"], reduce: bool = True
):
    """Aggregates the values of the input (nested) mapping according to the
    specified aggregation method. This function modifies the input in place.

    Parameters
    ---------
    mapping
        The mapping to be aggregated.
    agg_fcn
        Aggregation function. Only  `mean` or `prod` aggregation supported.
    reduce
        If False, the aggregator will keep the first dimension of the value to be aggregated.


    Example
    -------
    >>> mapping = {'a': {'b': [[1, 2], [3, 4]]}}
    >>> agg_fcn = 'mean'
    >>> aggregate_values(mapping, agg_fcn, reduce=False)
    >>> {'a': {'b': [1.5, 3.5]}}

    """

    for key, value in mapping.items():
        if isinstance(value, dict):
            aggregate_values(mapping[key], agg_fcn, reduce=reduce)
        else:
            if reduce:
                aggregator = methodcaller(agg_fcn, value)
                mapping[key] = aggregator(np)
            else:
                if isinstance(mapping[key], list) and isinstance(mapping[key][0], list):
                    agg_res = []
                    for val in mapping[key]:
                        aggregator = methodcaller(agg_fcn, val)
                        agg_res.append(aggregator(np))
                    mapping[key] = agg_res
                else:
                    aggregator = methodcaller(agg_fcn, value)
                    mapping[key] = aggregator(np)


def default_to_regular(d: defaultdict) -> dict:
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


class PathMapping:

    split_names = ["train", "dev", "test"]

    def __init__(self, data_pckg_or_path: Union[str, pathlib.Path] = _DATA_PACKAGE):
        self.pckg = data_pckg_or_path
        try:
            self.data_root = importlib_resources.files(data_pckg_or_path)
        except (ModuleNotFoundError, NameError):
            if isinstance(data_pckg_or_path, str):
                self.data_root = pathlib.Path(data_pckg_or_path)
        self._all_files = [r for r in self.data_root.iterdir()]
        self.split_paths = self._split_paths()
        self.schema_paths = self._schema_paths()

    def _split_paths(self):
        paths = {}
        for split in PathMapping.split_names:
            r = [f for f in self._all_files if f.name == split]
            if not r:
                continue
            [paths[split]] = r
        return paths

    def _schema_paths(self):
        return {
            split: self.split_paths[split].joinpath("schema.json")
            for split in PathMapping.split_names
            if split in self.split_paths
        }

    def _get_split_path(self, split: Literal["train", "test", "dev"]) -> pathlib.Path:
        return self.split_paths[split]

    def _get_schema_path(self, split: Literal["train", "test", "dev"]) -> pathlib.Path:
        return self.schema_paths[split]

    def __getitem__(self, item):
        if item in PathMapping.split_names:
            return self.split_paths[item]
        else:
            if item != "schema":
                raise ValueError(
                    f"Keys available are schema and {*PathMapping.split_names,}"
                )
            return self.schema_paths
