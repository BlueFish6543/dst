from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import partial
from itertools import repeat
from operator import methodcaller
from pathlib import Path
from typing import Any, Callable, Generator, Literal, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


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


def save_data(
    data: Union[dict, list],
    path: Union[Path, str],
    metadata: Optional[DictConfig] = None,
    version: Optional[int] = None,
    override: bool = False,
):
    """Saves data along with the configuration that created it."""
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
    mapping: dict, agg_fcn: Literal["mean", "prod", "sum"], reduce: bool = True
):
    """Aggregates the values of the input (nested) mapping according to the
    specified aggregation method. This function modifies the input in place.

    Parameters
    ---------
    mapping
        The mapping to be aggregated.
    agg_fcn
        Aggregation function. Only  `mean`, `prod`, 'sum' aggregation is supported.
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
                        cts = Counter(val)
                        exclude = [v for v in cts if not isinstance(v, float)]
                        if exclude:
                            not_floats = {
                                c: cts[c] for c in cts if not isinstance(c, float)
                            }
                            logger.warning(f"Ignoring {not_floats}")
                        val = [v for v in val if isinstance(v, float)]
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


def nested_defaultdict(default_factory: Callable, depth: int = 1):
    """Creates a nested default dictionary of arbitrary depth with a specified callable as leaf."""
    if not depth:
        return default_factory()
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


def append_to_values(result: dict, new_data: dict):
    """Recursively appends to the values of `result` the values in
    `new_data` that have the same keys. If the keys in `new_data`
    do not exist in `result`, they are recursively added. The keys of
    `new_data` can be either lists or single float objects that
    are to be appended to existing `result` keys. List concatenation is
    performed in former case.

    Parameters
    ----------
    result
        Mapping whose values are to be extended with corresponding values from
        `new_data_map`
    new_data
        Data with which the values of `result_map` are extended.
    """

    def dict_factory():
        return defaultdict(list)

    for key in new_data:
        # recursively add any new keys to the result mapping
        if key not in result:
            if isinstance(new_data[key], dict):
                result[key] = dict_factory()
                append_to_values(result[key], new_data[key])
            else:
                if isinstance(new_data[key], float):
                    result[key] = [new_data[key]]
                elif isinstance(new_data[key], list):
                    result[key] = [*new_data[key]]
                else:
                    raise ValueError("Unexpected key type.")
        # updated existing values with the value present in `new_data_map`
        else:
            if isinstance(result[key], dict):
                append_to_values(result[key], new_data[key])
            else:
                if isinstance(new_data[key], list):
                    result[key] += new_data[key]
                elif isinstance(new_data[key], float):
                    result[key].append(new_data[key])
                else:
                    raise ValueError("Unexpected key type")


def load_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def to_json(data: Any, path: pathlib.Path, sort_keys: bool = True):
    """Save data to .json format."""

    Path(path.parent).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving json file at path: {path}")
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=sort_keys, indent=4)


def cleanup_files(dir_: pathlib.Path, pattern: re.Pattern):
    """Remove files matching a given pattern."""
    for f in dir_.iterdir():
        if pattern.match(f.name):
            logger.info(f"Removing file: {f}")
            os.remove(f)


def store_data(data, store: defaultdict, store_fields: list[str]):
    """Stores data in a nested default dictionary at a specified location.

    Parameters
    ----------
    data
        Data to be stored. If this is a list and the store location is a list,
        the store location is extended with the new elements. If data is not a list,
        then it is appended to the store location.
    store
        Nested default dictionary where data is to be stored.
    store_fields
        Key where the data is to be stored

    Notes
    -----
    Supported types for `store` leaves are ``int``, ``list`` and ``dict``.
    """

    store_location = safeget(store, *store_fields)
    if isinstance(store_location, list):
        if isinstance(data, list):
            store_location.extend(data)
        else:
            store_location.append(data)
    elif isinstance(store_location, int) or isinstance(store_location, float):
        store_location = safeget(store, *store_fields[:-1])
        if isinstance(data, int) or isinstance(data, float):
            if isinstance(store_location, defaultdict):
                if isinstance(store_location[store_fields[-1]], int) or isinstance(
                    store_location[store_fields[-1]], float
                ):
                    store_location[store_fields[-1]] += data
                else:
                    raise NotImplementedError
    elif isinstance(store_location, dict):
        # assert isinstance(data, dict)
        if isinstance(data, dict):
            store_location.update(data)
        else:
            logger.debug(
                f"Storing data of type {type(data)} at leaf with nested key {store_fields[:-1]}"
            )
            store_location = safeget(store, *store_fields[:-1])
            store_location[store_fields[-1]] = data
            logger.debug(
                f"Stored data at key {store_fields[-1]}. Upstream keys are {store_fields[:-1]}"
            )
    else:
        raise NotImplementedError


def get_paths_to_values(d: dict) -> Generator[list]:
    """Get paths to all values in an arbitrary nested mapping.

    Parameters
    ----------
    d
        Arbitrary nested mapping

    Returns
    -------
    A generator which generates list, where each list is a path to a value in `d`.
    """

    def _paths_generator(d: dict, paths: list):
        if not isinstance(d, dict):
            yield paths
        else:
            yield from [
                j for a, b in d.items() for j in _paths_generator(b, paths + [a])
            ]

    return _paths_generator(d, [])


def average_nested_dicts(
    dicts: list[dict], excluded_keys: Optional[list] = None
) -> dict:
    """Average the values of nested dictionaries with identical structure.

    Parameters
    ----------
    dicts
        Dictionaries to be averaged
    excluded_keys
        Which keys should be ignored. They will not be included in the output dictionary.
    """

    def infer_depth(mapping: dict, depth: int = 1):
        """Calculates depth of `mapping`, assumed to be constant depth."""
        if not mapping:
            return 0
        k = list(mapping.keys())[0]
        if isinstance(mapping[k], dict):
            depth = infer_depth(mapping[k], depth + 1)
        return depth

    if not dicts:
        return None
    # mappings assumed identical, so we create the average dict
    # to be same depth.
    depth = infer_depth(dicts[0])
    average = nested_defaultdict(float, depth=depth)
    for value_path in get_paths_to_values(dicts[0]):
        # check if any of the keys should be excluded
        if excluded_keys is not None and any(
            key in excluded_keys for key in value_path
        ):
            continue
        else:
            this_path_values = [safeget(d, *value_path) for d in dicts]
            store_data(
                sum(this_path_values) / len(this_path_values), average, value_path
            )
    return default_to_regular(average)
