"""A module containing iterators for the SGD corpus and other utilities
for working with this data format."""
from __future__ import annotations

import json
import logging
import pathlib
import re
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Generator, Literal, Optional, Union

try:
    import importlib_resources
except ImportError:
    pass

_DATA_PACKAGE = "data.raw"
_SCHEMA_VARIANTS = ["v1", "v2", "v3", "v4", "v5"]
_EXPECTED_SCHEMA_VARIANTS = ["v1", "v2", "v3", "v4", "v5"]
_EXPECTED_SPLITS = ["train", "test", "dev", "dev_small"]

logger = logging.getLogger(__name__)

SchemaElement = namedtuple("SchemaElement", ["name", "description", "type"])


class PathMapping:

    split_names = ["train", "dev", "test"]

    def __init__(self, data_pckg_or_path: Union[str, pathlib.Path] = _DATA_PACKAGE):
        self.pckg = data_pckg_or_path
        try:
            self.data_root = importlib_resources.files(str(data_pckg_or_path))
        except (ModuleNotFoundError, NameError):
            if isinstance(data_pckg_or_path, str):
                self.data_root = pathlib.Path(data_pckg_or_path)
            else:
                self.data_root = data_pckg_or_path
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


def reconstruct_filename(dial_id: str) -> str:
    """Reconstruct filename from dialogue ID."""

    file_prefix = int(dial_id.split("_")[0])

    if file_prefix in range(10):
        str_file_prefix = f"00{file_prefix}"
    elif file_prefix in range(10, 100):
        str_file_prefix = f"0{file_prefix}"
    else:
        str_file_prefix = f"{file_prefix}"

    return f"dialogues_{str_file_prefix}.json"


def get_file_map(
    dialogue_ids: list[str],
    split: Literal["train", "test", "dev"],
    data_pckg_or_path: str = "data.raw",
) -> dict[pathlib.Path, list[str]]:
    """Returns a map where the keys are file paths and values are lists
    comprising dialogues from `dialogue_ids` that are in the same file.

    dialogue_ids:
        IDs of the dialogues whose paths are to be returned, formated as the schema 'dialogue_id' field.
    split:
        The name of the split whose paths are to be returned.
    data_pckg_or_path:
        The location of the python package where the data is located
    """

    file_map = defaultdict(list)
    path_map = PathMapping(data_pckg_or_path=data_pckg_or_path)
    for id in dialogue_ids:
        # ValueError occurs if dialogue IDs do not match SGD convention
        try:
            fpath = path_map[split].joinpath(reconstruct_filename(id))
        except ValueError:
            found_dialogue = False
        else:
            # for the original SGD data, one can reconstruct the filename
            # from dial ID to load the dialogue
            file_map[fpath].append(id)
            continue
        # in general, just iterate through the file to find a given
        # dialogue
        if not found_dialogue:
            for fpath in path_map[split].iterdir():
                if not fpath.name.startswith("dialogues"):
                    continue
                with open(fpath, "r") as f:
                    dial_bunch = json.load(f)
                for dial in dial_bunch:

                    if dial["dialogue_id"] == id:
                        found_dialogue = True
                        break

                if found_dialogue:
                    break

            if found_dialogue:
                file_map[fpath].append(id)
            else:
                logger.warning(f"Could not find dialogue {id}...")

    return file_map


def get_filepaths(
    split: Literal["train", "test", "dev"], data_pckg_or_path: str = "data.raw"
) -> list[pathlib.Path]:
    """Returns a list of file paths for all dialogue batches in a given split.

    Parameters
    ----------
    split
        The split whose filepaths should be returned.
    data_pckg_or_path
        The package where the data is located.
    """
    path_map = PathMapping(data_pckg_or_path=data_pckg_or_path)
    try:
        fpaths = list(path_map[split].glob("dialogues_*.json"))
        fpaths = sorted(
            fpaths, key=lambda x: int(x.name.replace(".json", "").split("_")[1])
        )
    except KeyError:
        fpaths = list(pathlib.Path(data_pckg_or_path).glob("dialogues_*.json"))
        fpaths = sorted(
            fpaths, key=lambda x: int(x.name.replace(".json", "").split("_")[1])
        )
    if "dialogues_and_metrics.json" in fpaths:
        fpaths.remove("dialogues_and_metrics.json")
    return fpaths


def file_iterator(
    fpath: pathlib.Path, return_only: Optional[set[str]] = None
) -> tuple[str, dict]:
    """
    Iterator through an SGD .json file.

    Parameters
    ----------
    fpath:
        Absolute path to the file.
    return_only
        A set of dialogues to be returned. Specified by dialogue IDs as
        found in the `dialogue_id` file of the schema.
    """

    with open(fpath, "r") as f:
        dial_bunch = json.load(f)

    n_dialogues = len(dial_bunch)
    try:
        max_index = int(dial_bunch[-1]["dialogue_id"].split("_")[1]) + 1
    except IndexError:
        max_index = -100
    missing_dialogues = not (max_index == n_dialogues)

    if return_only:
        if not missing_dialogues:
            for dial_idx in (int(dial_id.split("_")[1]) for dial_id in return_only):
                yield fpath, dial_bunch[dial_idx]
        else:
            returned = set()
            for dial in dial_bunch:
                found_id = dial["dialogue_id"]
                if found_id in return_only:
                    returned.add(found_id)
                    yield fpath, dial
                    if returned == return_only:
                        break
            if returned != return_only:
                logger.warning(f"Could not find dialogues: {return_only - returned}")
    else:
        for dial in dial_bunch:
            yield fpath, dial


def split_iterator(
    split: Literal["train", "dev", "test"],
    return_only: Optional[set[str]] = None,
    data_pckg_or_path: str = _DATA_PACKAGE,
) -> Generator[tuple[pathlib.Path, dict]]:
    """

    Parameters
    ----------
    split
        Split through which to iterate.
    return_only
        Return only certain dialogues, specified by their schema ``dialogue_id`` field.
    data_pckg_or_path
        Package where the data is located.
    """
    # return specified dialogues only
    if return_only:
        fpath_map = get_file_map(
            list(return_only), split, data_pckg_or_path=data_pckg_or_path
        )
        for fpth, dial_ids in fpath_map.items():
            yield from file_iterator(fpth, return_only=set(dial_ids))
    # iterate through all dialogues
    else:
        for fp in get_filepaths(split, data_pckg_or_path=data_pckg_or_path):
            with open(fp, "r") as f:
                dial_bunch = json.load(f)
            for dial in dial_bunch:
                yield fp, dial


def dialogue_iterator(
    dialogue: dict, user: bool = True, system: bool = True
) -> Generator[dict]:
    """Iterate through turns in a dialogue."""

    if (not user) and (not system):
        raise ValueError("At least a speaker needs to be specified!")

    filter = "USER" if not user else "SYSTEM" if not system else ""

    for turn in dialogue["turns"]:
        if filter and turn.get("speaker", "") == filter:
            continue
        else:
            yield turn


def turn_iterator(turn: dict, service: Optional[str] = None) -> dict:
    """Iterate through the frames in a turn."""
    if service is None:
        for frame in turn["frames"]:
            yield frame
    else:
        for frame in turn["frames"]:
            if frame["service"] == service:
                yield frame


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


def load_schema(data_path: Union[Path, str]) -> Schema:
    return Schema(data_path)
