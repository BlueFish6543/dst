from __future__ import annotations

import json
import logging
import pathlib
import random
import re
import string
import sys
from typing import List, Optional

import click
from omegaconf import DictConfig, OmegaConf

from dst.metadata.prepocessing import (
    ALPHA_NUMERIC_SLOT_EXCEPTIONS,
    CATEGORICALS_WITH_DONTCARE_VALUE,
)
from dst.utils import get_datetime, infer_schema_variant_from_path, save_data, set_seed

logger = logging.getLogger(__name__)


def has_alphanumeric_words(
    word: str,
) -> bool:
    """Returns `True` if there are words tha contain both digits and other characters
    or are digits"""
    res = re.findall(r"(?:\d+[a-zA-Z]+|[a-zA-Z]+\d+)|\d+", word)
    if res:
        return True
    return False


def value_in_utterance(
    values: List[str], system_utterance: str, user_utterance: str
) -> Optional[str]:
    # Returns the first value in a list of values that is found in system or user utterance
    for value in values:
        if value in system_utterance or value in user_utterance:
            return value
    return


def heuristic_slot_value_selection(
    frame: dict,
    previous_slots: dict[str, dict[str, str]],
    system_utterance: str,
    user_utterance: str,
) -> dict[str, str]:
    """Returns the current values of all slots in the dialogue state, handling
    situations when there are multiple values by:

        - carrying over a slot value if it has been previously mentioned in the dialogue

        - if the value has not been mentioned in the dialogue, it is searched in the current system utterance
        first and the current user utterance

        - if none of the above hold true, the first value in the annotation is selected
    """

    current_slots = {}
    frame_state = frame["state"]
    service = frame["service"]
    # Need to handle case when there are multiple possible values for the slot
    # We pick either the one that was previously in the state, or the one that
    # appears in the system/user utterance, or failing which, the first value
    # in the list
    for slot, values in frame_state["slot_values"].items():
        assert isinstance(values, list)
        if service in previous_slots and slot in previous_slots[service]:
            assert not isinstance(previous_slots[service][slot], list)
        if (
            service in previous_slots
            and slot in previous_slots[service]
            and previous_slots[service][slot] in values
        ):
            current_slots[slot] = previous_slots[service][slot]
        else:
            value = value_in_utterance(values, system_utterance, user_utterance)
            current_slots[slot] = value if value is not None else values[0]
    return current_slots


def concatenated_slot_value_selection(
    frame: dict, value_selection_config: DictConfig
) -> dict[str, str]:
    """Handles multiple values in dialogue state of a given service by concatenating them.

    Parameters
    ----------
    frame
        Data structure containing the dialogue state of a given service.
    value_selection_config
        Options that configure the concatenation formatting. These include:

            - shuffle_before_concat: values are shuffled before being concatenated

            - value_separator: the symbol used to concatenate the values
    """
    frame_state = frame["state"]
    shuffle = value_selection_config.shuffle_before_concat
    value_sep = value_selection_config.value_separator
    return {
        slot: concatenate_values(values, shuffle=shuffle, separator=value_sep)
        for slot, values in frame_state["slot_values"].items()
    }


def concatenate_values(
    values: list[str], shuffle: bool = True, separator: str = "|"
) -> str:
    assert isinstance(values, list) and isinstance(values[0], str)
    if len(values) == 1:
        return values[0]
    if shuffle:
        random.shuffle(values)
    return f"{separator}".join(values)


def linearize_targets(
    frame: dict,
    turn_info: dict,
    previous_slots: dict,
    system_utterance: str,
    user_utterance: str,
    value_selection_config: DictConfig,
    lowercase: bool = False,
    slot_index_separator: Optional[str] = ":",
    dontcare_handling="qa_option",
):
    """
    Create a target sequence from dialogue annotations.

    Parameters
    -----------
    frame
        The frame currently processed, extracted from SGD-corpus
    turn_info
        A dictionary with the following format::

            {
                service_name: str
                {
                    'description': str
                    'slot_mapping': dict[Union[int, str], Union[str, int]]
                    'intent_mapping': dict[Union[int, str], Union[str, int]]
                    'cat_mapping': dict[str, dict[str, str]]

                }

            }

            where:

                - `slot_mapping` maps all slot names to slot indices and slot indices to slot names.

                - `cat_mapping` maps the slot name to a mapping of slot values to slot value indices (e.g., 2a, 2b, ...)
                for each categorical slot

                - `intent_mapping` maps the intent names to intent indices and intent indices to indent names
    previous_slots:
        Dialogue state at previous turn. Derived from annotation so that only one value is kept or all values
        are concatenated as described in the options in `value_selection_config`.
    system_utterance, user_utterance
        Last utterances said by SYSTEM and USER.
    value_selection_config
        Configuration options for value selection
    lowercase
        If True, targets are lowercased.
    slot_index_separator
        Symbol separating slot indices and values.
    dontcare_handling:
        If `predict_dontcare` then `dontcare` is included in the target as a string and not as a QA option in the
        description.
    """
    service = frame["service"]
    state = frame["state"]
    targets = "[states] "
    # handle cases where annotations contain multiple values
    if value_selection_config.method == "heuristic":
        current_slots = heuristic_slot_value_selection(
            frame, previous_slots, system_utterance, user_utterance
        )  # type: dict[str, str]
    elif value_selection_config.method == "concatenate":
        current_slots = concatenated_slot_value_selection(
            frame, value_selection_config
        )  # type: dict[str, str]
    else:
        raise ValueError(
            "Unknown argument for value_selection_config! Expected one of 'heuristic' or 'concatenate"
        )
    previous_slots[service] = current_slots
    # Slot values
    slot_mapping = turn_info[service]["slot_mapping"]
    cat_values_mapping = turn_info[service]["cat_values_mapping"]
    this_service_expected_indices = []
    for i in range(len(slot_mapping) // 2):
        slot_name = slot_mapping[i]
        if slot_name not in ALPHA_NUMERIC_SLOT_EXCEPTIONS:
            assert not has_alphanumeric_words(slot_name)
        if slot_name in current_slots:
            if slot_name in cat_values_mapping:
                value_mapping = cat_values_mapping[slot_name]
                current_slot_value = current_slots[slot_name]
                if current_slot_value == "dontcare":
                    if dontcare_handling == "predict_dontcare":
                        assert "dontcare" not in value_mapping.values()
                        targets += f"{i}{slot_index_separator}dontcare "
                        continue
                targets += (
                    f"{i}{slot_index_separator}{value_mapping[current_slot_value]} "
                )
            else:
                # Non-categorical
                targets += f"{i}{slot_index_separator}{current_slots[slot_name]} "
            this_service_expected_indices.append(i)
    turn_info[service]["target_slot_indices"] = this_service_expected_indices

    # Active intent
    targets += "[intents] "
    if state["active_intent"] != "NONE":
        targets += turn_info[service]["intent_mapping"][state["active_intent"]] + " "

    # Requested slots
    targets += "[req_slots] "
    for i in range(len(slot_mapping) // 2):
        slot_name = slot_mapping[i]
        if slot_name in state["requested_slots"]:
            targets += f"{i} "

    # Update
    turn_info[service]["expected_output"] = (
        targets.strip().lower() if lowercase else targets.strip()
    )


def allows_dontcare_value(slot_dict: dict, service: str) -> bool:
    slot_name = slot_dict["name"]
    return all(
        (
            service in CATEGORICALS_WITH_DONTCARE_VALUE
            and slot_name in CATEGORICALS_WITH_DONTCARE_VALUE[service],
            "dontcare" not in slot_dict["possible_values"],
        )
    )


def linearize_description(
    schema: List[dict],
    turn: dict,
    prefix_separators: DictConfig,
    lowercase: bool = False,
    dontcare_handling: str = "qa_option",
) -> dict:
    services = list(sorted([frame["service"] for frame in turn["frames"]]))
    ordered_services = [s["service_name"] for s in schema]
    assert ordered_services == sorted(ordered_services)
    result = {}
    for service in schema:
        if service["service_name"] == services[0]:
            service_name = service["service_name"]
            description = ""
            slot_mapping = {}  # maps slot names to indices and indices to slot names
            cat_values_mapping = {}  # nested mapping of from slot to value to index
            intent_mapping = (
                {}
            )  # maps intent names to indices and indices to intent names

            random.shuffle(service["slots"])
            for i, slot in enumerate(service["slots"]):
                slot_name = slot["name"]
                slot_description = slot["description"]
                if slot["is_categorical"]:
                    separator = prefix_separators.categorical_slots
                else:
                    separator = prefix_separators.noncategorical_slots
                description += f"{i}{separator}{slot_description} "
                slot_mapping[slot_name] = i
                slot_mapping[i] = slot_name

                if slot["is_categorical"]:
                    # append dontcare to descriptions of categorical slots if this value is possible
                    cat_values_mapping[slot_name] = {}  # value to index
                    random.shuffle(slot["possible_values"])
                    if dontcare_handling == "qa_option" and allows_dontcare_value(
                        slot, service_name
                    ):
                        slot["possible_values"].append("dontcare")
                        random.shuffle(slot["possible_values"])
                    assert len(slot["possible_values"]) == len(
                        set(slot["possible_values"])
                    )
                    for index_letter, value in zip(
                        list(string.ascii_lowercase), slot["possible_values"]
                    ):
                        description += f"{i}{index_letter}) {value} "
                        cat_values_mapping[slot_name][value] = f"{i}{index_letter}"

            random.shuffle(service["intents"])
            for i, intent in enumerate(service["intents"], 1):
                intent_name = intent["name"]
                intent_description = intent["description"]
                description += f"i{i}{prefix_separators.intents}{intent_description} "
                intent_mapping[intent_name] = f"i{i}"
                intent_mapping[f"i{i}"] = intent_name

            result[service_name] = {
                "description": description.strip().lower()
                if lowercase
                else description.strip(),
                "slot_mapping": slot_mapping,
                "cat_values_mapping": cat_values_mapping,
                "intent_mapping": intent_mapping,
            }
            services.pop(0)
            if not services:
                break
    return result


def process_file(
    schema: List[dict],
    raw_dialogues: list,
    config: DictConfig,
    downsample_factor: int = 1,
    downsample_mode="dialogue",
) -> dict:
    result = {}
    lowercase_model_inputs = config.lowercase_model_inputs
    try:
        target_slot_index_separator = config.target_slot_index_separator
    except AttributeError:
        logger.info("Defaulting to : separator for slot indices in targets.")
        target_slot_index_separator = ":"
    logger.info(
        f"Selected separator {target_slot_index_separator} for slot indices in targets"
    )
    for dialogue_idx, dialogue in enumerate(raw_dialogues):
        if downsample_mode == "dialogue":
            if dialogue_idx % downsample_factor != 0:
                continue
        dialogue_id = dialogue["dialogue_id"]
        result[dialogue_id] = []
        system_utterance = ""
        previous_slots = {}
        for turn_idx, turn in enumerate(dialogue["turns"]):
            if downsample_mode == "turn":
                if turn_idx % downsample_factor != 0:
                    continue
            if turn["speaker"] == "SYSTEM":
                system_utterance = turn["utterance"]
            elif turn["speaker"] == "USER":
                turn_info = linearize_description(
                    schema,
                    turn,
                    config.prefix_separators,
                    lowercase=lowercase_model_inputs,
                    dontcare_handling=config.dontcare_handling,
                )
                user_utterance = turn["utterance"]
                for frame in turn["frames"]:
                    # Each frame represents one service
                    linearize_targets(
                        frame,
                        turn_info,
                        previous_slots,
                        system_utterance,
                        user_utterance,
                        config.value_selection,
                        lowercase=config.lowercase_model_targets,
                        slot_index_separator=target_slot_index_separator,
                        dontcare_handling=config.dontcare_handling,
                    )
                result[dialogue_id].append(
                    {
                        "frames": turn_info,
                        "system_utterance": system_utterance.lower()
                        if lowercase_model_inputs
                        else system_utterance,
                        "user_utterance": user_utterance.lower()
                        if lowercase_model_inputs
                        else user_utterance,
                    }
                )

            else:
                raise ValueError(f"Unknown speaker {turn['speaker']}.")
    return result


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to data preprocessing config file.",
)
@click.option(
    "-d",
    "--data_paths",
    "data_paths",
    required=True,
    type=click.Path(exists=True),
    help="Path to one or more raw SGD data directories.",
    multiple=True,
)
@click.option(
    "-o",
    "--output_path",
    "output_path",
    required=True,
    type=click.Path(exists=False),
    help="Directory where processed data is output.",
)
@click.option(
    "-ver",
    "--version",
    "version",
    default=None,
    help="By default, the version is incremented automatically when the pre-processing script"
    " is run. Use this option when you pre-process data with a given data format for different"
    "experiments to avoid version discrepancies and errors while decoding.",
)
@click.option(
    "--override", is_flag=True, default=False, help="Override previous results."
)
@click.option("--train", "split", flag_value="train")
@click.option("--dev", "split", flag_value="dev")
@click.option("--dev_small", "split", flag_value="dev_small")
@click.option("--test", "split", flag_value="test")
def main(
    cfg_path: pathlib.Path,
    log_level: int,
    data_paths: tuple[str],
    output_path: pathlib.Path,
    override: bool,
    split: str,
    version: Optional[int],
):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = OmegaConf.load(cfg_path)
    set_seed(config.reproduce)
    config.metadata.date = get_datetime()
    config.metadata.raw_data_path = [p for p in data_paths]
    config.metadata.split = split
    config.metadata.output_path = output_path
    output_path = pathlib.Path(output_path)
    data_paths = [pathlib.Path(p) for p in data_paths]

    try:
        downsample_factor = config.downsample_factor
    except AttributeError:
        downsample_factor = 1
    try:
        downsample_mode = config.downsample_mode
    except AttributeError:
        downsample_mode = "dialogue"
    assert downsample_mode in [
        "dialogue",
        "turn",
    ], f"Unknown down-sampling mode {downsample_mode}"
    for shard_path in data_paths:
        logger.info(f"Preprocessing split {split}, shard {shard_path}")
        this_shard_data_dir = shard_path.joinpath(split)
        schema_variant = infer_schema_variant_from_path(str(this_shard_data_dir))
        logger.info(f"Inferred schema variant: {schema_variant}")
        config.metadata.schema_variant = schema_variant
        if shard_path.joinpath(f"{split}_generator_config.yaml").exists():
            gen_config = OmegaConf.load(
                shard_path.joinpath(f"{split}_generator_config.yaml")
            )
            config.metadata.generator = gen_config
        with open(this_shard_data_dir.joinpath("schema.json"), "r") as f:
            schema = json.load(f)
        pattern = re.compile(r"dialogues_[0-9]+\.json")
        result = {}
        for file in this_shard_data_dir.iterdir():
            if pattern.match(file.name):
                with open(file, "r") as f:
                    raw_dialogues = json.load(f)
                result.update(
                    process_file(
                        schema,
                        raw_dialogues,
                        config.preprocessing,
                        downsample_factor=downsample_factor,
                        downsample_mode=downsample_mode,
                    )
                )
        save_data(
            result,
            output_path.joinpath(schema_variant, split),
            metadata=config,
            version=version,
            override=override,
        )


if __name__ == "__main__":
    main()
