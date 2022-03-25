import json
import logging
import os
import pathlib
import re
import sys
from distutils.dir_util import copy_tree
from pathlib import Path

import click
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def parse_predicted_string(
        dialogue_id: str,
        turn_index: str,
        predicted_str: str,
        slot_mapping: dict,
        cat_values_mapping: dict,
        intent_mapping: dict,
        context: str
) -> dict:

    state = {
        "slot_values": {},
        "active_intent": "NONE",
        "requested_slots": []
    }
    # Expect [states] 0:value 1:1a ... [intents] i1 [req_slots] 2 ...
    match = re.search(r"\[states](.*)\[intents](.*)\[req_slots](.*)", predicted_str)
    if match is None:
        # String was not in expected format
        logger.warning(f"Could not parse predicted string {predicted_str} in {dialogue_id}_{turn_index}.")
        return state

    # Parse slot values
    if match.group(1).strip():  # if the string is not empty
        substrings = re.compile(r"(?<!^)\s+(?=[0-9]+:)").split(match.group(1).strip())
        skip = 0
        for i, pair in enumerate(substrings):
            if skip > 0:
                skip -= 1
                continue
            pair = pair.strip().split(":", 1)  # slot value pair
            if len(pair) != 2:
                # String was not in expected format
                logger.warning(f"Could not extract slot values in {predicted_str} in {dialogue_id}_{turn_index}.")
                continue
            try:
                slot = slot_mapping[pair[0].strip()]
                if slot in cat_values_mapping:
                    # Categorical
                    success = False
                    # Invert the mapping to get the categorical value
                    for key, value in cat_values_mapping[slot].items():
                        if value == pair[1].strip():
                            state["slot_values"][slot] = [key]
                            success = True
                            break
                    if not success:
                        logger.warning(
                            f"Could not extract categorical value for slot {pair[0].strip()} in "
                            f"{predicted_str} in {dialogue_id}_{turn_index}.")
                else:
                    # Non-categorical
                    # Check if the next slot could potentially be part of the current slot
                    value = pair[1]
                    # TODO: DEPENDING ON INPUT PREPROCESSING SPLIT MULTIPLE VALUES HERE
                    j = i + 1
                    # TODO: COULD THIS ALSO BE FUZZY MATCH? CHECK OUTPUTS TO FIND OUT!
                    while j < len(substrings):
                        # Check if the combined string exists in the context
                        if (value + substrings[j]).replace(" ", "").lower() not in context.replace(" ", "").lower():
                            # Replace spaces to avoid issues with whitespace
                            break
                        value += " " + substrings[j]
                        skip += 1  # skip the next iteration through substring as it was part of the current slot
                    state["slot_values"][slot] = [value.strip()]
                    if value.replace(" ", "").lower() not in context.replace(" ", "").lower():
                        # Replace spaces to avoid issues with whitespace
                        logger.warning(
                            f"Predicted value {value.strip()} for slot {pair[0].strip()} "
                           f"not in context in {dialogue_id}_{turn_index}."
                        )
            except KeyError:
                logger.warning(
                    f"Could not extract slot {pair[0].strip()} in {predicted_str} in {dialogue_id}_{turn_index}.")

    # Parse intent
    intent = match.group(2).strip()
    if intent:
        try:
            state["active_intent"] = intent_mapping[intent]
        except KeyError:
            logger.warning(f"Could not extract intent in {predicted_str} in {dialogue_id}_{turn_index}.")

    # Parse requested slots
    requested = match.group(3).strip().split()
    for index in requested:
        try:
            state["requested_slots"].append(slot_mapping[index.strip()])
        except KeyError:
            logger.warning(
                f"Could not extract requested slot {index.strip()} in {predicted_str} in {dialogue_id}_{turn_index}.")

    return state


def populate_slots(
        predicted_data: dict,
        template_dialogue: dict,
        dialogue_id: str,
        schema: dict,
        model_name: str,
        preprocessed_references: dict
):
    context = ""
    for turn_index in predicted_data:
        # Loop over turns
        if int(turn_index) > 0:
            # Concatenate system utterance
            context += template_dialogue["turns"][int(turn_index) * 2 - 1]["utterance"] + " "
        template_turn = template_dialogue["turns"][int(turn_index) * 2]  # skip system turns
        context += template_turn["utterance"] + " "  # concatenate user utterance
        assert template_turn["speaker"] == "USER"
        ref_proc_turn = preprocessed_references[int(turn_index)]
        for empty_ref_frame in template_turn["frames"]:
            # Loop over frames (services)
            # Get schema from schema.json
            service_name = empty_ref_frame["service"]
            service_schema = None
            for s in schema:
                if s["service_name"] == service_name:
                    service_schema = s
                    break
            assert service_schema is not None
            predicted_str = predicted_data[turn_index][service_name]["predicted_str"]

            # Some checks
            if 'gpt2' in model_name.lower():
                try:
                    # Should contain the dialogue history
                    # We call replace() to avoid issues with extra whitespace
                    assert template_turn["utterance"].replace(" ", "") in predicted_str.replace(" ", "")
                except AssertionError:
                    logger.warning(
                        f"{predicted_str} in {dialogue_id}_{turn_index} does not match user utterance. Skipping.")
                    raise AssertionError
            if "<EOS>" not in predicted_str:
                logger.warning(f"No <EOS> token in {dialogue_id}_{turn_index}. Skipping.")
                continue

            # Extract string between <BOS> and <EOS>
            if 'gpt2' in model_name.lower():
                predicted_str = re.search(r"<BOS>(.*)<EOS>", predicted_str).group(1).strip()
            elif 't5' in model_name.lower():
                predicted_str = re.search(r"(.*)<EOS>", predicted_str).group(1).strip()
            else:
                raise ValueError("Unsupported model.")

            state = parse_predicted_string(
                dialogue_id, turn_index, predicted_str,
                ref_proc_turn["frames"][service_name]["slot_mapping"],
                ref_proc_turn["frames"][service_name]["cat_values_mapping"],
                ref_proc_turn["frames"][service_name]["intent_mapping"],
                context
            )
            # Update
            empty_ref_frame_state = empty_ref_frame["state"]
            empty_ref_frame_state.update(state)


def parse(
        schema: dict,
        predictions: dict,
        belief_states_dir: pathlib.Path,
        preprocessed_references: dict,
        save_to_subdir: str,
):
    """
    Parameters
    ----------
    save_to_subdir
        Processed files saved to a subdirectory located in `belief_states_dir` directory.
        Used to avoid overriding data during testing.
    """
    with open(belief_states_dir.joinpath("experiment_config.yaml"), "r") as f:
        config = OmegaConf.load(f)
    model_name = config.decode.model_name_or_path

    # save to a subdirectory, optionally
    if save_to_subdir:
        if not belief_states_dir.joinpath(save_to_subdir).exists():
            belief_states_dir.joinpath(save_to_subdir).mkdir(exist_ok=True, parents=True)

    # TODO: USE EXPERIMENT CONFIGURATION HERE TO RETRIEVE INFORMATION ABOUT PREPROCESSING
    pattern = re.compile(r"dialogues_[0-9]+\.json")
    for file in belief_states_dir.iterdir():
        if pattern.match(file.name):
            logger.info(f"Parsing file {file}.")
            with open(file, "r") as f:
                dialogue_templates = json.load(f)
            for blank_dialogue in dialogue_templates:
                dialogue_id = blank_dialogue["dialogue_id"]
                try:
                    predicted_data = predictions[dialogue_id]
                except KeyError:
                    logging.warning(f"Could not find dialogue {dialogue_id} in predicted states.")
                    raise KeyError
                populate_slots(
                    predicted_data,
                    blank_dialogue,
                    dialogue_id,
                    schema,
                    model_name,
                    preprocessed_references[dialogue_id]
                    )
            with open(belief_states_dir.joinpath(save_to_subdir, file.name), "w") as f:
                json.dump(dialogue_templates, f, indent=4)


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
    "-s",
    "--schema_path",
    "schema_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute path to the the schema of the split to be parsed",
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
    "--test-data",
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
        dialogue_templates: pathlib.Path,
        test_path: pathlib.Path,
        log_level: int):

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            f'{belief_path.joinpath("parse")}.log',
            mode='w',
        )
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
    assert belief_path.joinpath("belief_states.json").exists(), "Could not find belief state files"
    # Copy templates over first
    copy_tree(str(dialogue_templates), str(belief_path))
    logger.info(f"Parsing {belief_path} directory.")
    with open(belief_path.joinpath("belief_states.json"), "r") as f:
        predictions = json.load(f)
    parse(schema, predictions, belief_path, preprocessed_refs["data"])

if __name__ == '__main__':
    main()
