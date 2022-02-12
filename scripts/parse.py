import json
import logging
import os
import re
import sys
from argparse import ArgumentParser
from distutils.dir_util import copy_tree
from typing import Tuple

from omegaconf import OmegaConf

from src.dst.utils import humanise

logger = logging.getLogger(__name__)


def extract_intent(
        schema: dict,
        predicted_str: str,
        frame: dict
):
    for intent in schema["intents"]:
        if humanise(intent["name"]) == predicted_str:
            frame["state"]["active_intent"] = intent["name"]
            return
    # Default to "NONE"
    frame["state"]["active_intent"] = "NONE"


def parse_predicted_slot_string(
        dialogue_id: str,
        i: str,
        predicted_str: str,
        separators: dict
) -> Tuple[bool, str]:
    pair = separators["pair"].strip()
    default = separators["default"].strip()

    # Expect "requested = true/false <SEP> value = value"
    output = predicted_str.split(default)
    if len(output) != 2:
        # String was not in expected format
        logger.warning(f"Could not parse predicted string {predicted_str} in {dialogue_id}_{i}.")
        # Default to False for requested slots
        return False, ""

    requested = output[0].split(pair)
    if len(requested) != 2 or requested[0].strip() != "requested":
        # String was not in expected format
        logger.warning(f"Could not parse predicted string {predicted_str} in {dialogue_id}_{i}.")
        # Default to False for requested slots
        return False, ""
    requested = requested[1].strip().lower() == "true"

    value = output[1].split(pair)
    if len(value) != 2 or value[0].strip() != "value":
        # String was not in expected format
        logger.warning(f"Could not parse predicted string {predicted_str} in {dialogue_id}_{i}.")
        # Default to False for requested slots
        return False, ""
    value = value[1].strip()

    return requested, value


def populate_slots(
        predicted_data: dict,
        template_dialogue: dict,
        dialogue_id: str,
        schema: dict,
        model_name: str,
        separators: dict
):
    for i in predicted_data:
        # Loop over turns
        template_turn = template_dialogue["turns"][int(i) * 2]  # skip system turns
        assert template_turn["speaker"] == "USER"

        for frame in template_turn["frames"]:
            # Loop over frames (services)
            # Get schema from schema.json
            service = frame["service"]
            service_schema = None
            for s in schema:
                if s["service_name"] == service:
                    service_schema = s
                    break
            assert service_schema is not None
            humanised_service_name = humanise(service)
            service_schema["slots"].append({"name": "*intent*"})  # for active intent prediction

            for slot_name in [slot["name"] for slot in service_schema["slots"]]:
                # Loop over slots
                humanised_slot_name = humanise(slot_name)
                predicted_str = predicted_data[i][humanised_service_name][humanised_slot_name]
                # Some checks
                if 'gpt2' in model_name.lower():
                    try:
                        # Should contain the dialogue history
                        # We call replace() to avoid issues with extra whitespace
                        assert template_turn["utterance"].replace(" ", "") in predicted_str.replace(" ", "")
                    except AssertionError:
                        logger.warning(f"{predicted_str} in {dialogue_id}_{i} does not match user utterance. Skipping.")
                        continue
                if "<EOS>" not in predicted_str:
                    logger.warning(f"No <EOS> token in {dialogue_id}_{i}. Skipping.")
                    continue

                # Extract string between <BOS> and <EOS>
                if 'gpt2' in model_name.lower():
                    predicted_str = re.search(r"<BOS>(.*)<EOS>", predicted_str).group(1).strip()
                elif 't5' in model_name.lower():
                    predicted_str = re.search(r"(.*)<EOS>", predicted_str).group(1).strip()
                else:
                    raise ValueError("Unsupported model.")

                if humanised_slot_name == "*intent*":
                    # Active intent prediction
                    extract_intent(service_schema, predicted_str, frame)
                else:
                    # Requested slots and slot values prediction
                    requested, value = parse_predicted_slot_string(dialogue_id, i, predicted_str, separators)
                    if requested:
                        frame["state"]["requested_slots"].append(slot_name)
                    if value:
                        # Add to frame if not empty string
                        frame["state"]["slot_values"][slot_name] = [value]


def parse(
        schema: dict,
        predictions: dict,
        root: str,
):
    with open(os.path.join(root, "experiment_config.yaml"), "r") as f:
        config = OmegaConf.load(f)
        model_name = config.decode.model_name_or_path
    with open(os.path.join(os.getcwd(), config.decode.dst_test_path), "r") as f:
        separators = json.load(f)["separators"]

    pattern = re.compile(r"dialogues_[0-9]+\.json")
    for file in os.listdir(root):
        if pattern.match(file):
            logger.info(f"Parsing file {file}.")
            with open(os.path.join(root, file), "r") as f:
                dialogues = json.load(f)
                for dialogue in dialogues:
                    dialogue_id = dialogue["dialogue_id"]
                    try:
                        data = predictions[dialogue_id]
                    except KeyError:
                        logging.warning(f"Could not find dialogue {dialogue_id} in predicted states.")
                        continue
                    populate_slots(data, dialogue, dialogue_id, schema, model_name, separators)
            with open(os.path.join(root, file), "w") as f:
                json.dump(dialogues, f, indent=4)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", required=True,
                        help="Directory under which predicted belief states for all model checkpoints "
                             "are located")
    parser.add_argument("-s", "--schema", required=True,
                        help="Path to schema.json file")
    parser.add_argument("-t", "--template", required=True,
                        help="Directory containing blank dialogue templates")
    return parser.parse_args()


def main():
    args = parse_args()
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            '{}.log'.format(os.path.join(args.directory, "parse")),
            mode='w',
        )
    ]
    logging.basicConfig(
        handlers=handlers,
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.DEBUG)
    with open(args.schema, "r") as f:
        schema = json.load(f)

    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file == "belief_states.json":
                # Copy templates over first
                copy_tree(args.template, root)
                logger.info(f"Parsing {root} directory.")
                with open(os.path.join(root, file), "r") as f:
                    predictions = json.load(f)
                    parse(schema, predictions, root)


if __name__ == '__main__':
    main()
