import json
import logging
import os
import re
import sys
from argparse import ArgumentParser
from distutils.dir_util import copy_tree

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def parse_predicted_string(
        dialogue_id: str,
        i: str,
        predicted_str: str,
        slot_mapping: dict,
        cat_values_mapping: dict,
        intent_mapping: dict
) -> dict:
    state = {
        "values": {},
        "intent": "NONE",
        "requested": []
    }
    # Expect [states] 0:value 1:1a ... [intents] i1 [req_slots] 2 ...
    match = re.search(r"\[states](.*)\[intents](.*)\[req_slots](.*)", predicted_str)
    if match is None:
        # String was not in expected format
        logger.warning(f"Could not parse predicted string {predicted_str} in {dialogue_id}_{i}.")
        return state

    # Parse slot values
    substrings = [s.strip() for s in match.group(1).split(":")]
    for i in range(len(substrings) - 1):
        left = substrings[i].rsplit(" ", 1)
        right = substrings[i + 1].rsplit(" ", 1)
        if (len(left) != 2 and i > 0) or (len(right) != 2 and i < len(substrings) - 2):
            # String was not in expected format
            logger.warning(f"Could not extract slot values in {predicted_str} in {dialogue_id}_{i}.")
            continue
        try:
            slot = slot_mapping[left[-1].strip()]
            if slot in cat_values_mapping:
                success = False
                # Invert the mapping to get the categorical value
                for key, value in cat_values_mapping[slot].items():
                    if value == right[0].strip():
                        state["values"][slot] = key
                        success = True
                        break
                if not success:
                    logger.warning(f"Could not extract categorical value for slot {left[-1].strip()} in "
                                   f"{predicted_str} in {dialogue_id}_{i}.")
            else:
                # Non-categorical
                state["values"][slot] = right[0].strip()
        except KeyError:
            logger.warning(f"Could not extract slot {left[-1].strip()} in {predicted_str} in {dialogue_id}_{i}.")

    # Parse intent
    intent = match.group(2).strip()
    if intent:
        try:
            state["intent"] = intent_mapping[intent]
        except KeyError:
            logger.warning(f"Could not extract intent in {predicted_str} in {dialogue_id}_{i}.")

    # Parse requested slots
    requested = match.group(3).strip().split()
    for index in requested:
        try:
            state["requested"].append(slot_mapping[index.strip()])
        except KeyError:
            logger.warning(f"Could not extract requested slot {index.strip()} in "
                           f"{predicted_str} in {dialogue_id}_{i}.")

    return state


def populate_slots(
        predicted_data: dict,
        template_dialogue: dict,
        dialogue_id: str,
        schema: dict,
        model_name: str,
        data: dict,
        separators: dict
):
    for i in predicted_data:
        # Loop over turns
        template_turn = template_dialogue["turns"][int(i) * 2]  # skip system turns
        assert template_turn["speaker"] == "USER"
        data_turn = data[int(i)]

        for frame in template_turn["frames"]:
            # Loop over frames (services)
            # Get schema from schema.json
            service_name = frame["service"]
            service_schema = None
            for s in schema:
                if s["service_name"] == service_name:
                    service_schema = s
                    break
            assert service_schema is not None
            predicted_str = predicted_data[i][service_name]

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

            state = parse_predicted_string(
                dialogue_id, i, predicted_str,
                data_turn["frames"][service_name]["slot_mapping"],
                data_turn["frames"][service_name]["cat_values_mapping"],
                data_turn["frames"][service_name]["intent_mapping"]
            )
            # Update
            for slot, value in state["values"]:
                frame["state"]["slot_values"][slot] = [value]
            frame["state"]["active_intent"] = state["intent"]
            for requested in state["requested"]:
                frame["state"]["requested_slots"].append(requested)


def parse(
        schema: dict,
        predictions: dict,
        root: str,
        data: dict,
        separators: dict
):
    with open(os.path.join(root, "experiment_config.yaml"), "r") as f:
        config = OmegaConf.load(f)
        model_name = config.decode.model_name_or_path

    pattern = re.compile(r"dialogues_[0-9]+\.json")
    for file in os.listdir(root):
        if pattern.match(file):
            logger.info(f"Parsing file {file}.")
            with open(os.path.join(root, file), "r") as f:
                dialogues = json.load(f)
                for dialogue in dialogues:
                    dialogue_id = dialogue["dialogue_id"]
                    try:
                        predicted_data = predictions[dialogue_id]
                    except KeyError:
                        logging.warning(f"Could not find dialogue {dialogue_id} in predicted states.")
                        continue
                    populate_slots(predicted_data, dialogue, dialogue_id, schema, model_name,
                                   data[dialogue_id], separators)
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
    parser.add_argument("-j", "--json", required=True,
                        help="Path to JSON file containing test data")
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
    with open(args.json, "r") as f:
        dataset = json.load(f)
        data = dataset["data"]
        separators = dataset["separators"]

    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file == "belief_states.json":
                # Copy templates over first
                copy_tree(args.template, root)
                logger.info(f"Parsing {root} directory.")
                with open(os.path.join(root, file), "r") as f:
                    predictions = json.load(f)
                    parse(schema, predictions, root, data, separators)


if __name__ == '__main__':
    main()
