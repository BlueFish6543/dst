import json
import logging
import operator
import os
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict
from distutils.dir_util import copy_tree
from typing import Optional

from omegaconf import OmegaConf

from src.dst.utils import humanise

logger = logging.getLogger(__name__)


def build_predicted_state(
        predicted_str: str,
        i: int,
        dialogue_id: str,
        separators: dict
) -> dict:
    predicted_state = defaultdict(dict)
    split_by_services = predicted_str.split(separators["service"].strip())
    for split in split_by_services:
        if not split:
            continue  # empty string
        split_by_default_separator = split.split(separators["default"].strip())
        if len(split_by_default_separator) < 2:
            logger.warning(f"Could not split string {split} in {dialogue_id}_{i}. Skipping.")
            continue
        service = split_by_default_separator[0].strip()
        for slot_value_pair in split_by_default_separator[1:]:
            split_slot_value = slot_value_pair.split(separators["slot-value"].strip())
            if len(split_slot_value) != 2:
                logger.warning(f"Could not split string {slot_value_pair} in {dialogue_id}_{i}. Skipping.")
                continue
            predicted_state[service][split_slot_value[0].strip()] = split_slot_value[1].strip()  # slot: value
    return predicted_state


def extract_best_match_service(
        predicted_state: dict,
        service_schema: dict
) -> Optional[str]:
    matches = {
        predicted_service: len(
            set(map(humanise, [slot["name"] for slot in service_schema["slots"]])).intersection(
                set(predicted_state[predicted_service].keys())
            )) for predicted_service in predicted_state.keys()
    }
    best_match_service_list = [key for key, value in sorted(matches.items(), key=operator.itemgetter(1))]
    if not best_match_service_list:
        return None
    return best_match_service_list[-1]


def populate_slot_values(
        predicted_data: dict,
        template_dialogue: dict,
        dialogue_id: str,
        separators: dict,
        schema: dict
):
    for i, predicted_str in enumerate(predicted_data["bs_str"]):
        template_turn = template_dialogue["turns"][i * 2]  # skip system turns
        assert template_turn["speaker"] == "USER"
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
        predicted_str = re.search(r"<BOS>(.*)<EOS>", predicted_str).group(1)
        # Accumulate services and slots
        predicted_state = build_predicted_state(predicted_str, i, dialogue_id, separators)

        for frame in template_turn["frames"]:
            # Get schema from schema.json
            service = frame["service"]
            service_schema = None
            for s in schema:
                if s["service_name"] == service:
                    service_schema = s
                    break
            assert service_schema is not None

            # Find the best matching service
            best_match_service = extract_best_match_service(predicted_state, service_schema)
            if best_match_service is None:
                continue
            # Populate slots
            for slot_name in [slot["name"] for slot in service_schema["slots"]]:
                humanised_name = humanise(slot_name)
                if humanised_name in predicted_state[best_match_service]:
                    # Add to frame
                    frame["state"]["slot_values"][slot_name] = [predicted_state[best_match_service][humanised_name]]


def parse(
        schema: dict,
        predictions: dict,
        root: str,
):
    # Load separators
    with open(os.path.join(root, "experiment_config.yaml"), "r") as f:
        config = OmegaConf.load(f)
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
                    populate_slot_values(data, dialogue, dialogue_id, separators, schema)
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
