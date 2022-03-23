from __future__ import annotations
import json
import logging
import pathlib
import random
import sys

import click

import re
import string
from typing import Optional, List

from omegaconf import OmegaConf, DictConfig

from src.dst.utils import infer_schema_variant_from_path, save_data, get_datetime

logger = logging.getLogger(__name__)

CATEGORICALS_WITH_DONTCARE_VALUE = {
    "Banks_1": ["recipient_account_type"],
    "Banks_11": ["recipient_account_type"],
    "Banks_12": ["recipient_account_type"],
    "Banks_13": ["recipient_account_type"],
    "Banks_14": ["recipient_account_type"],
    "Banks_15": ["recipient_account_type"],
    "Banks_2": ["recipient_account_type"],
    "Banks_21": ["recipient_account_type"],
    "Banks_22": ["recipient_account_type"],
    "Banks_23": ["recipient_account_type"],
    "Banks_24": ["recipient_account_type"],
    "Banks_25": ["recipient_account_type"],
    "Buses_2": ["fare_type"],
    "Buses_21": ["fare_type"],
    "Buses_22": ["fare_type"],
    "Buses_23": ["fare_type"],
    "Buses_24": ["fare_type"],
    "Buses_25": ["fare_type"],
    "Buses_3": ["category"],
    "Buses_31": ["category"] + ["stops"],
    "Buses_32": ["category"] + ["route_stops"],
    "Buses_33": ["category"] + ["total_stops"],
    "Buses_34": ["category"] + ["bus_route_stops"],
    "Buses_35": ["category"] + ["number_of_stops"],
    "Flights_1": ["airlines", "seating_class"],
    "Flights_11": ["airlines", "seating_class"],
    "Flights_12": ["airlines", "seating_class"],
    "Flights_13": ["airlines", "seating_class"],
    "Flights_14": ["airlines", "seating_class"],
    "Flights_15": ["airlines", "seating_class"],
    "Flights_2": ["airlines", "seating_class"],
    "Flights_21": ["airlines", "seating_class"],
    "Flights_22": ["airlines", "seating_class"],
    "Flights_23": ["airlines", "seating_class"],
    "Flights_24": ["airlines", "seating_class"],
    "Flights_25": ["airlines", "seating_class"],
    "Flights_3": ["airlines", "flight_class"],
    "Flights_31": ["airlines", "flight_class"],
    "Flights_32": ["airlines", "flight_class"],
    "Flights_33": ["airlines", "flight_class"],
    "Flights_34": ["airlines", "flight_class"],
    "Flights_35": ["airlines", "flight_class"],
    "Flights_4": ["airlines", "seating_class"],
    "Flights_41": ["airlines", "seating_class"] + ["airline", "seat_class"],
    "Flights_42": ["airlines", "seating_class"] + ['airline_name', 'seating_choices'],
    "Flights_43": ["airlines", "seating_class"] + ["company", "seat_choice"],
    "Flights_44": ["airlines", "seating_class"] + ["company_name", "cabin_seat_class"],
    "Flights_45": ["airlines", "seating_class"] + ["air_transport_services_company_name", "seat_option"],
    "Media_2": ["subtitle_language"],
    "Media_21": ["subtitle_language"],
    "Media_22": ["subtitle_language"],
    "Media_23": ["subtitle_language"],
    "Media_24": ["subtitle_language"],
    "Media_25": ["subtitle_language"],
    "Media_3": ["subtitle_language"],
    "Media_31": ["subtitle_language"] + ['movie_subtitle_language'],
    "Media_32": ["subtitle_language"] + ['subtitles'],
    "Media_33": ["subtitle_language"] + ["language"],
    "Media_34": ["subtitle_language"] + ["closed_caption_language"],
    "Media_35": ["subtitle_language"] + ["language_of_subtitles"],
    "Movies_1": ["show_type"],
    "Movies_11": ["show_type"] + ["movie_type"],
    "Movies_12": ["show_type"] + ["visuals_type"],
    "Movies_13": ["show_type"] + ["show_category"],
    "Movies_14": ["show_type"] + ["presentation_type"],
    "Movies_15": ["show_type"] + ["type_of_show"],
    "Music_1": ["playback_device"],
    "Music_11": ["playback_device"],
    "Music_12": ["playback_device"],
    "Music_13": ["playback_device"],
    "Music_14": ["playback_device"],
    "Music_15": ["playback_device"],
    "Music_2": ["playback_device"],
    "Music_21": ["playback_device"],
    "Music_22": ["playback_device"],
    "Music_23": ["playback_device"],
    "Music_24": ["playback_device"],
    "Music_25": ["playback_device"],
    "Music_3": ["device"],
    "Music_31": ["device"] + ["playback_device"],
    "Music_32": ["device"] + ["media_player"],
    "Music_33": ["device"] + ['media_player_name'],
    "Music_34": ["device"] + ["playback_location"],
    "Music_35": ["device"] + ["name_of_media_player"],
    "RentalCars_1": ["type"],
    "RentalCars_11": ["type"],
    "RentalCars_12": ["type"],
    "RentalCars_13": ["type"],
    "RentalCars_14": ["type"],
    "RentalCars_15": ["type"],
    "RentalCars_2": ["car_type"],
    "RentalCars_21": ["car_type"],
    "RentalCars_22": ["car_type"],
    "RentalCars_23": ["car_type"],
    "RentalCars_24": ["car_type"],
    "RentalCars_25": ["car_type"],
    "RentalCars_3": ["car_type"],
    "RentalCars_31": ["car_type"] + ["car_style"],
    "RentalCars_32": ["car_type"] + ['auto_type'],
    "RentalCars_33": ["car_type"] + ["vehicle_type"],
    "RentalCars_34": ["car_type"] + ["type_of_car"] ,
    "RentalCars_35": ["car_type"] + ["model_of_car"],
    "Restaurants_1": ["price_range"],
    "Restaurants_11": ["price_range"],
    "Restaurants_12": ["price_range"],
    "Restaurants_13": ["price_range"],
    "Restaurants_14": ["price_range"],
    "Restaurants_15": ["price_range"],
    "Restaurants_2": ["price_range"],
    "Restaurants_21": ["price_range"] + ["cost_range"],
    "Restaurants_22": ["price_range"] + ["estimated_price_range"],
    "Restaurants_23": ["price_range"] + ["restaurant_price_range"],
    "Restaurants_24": ["price_range"] + ["restaurant_prices"],
    "Restaurants_25": ["price_range"] + ["restaurant_range_of_price"],
    "Trains_1": ["class"],
    "Trains_11": ["class"] + ['fare_class'],
    "Trains_12": ["class"] + ['train_fare_class'],
    "Trains_13": ["class"] + ["class_of_reservation"],
    "Trains_14": ["class"] + ["reservation_fare_class"],
    "Trains_15": ["class"] + ["fare_class_for_reservation"],
    "Travel_1": ["category"],
    "Travel_11": ["category"] + ["type"],
    "Travel_12": ["category"] + ["attraction_category"],
    "Travel_13": ["category"] + ["attraction_type"],
    "Travel_14": ["category"] + ["category_of_attraction"],
    "Travel_15": ["category"] + ["type_of_attraction"],
}


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
        values: List[str],
        system_utterance: str,
        user_utterance: str
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
        if service in previous_slots and slot in previous_slots[service] and \
                previous_slots[service][slot] in values:
            current_slots[slot] = previous_slots[service][slot]
        else:
            value = value_in_utterance(values, system_utterance, user_utterance)
            current_slots[slot] = value if value is not None else values[0]
    return current_slots


def concatenated_slot_value_selection(frame: dict, value_selection_config: DictConfig) -> dict[str, str]:
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


def concatenate_values(values: list[str], shuffle: bool = True, separator: str = "|") -> str:
    assert isinstance(values, list) and isinstance(values[0], str)
    if len(values) == 1:
        return values[0]
    if shuffle:
        random.shuffle(values)
    return f"{separator}".join(values)


def process_frame(
        frame: dict,
        turn_info: dict,
        previous_slots: dict,
        system_utterance: str,
        user_utterance: str,
        value_selection_config: DictConfig,
):
    service = frame["service"]
    state = frame["state"]
    targets = "[states] "

    # handle cases where annotations contain multiple values
    if value_selection_config.method == 'heuristic':
        current_slots = heuristic_slot_value_selection(
            frame, previous_slots, system_utterance, user_utterance
        )
    elif value_selection_config.method == 'concatenate':
        current_slots = concatenated_slot_value_selection(
            frame, value_selection_config
        )
    else:
        raise ValueError(
            "Unknown argument for value_selection_config! Expected one of 'heuristic' or 'concatenate"
         )
    previous_slots[service] = current_slots

    # Slot values
    slot_mapping = turn_info[service]["slot_mapping"]
    cat_values_mapping = turn_info[service]["cat_values_mapping"]
    for i in range(len(slot_mapping) // 2):
        slot = slot_mapping[i]
        assert not has_alphanumeric_words(slot)
        if slot in current_slots:
            # Active
            if slot in cat_values_mapping:
                # Categorical
                # if current_slots[slot] == "dontcare":
                #     # "dontcare" is not in the schema
                #     expected_output += f"{i}:dontcare "
                # else:
                #     expected_output += f"{i}:{cat_values_mapping[slot][current_slots[slot]]} "
                targets += f"{i}:{cat_values_mapping[slot][current_slots[slot]]} "
            else:
                # Non-categorical
                targets += f"{i}:{current_slots[slot]} "

    # Active intent
    targets += "[intents] "
    if state["active_intent"] != "NONE":
        targets += turn_info[service]["intent_mapping"][state["active_intent"]] + " "

    # Requested slots
    targets += "[req_slots] "
    for i in range(len(slot_mapping) // 2):
        slot = slot_mapping[i]
        if slot in state["requested_slots"]:
            targets += f"{i} "

    # Update
    turn_info[service]["expected_output"] = targets.strip()


def generate_description(
        schema: List[dict],
        turn: dict,
        prefix_separators: DictConfig,
        lowercase: bool = False
)-> dict:
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
            intent_mapping = {}  # maps intent names to indices and indices to intent names

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
                    if service_name in CATEGORICALS_WITH_DONTCARE_VALUE \
                            and slot_name in CATEGORICALS_WITH_DONTCARE_VALUE[service_name] and \
                            "dontcare" not in slot["possible_values"]:
                        slot["possible_values"].append("dontcare")
                    assert len(slot["possible_values"]) == len(set(slot["possible_values"]))
                    random.shuffle(slot["possible_values"])
                    cat_values_mapping[slot_name] = {}  # value to index
                    for index_letter, value in zip(list(string.ascii_lowercase), slot["possible_values"]):
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
                "description": description.strip().lower() if lowercase else description.strip(),
                "slot_mapping": slot_mapping,
                "cat_values_mapping": cat_values_mapping,
                "intent_mapping": intent_mapping
            }
            services.pop(0)
            if not services:
                break
    return result


def process_file(schema: List[dict], data: list, config: DictConfig) -> dict:
    result = {}
    lowercase_model_inputs=config.lowercase_model_inputs
    for dialogue in data:
        dialogue_id = dialogue["dialogue_id"]
        result[dialogue_id] = []
        system_utterance = ""
        previous_slots = {}
        for turn in dialogue["turns"]:
            if turn["speaker"] == "SYSTEM":
                system_utterance = turn["utterance"]
            elif turn["speaker"] == "USER":
                turn_info = generate_description(
                    schema,
                    turn,
                    config.prefix_separators,
                    lowercase=lowercase_model_inputs
                )
                user_utterance = turn["utterance"]
                for frame in turn["frames"]:
                    # Each frame represents one service
                    process_frame(
                        frame,
                        turn_info,
                        previous_slots,
                        system_utterance,
                        user_utterance,
                        config.value_selection,
                        config.lowercase_model_targets
                    )
                result[dialogue_id].append({
                    "frames": turn_info,
                    "system_utterance": system_utterance.lower() if lowercase_model_inputs else system_utterance,
                    "user_utterance": user_utterance.lower() if lowercase_model_inputs else user_utterance,
                })

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
    help="path to config file",
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
@click.option('--train', 'split', flag_value='train')
@click.option('--dev', 'split', flag_value='dev')
@click.option('--dev_small', 'split', flag_value='dev_small')
@click.option('--test', 'split', flag_value='test')
def main(
        cfg_path: pathlib.Path,
        log_level: int,
        data_paths: tuple[str],
        output_path: pathlib.Path,
        split: str,
):

    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = OmegaConf.load(cfg_path)
    config.metadata.date = get_datetime()
    config.metadata.raw_data_path = [p for p in data_paths]
    config.metadata.split = split
    config.metadata.output_path = output_path
    output_path = pathlib.Path(output_path)
    data_paths = [pathlib.Path(p) for p in data_paths]
    for shard_path in data_paths:
        logger.info(
            f"Preprocessing split {split}, shard {shard_path}"
        )
        this_shard_data_dir = shard_path.joinpath(split)
        schema_variant = infer_schema_variant_from_path(str(this_shard_data_dir))
        logger.info(
            f"Inferred schema variant: {schema_variant}"
        )
        config.metadata.schema_variant = schema_variant
        with open(this_shard_data_dir.joinpath("schema.json"), 'r') as f:
            schema = json.load(f)
        pattern = re.compile(r"dialogues_[0-9]+\.json")
        result = {}
        for file in this_shard_data_dir.iterdir():
            if pattern.match(file.name):
                with open(file, "r") as f:
                    data = json.load(f)
                result.update(process_file(schema, data, config.preprocessing))
        save_data(result, output_path.joinpath(schema_variant, split), metadata=config)


if __name__ == '__main__':
    main()
