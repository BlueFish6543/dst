from __future__ import annotations

import functools
import json
import logging
import pathlib
import re
from typing import Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

SPECIAL_VALUES = {'dontcare'}
lower_to_schema_case = {
    'true': 'True',
    'false': 'False',
    'economy': 'Economy',
    'economy extra': 'Economy extra',
    'flexible': 'Flexible',
    'music': 'Music',
    'sports': 'Sports',
    'premium economy': "Premium Economy",
    'business': "Business",
    'first class': "First Class",
    "united airlines": "United Airlines",
    "american airlines": "American Airlines",
    "delta airlines": "Delta Airlines",
    "southwest airlines": "Southwest Airlines",
    "alaska airlines": "Alaska Airlines",
    "british airlines": "British Airways",
    "air canada": "Air Canada",
    "air france": "Air France",
    "tv": {
        'Music_1': 'TV',
        'Music_11': 'TV',
        'Music_12': 'TV',
        'Music_13': 'TV',
        'Music_14': 'TV',
        'Music_15': 'TV',
        'Music_2': 'TV',
        'Music_21': 'TV',
        'Music_22': 'TV',
        'Music_23': 'TV',
        'Music_24': 'TV',
        'Music_25': 'TV',
    },
    "kitchen speaker": {
        'Music_1': 'Kitchen speaker',
        'Music_11': 'Kitchen speaker',
        'Music_12': 'Kitchen speaker',
        'Music_13': 'Kitchen speaker',
        'Music_14': 'Kitchen speaker',
        'Music_15': 'Kitchen speaker',
        'Music_2': 'kitchen speaker',
        'Music_21': 'kitchen speaker',
        'Music_22': 'kitchen speaker',
        'Music_23': 'kitchen speaker',
        'Music_24': 'kitchen speaker',
        'Music_25': 'kitchen speaker',
    },
    "bedroom_speaker": {
        'Music_1': "Bedroom speaker",
        'Music_11': "Bedroom speaker",
        'Music_12': "Bedroom speaker",
        'Music_13': "Bedroom speaker",
        'Music_14': "Bedroom speaker",
        'Music_15': "Bedroom speaker",
        'Music_2': "bedroom speaker",
        'Music_21': "bedroom speaker",
        'Music_22': "bedroom speaker",
        'Music_23': "bedroom speaker",
        'Music_24': "bedroom speaker",
        'Music_25': "bedroom speaker",
    },
    "compact": "Compact",
    "standard": "Standard",
    "full-size": "Full-size",
    "pool": "Pool",
    "regular": "Regular",
    "luxury": "Luxury",
    "gynecologist": "Gynecologist",
    "ent specialist": "ENT Specialist",
    "ophthalmologist": "Ophthalmologist",
    "general practitioner": "General Practitioner",
    "dermatologist": "Dermatologist",
    "place of worship": "Place of Worship",
    "theme park": "Theme Park",
    "museum": "Museum",
    "historical landmark": "Historical Landmark",
    "park": "Park",
    "tourist attraction": "Tourist Attraction",
    "sports venue": "Sports Venue",
    "shopping area": "Shopping Area",
    "performing arts venue": "Performing Arts Venue",
    "nature preserve": "Nature Preserve",
    "none": 'None',
    "english": "English",
    "mandarin": "Mandarin",
    "spanish": "Spanish",
    "psychologist": "Psychologist",
    "family counselor": "Family Counselor",
    "psychiatrist": "Psychiatrist",
    "theater": "Theater",
    "south african airways": "South African Airways",
    "lot polish airlines": "LOT Polish Airlines",
    "latam brasil": "LATAM Brasil",
    "hindi": "Hindi",
    "french": "French",
    "living room": "Living room",
    "kitchen": "Kitchen",
    "patio": "Patio",
    "hatchback": "Hatchback",
    "sedan": "Sedan",
    "suv": "SUV",
    "value": "Value",
}


def restore_case(value: str, service: str, restore_categorical_case: bool = True) -> str:
    if not restore_categorical_case or value not in lower_to_schema_case:
        return value
    if value in lower_to_schema_case:
        recased_data = lower_to_schema_case[value]
        if isinstance(recased_data, str):
            return recased_data
        else:
            assert isinstance(recased_data, dict)
            return recased_data[service]


def parse_predicted_string(
        dialogue_id: str,
        turn_index: str,
        service: str,
        predicted_str: str,
        slot_mapping: dict,
        cat_values_mapping: dict,
        intent_mapping: dict,
        context: str,
        value_separator: Optional[str] = None,
        restore_categorical_case: bool = False,
        target_slot_index_separator: Optional[str] = ":"
) -> dict:

    state = {
        "slot_values": {},
        "active_intent": "NONE",
        "requested_slots": []
    }
    recase = functools.partial(restore_case, restore_categorical_case=restore_categorical_case)
    # Expect [states] 0:value 1:1a ... [intents] i1 [req_slots] 2 ...
    match = re.search(r"\[states](.*)\[intents](.*)\[req_slots](.*)", predicted_str)
    if match is None:
        # String was not in expected format
        logger.warning(f"Could not parse predicted string {predicted_str} in {dialogue_id}_{turn_index}.")
        return state

    # Parse slot values
    if match.group(1).strip():  # if the string is not empty
        pattern = rf"(?<!^)\s+(?=[0-9]+{target_slot_index_separator})"
        if value_separator == " || ":
            pattern = fr"(?<!^)\s+(?=[0-9]+{target_slot_index_separator})(?<!\|\| )"
        elif value_separator is not None:
            logger.error(f"State splitting pattern undefined for value separator {value_separator}")
        substrings = re.compile(pattern).split(match.group(1).strip())
        if target_slot_index_separator == ":":
            # updates state  in-place
            parse_with_context(
                state,
                substrings,
                dialogue_id,
                turn_index,
                service,
                predicted_str,
                slot_mapping,
                cat_values_mapping,
                context,
                value_separator=value_separator,
                restore_categorical_case=restore_categorical_case,
                target_slot_index_separator=target_slot_index_separator,
            )
        else:
            raise ValueError(f"Unknown target slot index seprator {target_slot_index_separator}")

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


def parse_with_context(
        state: dict,
        substrings: list[str],
        dialogue_id: str,
        turn_index: str,
        service: str,
        predicted_str: str,
        slot_mapping: dict,
        cat_values_mapping: dict,
        context: str,
        value_separator: Optional[str] = None,
        restore_categorical_case: bool = False,
        target_slot_index_separator: Optional[str] = ":"
):
    """Using ":" to separate slot indices from their values in the target sequence makes parsing difficult because this
    is a symbol that appears in values. Therefore, 1:morning 11:15 is parsed as (1, morning), (11, 15) which is wrong.
    This algorithm assumes the following substrings could be part of the current slot values and uses the context to
    check whether this is the case. This algorithm is __NOT__ correct if the value separator is not `None`.
    """

    if value_separator is not None:
        logger.warning(
            "Parser algorithm is not correct for when multiple values are in the target sequence."
            "Use at your own risk!"
        )
    recase = functools.partial(restore_case, restore_categorical_case=restore_categorical_case)
    skip = 0
    for i, pair in enumerate(substrings):
        if skip > 0:
            skip -= 1
            continue
        pair = pair.strip().split(f"{target_slot_index_separator}", 1)  # slot value pair
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
                for categorical_value, categorical_value_idx in cat_values_mapping[slot].items():
                    if categorical_value_idx == pair[1].strip():
                        recased_value = recase(value=categorical_value, service=service)
                        assert isinstance(recased_value, str)
                        state["slot_values"][slot] = [recased_value]
                        success = True
                        break
                if not success:
                    logger.warning(
                        f"Could not extract categorical value for slot {pair[0].strip()} in "
                        f"{predicted_str} in {dialogue_id}_{turn_index}. "
                        f"Values defined for this slot were {cat_values_mapping[slot]}"
                    )
            else:
                # Non-categorical
                # Check if the next slot could potentially be part of the current slot
                value = pair[1]
                if value_separator is not None and value_separator in value:
                    value_list = value.split(value_separator)
                    value_list = [v.strip() for v in value_list]
                else:
                    value_list = [value.strip()]
                j = i + 1
                while j < len(substrings):
                    # Check if the combined string exists in the context
                    for idx, value in enumerate(value_list):
                        if value_separator is None:
                            possible_continuations = [substrings[j]]
                        elif value_separator in substrings[j]:
                            possible_continuations = substrings[j].split(value_separator)
                        else:
                            possible_continuations = [substrings[j]]
                        for continuation in possible_continuations:
                            # Replace spaces to avoid issues with whitespace
                            if (value + continuation).replace(" ", "").lower() not in context.replace(" ", "").lower():
                                continue
                            else:
                                value_list[idx] += " " + continuation
                                skip += 1
                                break
                    else:
                        break
                state["slot_values"][slot] = value_list
                for value in value_list:
                    if value.replace(" ", "").lower() not in context.replace(" ", "").lower():
                        # Replace spaces to avoid issues with whitespace
                        if value.strip() not in SPECIAL_VALUES:
                            logger.warning(
                                f"Predicted value {value.strip()} for slot {pair[0].strip()} "
                                f"not in context in {dialogue_id}_{turn_index}."
                            )
        except KeyError:
            logger.warning(
                f"Could not extract slot {pair[0].strip()} in {predicted_str} in {dialogue_id}_{turn_index}.")


def populate_slots(
        predicted_data: dict,
        template_dialogue: dict,
        dialogue_id: str,
        schema: dict,
        model_name: str,
        preprocessed_references: dict,
        value_separator: Optional[str] = None,
        restore_categorical_case: bool = False
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
                raise ValueError

            # Extract string between <BOS> and <EOS>
            if 'gpt2' in model_name.lower():
                predicted_str = re.search(r"<BOS>(.*)<EOS>", predicted_str).group(1).strip()
            elif 't5' in model_name.lower():
                predicted_str = re.search(r"(.*)<EOS>", predicted_str).group(1).strip()
            else:
                raise ValueError("Unsupported model.")

            state = parse_predicted_string(dialogue_id, turn_index, service_name, predicted_str,
                                           ref_proc_turn["frames"][service_name]["slot_mapping"],
                                           ref_proc_turn["frames"][service_name]["cat_values_mapping"],
                                           ref_proc_turn["frames"][service_name]["intent_mapping"],
                                           context, value_separator=value_separator,
                                           restore_categorical_case=restore_categorical_case

                                           )
            # Update
            empty_ref_frame_state = empty_ref_frame["state"]
            empty_ref_frame_state.update(state)


def parse(
        schema: dict,
        predictions: dict,
        preprocessed_references: dict,
        output_dir: pathlib.Path,
        experiment_config: DictConfig
):
    model_name = experiment_config.decode.model_name_or_path
    try:
        data_processing_config = experiment_config.data.preprocessing
    except AttributeError:
        data_processing_config = OmegaConf.create()
    try:
        value_separator = data_processing_config.value_selection.value_separator
    except AttributeError:
        value_separator = None
    try:
        recase_categorical_values = data_processing_config.lowercase_model_targets
    except AttributeError:
        recase_categorical_values = False

    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    pattern = re.compile(r"dialogues_[0-9]+\.json")
    for file in output_dir.iterdir():
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
                    preprocessed_references[dialogue_id],
                    value_separator=value_separator,
                    restore_categorical_case=recase_categorical_values
                )
            with open(output_dir.joinpath(file.name), "w") as f:
                json.dump(dialogue_templates, f, indent=4)
