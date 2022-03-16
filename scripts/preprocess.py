import argparse
import json
import os
import random

import re
import string
from typing import Optional, List

DONTCARE = {
    "Banks_1": ["recipient_account_type"],
    "Banks_2": ["recipient_account_type"],
    "Buses_2": ["fare_type"],
    "Buses_3": ["category"],
    "Flights_1": ["airlines", "seating_class"],
    "Flights_2": ["airlines", "seating_class"],
    "Flights_3": ["airlines", "flight_class"],
    "Flights_4": ["airlines", "seating_class"],
    "Media_2": ["subtitle_language"],
    "Media_3": ["subtitle_language"],
    "Movies_1": ["show_type"],
    "Music_1": ["playback_device"],
    "Music_2": ["playback_device"],
    "Music_3": ["device"],
    "RentalCars_1": ["type"],
    "RentalCars_2": ["car_type"],
    "RentalCars_3": ["car_type"],
    "Restaurants_1": ["price_range"],
    "Restaurants_2": ["price_range"],
    "Trains_1": ["class"],
    "Travel_1": ["category"]
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
    return None


def process_frame(
        frame: dict,
        turn_info: dict,
        previous_slots: dict,
        system_utterance: str,
        user_utterance: str
):
    service = frame["service"]
    state = frame["state"]
    expected_output = "[states] "

    # Need to handle case when there are multiple possible values for the slot
    # We pick either the one that was previously in the state, or the one that
    # appears in the system/user utterance, or failing which, the first value
    # in the list
    current_slots = {}
    # TODO: THIS VERY LIKELY SELECTS VALUE at index 0
    for slot, values in state["slot_values"].items():
        assert isinstance(values, list)
        if service in previous_slots and slot in previous_slots[service]:
            assert not isinstance(previous_slots[service][slot], list)
        if service in previous_slots and slot in previous_slots[service] and \
                previous_slots[service][slot] in values:
            current_slots[slot] = previous_slots[service][slot]
        else:
            value = value_in_utterance(values, system_utterance, user_utterance)
            current_slots[slot] = value if value is not None else values[0]
    # Update
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
                expected_output += f"{i}:{cat_values_mapping[slot][current_slots[slot]]} "
            else:
                # Non-categorical
                expected_output += f"{i}:{current_slots[slot]} "

    # Active intent
    expected_output += "[intents] "
    if state["active_intent"] != "NONE":
        expected_output += turn_info[service]["intent_mapping"][state["active_intent"]] + " "

    # Requested slots
    expected_output += "[req_slots] "
    for i in range(len(slot_mapping) // 2):
        slot = slot_mapping[i]
        if slot in state["requested_slots"]:
            expected_output += f"{i} "

    # Update
    turn_info[service]["expected_output"] = expected_output.strip()


def generate_description(
        schema: List[dict],
        turn: dict
) -> dict:
    services = list(sorted([frame["service"] for frame in turn["frames"]]))
    ordered_services = [s["service_name"] for s in services]
    assert ordered_services == sorted(ordered_services)
    result = {}
    for service in schema:
        if service["service_name"] == services[0]:
            service_name = service["service_name"]
            description = ""
            slot_mapping = {}  # two-way dictionary between slot names and indices
            cat_values_mapping = {}  # slot to (value to index)
            intent_mapping = {}  # two-way dictionary between intent names and indices

            random.shuffle(service["slots"])
            for i, slot in enumerate(service["slots"]):
                slot_name = slot["name"]
                slot_description = slot["description"]
                # TODO: SEPARATOR IS = not :
                description += f"{i}:{slot_description} "
                slot_mapping[slot_name] = i
                slot_mapping[i] = slot_name

                if slot["is_categorical"]:
                    if service_name in DONTCARE and slot_name in DONTCARE[service_name] and \
                            "dontcare" not in slot["possible_values"]:
                        slot["possible_values"].append("dontcare")
                    assert len(slot["possible_values"]) == len(set(slot["possible_values"]))
                    random.shuffle(slot["possible_values"])
                    cat_values_mapping[slot_name] = {}  # value to index
                    for s, value in zip(list(string.ascii_lowercase), slot["possible_values"]):
                        description += f"{i}{s}) {value} "
                        cat_values_mapping[slot_name][value] = f"{i}{s}"

            random.shuffle(service["intents"])
            for i, intent in enumerate(service["intents"], 1):
                intent_name = intent["name"]
                intent_description = intent["description"]
                # TODO: SEPARATOR IS = NOT :
                description += f"i{i}:{intent_description} "
                intent_mapping[intent_name] = f"i{i}"
                intent_mapping[f"i{i}"] = intent_name

            result[service_name] = {
                "description": description.strip(),
                "slot_mapping": slot_mapping,
                "cat_values_mapping": cat_values_mapping,
                "intent_mapping": intent_mapping
            }
            services.pop(0)
            if not services:
                break
    return result


def process_file(
        schema: List[dict],
        data: list
) -> dict:
    result = {}
    for dialogue in data:
        dialogue_id = dialogue["dialogue_id"]
        result[dialogue_id] = []
        system_utterance = ""
        previous_slots = {}

        for turn in dialogue["turns"]:
            if turn["speaker"] == "SYSTEM":
                system_utterance = turn["utterance"]
                # We don't need to do anything else

            elif turn["speaker"] == "USER":
                turn_info = generate_description(schema, turn)
                user_utterance = turn["utterance"]
                for frame in turn["frames"]:
                    # Each frame represents one service
                    process_frame(frame, turn_info, previous_slots, system_utterance, user_utterance)

                result[dialogue_id].append({
                    "frames": turn_info,
                    "system_utterance": system_utterance,
                    "user_utterance": user_utterance
                })

            else:
                raise ValueError("Unknown speaker.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='Directory containing `dialogues_XXX.json` files', required=True)
    parser.add_argument('-o', '--out', help='Output file location and name', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.dir, "schema.json")) as f:
        schema = json.load(f)
    pattern = re.compile(r"dialogues_[0-9]+\.json")
    result = {}
    for file in os.listdir(args.dir):
        if pattern.match(file):
            with open(os.path.join(args.dir, file), "r") as f:
                data = json.load(f)
            result.update(process_file(schema, data))

    out = {
        "data": result
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=4)


if __name__ == '__main__':
    main()
