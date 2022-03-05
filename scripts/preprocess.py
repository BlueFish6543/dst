import argparse
import json
import os
import random

import re
from typing import Optional, List

SEPARATORS = {
    # "service": " <SVC> ",
    # "description": " : ",
    "default": " <SEP> ",
    "pair": " = ",
    # "intent": " <INT> ",
    # "slot": " <SLT> ",
    # "values": " <VAL> "
}


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
        intent_dict: dict,
        previous_slots: dict,
        slot_dict: dict,
        system_utterance: str,
        user_utterance: str
):
    service = frame["service"]
    state = frame["state"]

    # Update active intent
    if state["active_intent"] != "NONE":
        intent_dict[service]["active"] = state["active_intent"]

    # Update requested slots
    for slot in state["requested_slots"]:
        slot_dict[service][slot]["requested"] = True

    if not state["slot_values"]:
        # We are done
        return

    # Need to handle case when there are multiple possible values for the slot
    # We pick either the one that was previously in the state, or the one that
    # appears in the system/user utterance, or failing which, the first value
    # in the list
    current_slots = {}
    for slot, values in state["slot_values"].items():
        if service in previous_slots and slot in previous_slots[service] and \
                previous_slots[service][slot] in values:
            current_slots[slot] = previous_slots[service][slot]
        else:
            value = value_in_utterance(values, system_utterance, user_utterance)
            current_slots[slot] = value if value is not None else values[0]
    # Update
    previous_slots[service] = current_slots

    # Update slot_dict
    for slot, value in current_slots.items():
        slot_dict[service][slot]["value"] = value


def get_intents(
        schema: List[dict],
        turn: dict
) -> dict:
    services = list(sorted([frame["service"] for frame in turn["frames"]]))
    result = {}
    for service in schema:
        if service["service_name"] == services[0]:
            service_name = service["service_name"]
            service_description = service["description"]

            description = "Intent: Service: " + service_description.strip()
            mapping = {}
            for index, intent in enumerate(service["intents"], 1):
                # Intent: Service: description 1: description 2: description ...
                description += " {}: ".format(index) + intent["description"].strip()
                mapping[intent["name"]] = index

            result[service_name] = {
                "description": description.strip(),
                "active": "",
                "mapping": mapping
            }
            services.pop(0)
            if not services:
                break
    return result


def get_slots(
        schema: List[dict],
        turn: dict
) -> dict:
    services = list(sorted([frame["service"] for frame in turn["frames"]]))
    result = {}
    for service in schema:
        if service["service_name"] == services[0]:
            service_name = service["service_name"]
            service_description = service["description"]
            result[service_name] = {}

            for slot in service["slots"]:
                # Categorical/Non-categorical: Service: description Slot: description [1: value 2: value ...]
                description = "Service: " + service_description.strip() + " Slot: " + slot["description"].strip()
                mapping = {}
                if slot["is_categorical"]:
                    for index, value in enumerate(slot["possible_values"], 1):
                        description += " {}: ".format(index) + value.strip()
                        mapping[value] = index
                    description = "Categorical: " + description
                else:
                    description = "Non-categorical: " + description

                result[service_name][slot["name"]] = {
                    "description": description.strip(),
                    "requested": False,
                    "value": "",
                    "mapping": mapping
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
                intent_dict = get_intents(schema, turn)
                slot_dict = get_slots(schema, turn)
                user_utterance = turn["utterance"]
                for frame in turn["frames"]:
                    # Each frame represents one service (?)
                    process_frame(frame, intent_dict, previous_slots,
                                  slot_dict, system_utterance, user_utterance)

                res = {
                    "system_utterance": system_utterance,
                    "user_utterance": user_utterance,
                    "intent_dict": intent_dict,
                    "slot_dict": slot_dict
                }
                result[dialogue_id].append(res)

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
        "data": result,
        "separators": SEPARATORS
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
