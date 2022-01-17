import argparse
import json
import os

import re
from typing import Optional, List

from src.dst.utils import humanise

SEPARATORS = {
    "service": " <SVC> ",
    "description": " : ",
    "default": " <SEP> ",
    "slot-value": " = "
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
        active_intent_list: list,
        requested_slots_list: list,
        slot_values_list: list,
        previous_slots: dict,
        slot_dict: dict,
        system_utterance: str,
        user_utterance: str
):
    service = humanise(frame["service"])
    state = frame["state"]

    # <SVC> service <SEP> intent
    active_intent = SEPARATORS["service"] + service + SEPARATORS["default"] + humanise(state["active_intent"])
    active_intent_list.append(active_intent.strip())

    # <SVC> service <SEP> slot1 <SEP> slot2
    if state["requested_slots"]:
        requested_slots = SEPARATORS["default"].join(sorted(map(humanise, state["requested_slots"])))
        requested_slots = SEPARATORS["service"] + service + SEPARATORS["default"] + requested_slots
        requested_slots_list.append(requested_slots.strip())

    if not state["slot_values"]:
        # We are done
        return

    # Need to handle case when there are multiple possible values for the slot
    # We pick either the one that was previously in the state, or the one that
    # appears in the system/user utterance, or failing which, the first value
    # in the list
    current_slots = {}
    for slot, values in state["slot_values"].items():
        slot = humanise(slot)
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

    # <SVC> service <SEP> slot1 = value1 <SEP> slot2 = value2
    current_slots = map(
        lambda item: item[0] + SEPARATORS["slot-value"] + item[1],
        current_slots.items()
    )
    current_slots = SEPARATORS["default"].join(sorted(current_slots))
    slot_values = SEPARATORS["service"] + service + SEPARATORS["default"] + current_slots
    slot_values_list.append(slot_values.strip())


def get_slot_names(
        schema: List[dict],
        turn: dict
) -> str:
    services = [frame["service"] for frame in turn["frames"]]
    result = ""
    for service in schema:
        if service["service_name"] == services[0]:
            slot_names = list(sorted([humanise(slot["name"]) for slot in service["slots"]]))
            result += SEPARATORS["service"] + humanise(service["service_name"]) + \
                SEPARATORS["default"] + SEPARATORS["default"].join(slot_names)
            services.pop(0)
            if not services:
                break
    return result.strip()


def get_slots(
        schema: List[dict],
        turn: dict
) -> dict:
    services = list(sorted([frame["service"] for frame in turn["frames"]]))
    result = {}
    for service in schema:
        if service["service_name"] == services[0]:
            service_name = humanise(service["service_name"])
            result[service_name] = {}
            for slot in service["slots"]:
                # <SVC> service <SEP> slot : description
                slot_name = humanise(slot["name"])
                description = SEPARATORS["service"] + service_name + \
                    SEPARATORS["default"] + slot_name + SEPARATORS["description"] + \
                    slot["description"]
                result[service_name][slot_name] = {
                    "description": description.strip(),
                    "value": ""
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
                slot_names = get_slot_names(schema, turn)
                slot_dict = get_slots(schema, turn)
                user_utterance = turn["utterance"]
                # These lists accumulate across frames
                active_intent_list = []
                requested_slots_list = []
                slot_values_list = []
                for frame in turn["frames"]:
                    # Each frame represents one service (?)
                    process_frame(frame,
                                  active_intent_list, requested_slots_list, slot_values_list,
                                  previous_slots, slot_dict, system_utterance, user_utterance)

                res = {
                    "system_utterance": system_utterance,
                    "user_utterance": user_utterance,
                    "active_intent": " ".join(sorted(active_intent_list)),
                    "requested_slots": " ".join(sorted(requested_slots_list)),
                    "slot_values": " ".join(sorted(slot_values_list)),
                    "slot_names": slot_names,
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
