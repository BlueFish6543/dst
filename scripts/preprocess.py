import argparse
import json
import os

import re
from typing import Optional, List

SEPARATORS = {
    "service": " <SVC> ",
    "default": " <SEP> ",
    "slot-value": " = "
}


def humanise(
        name: str,
        remove_trailing_numbers: bool = False
) -> str:
    # Convert a potentially camel or snake case string to a lower case string delimited by spaces
    # Adapted from https://stackoverflow.com/a/1176023
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    name = name.lower().replace("_", " ")
    if remove_trailing_numbers:
        # Remove trailing numbers
        name = re.sub('[0-9]+$', '', name)
    return re.sub(' +', ' ', name).strip()


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
        system_utterance: str,
        user_utterance: str
):
    service = humanise(frame["service"], remove_trailing_numbers=True)
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

    # <SVC> service <SEP> slot1 = value1 <SEP> slot2 = value2
    current_slots = map(
        lambda item: item[0] + SEPARATORS["slot-value"] + item[1],
        current_slots.items()
    )
    current_slots = SEPARATORS["default"].join(sorted(current_slots))
    slot_values = SEPARATORS["service"] + service + SEPARATORS["default"] + current_slots
    slot_values_list.append(slot_values.strip())


def process_file(data: list) -> dict:
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
                user_utterance = turn["utterance"]
                # These lists accumulate across frames
                active_intent_list = []
                requested_slots_list = []
                slot_values_list = []
                for frame in turn["frames"]:
                    # Each frame represents one service (?)
                    process_frame(frame,
                                  active_intent_list, requested_slots_list, slot_values_list,
                                  previous_slots, system_utterance, user_utterance)

                res = {
                    "system_utterance": system_utterance,
                    "user_utterance": user_utterance,
                    "active_intent": " ".join(sorted(active_intent_list)),
                    "requested_slots": " ".join(sorted(requested_slots_list)),
                    "slot_values": " ".join(sorted(slot_values_list))
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

    pattern = re.compile(r"dialogues_[0-9]+\.json")
    result = {}
    for file in os.listdir(args.dir):
        if pattern.match(file):
            with open(os.path.join(args.dir, file), "r") as f:
                data = json.load(f)
            result.update(process_file(data))

    r = re.compile(r" <.+> ")
    out = {
        "data": result,
        "special_tokens": list(map(str.strip, filter(r.match, SEPARATORS.values())))
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
