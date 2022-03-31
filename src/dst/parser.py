from __future__ import annotations

import functools
import json
import logging
import pathlib
import re
from collections import Counter
from typing import Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

SPECIAL_VALUES = {'dontcare'}
KNOWN_TARGET_INDEX_SEPARATORS = {"=>"}
"""Separators which do not result in sequences with multiple parses.
Used to call a parsing algorithm which does not use context to handle
ambiguous parses.
"""

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
    "regular": {
        "RideSharing_1": "Regular",
        "RideSharing_11": "Regular",
        "RideSharing_12": "Regular",
        "RideSharing_13": "Regular",
        "RideSharing_14": "Regular",
        "RideSharing_15": "Regular",
        "RideSharing_2": "Regular",
        "RideSharing_21": "Regular",
        "RideSharing_22": "Regular",
        "RideSharing_23": "Regular",
        "RideSharing_24": "Regular",
        "RideSharing_25": "Regular",
        "Movies_1": "regular",
        "Movies_11": "regular",
        "Movies_12": "regular",
        "Movies_13": "regular",
        "Movies_14": "regular",
        "Movies_15": "regular",

    },
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
time_slots = {'alert_time', 'depart_time_to_origin', 'time_flight_arrives_to_origin_airport', 'destination_flight_departure_time', 'start_time', 'local_destination_to_origin_arrival_time', 'from_origin_departure_time', 'time_of_alarm', 'arrival_time_at_destination', 'local_time_of_departure_from_origin', 'event_time', 'vehicle_retrieval_time', 'car_rental_pickup_time', 'outbound_arrival_time', 'event_start_time', 'free_time_end', 'pick_up_time_for_rental', 'dentist_appointment_time', 'time_of_therapy_visit', 'take_off_time_to_destination', 'arrival_time_return_flight', 'new_alarm_time', 'return_flight_arrival_time', 'appt_time', 'destination_departure_local_time', 'origin_flight_departure_time', 'consultation_time', 'arrival_time_outbound', 'departure_time', 'tentative_reservation_time', 'departure_hour', 'event_starts_at', 'outbound_local_destination_arrival_time', 'return_arrival_time', 'arrival_time_of_inbound_flight', 'clock_time_of_alarm', 'starting_time', 'arrival_time_return', 'alarm_time_to_set', 'returning_departure_time', 'movie_time', 'inbound_arrival_local_time', 'available_start_time', 'available_time_slot_start', 'time_of_outbound_departure', 'availability_end_time', 'leave_time', 'time_of_new_alarm', 'departing_time_from_origin', 'event_start', 'showtime', 'local_departing_time_to_destination', 'destination_to_origin_arrival_time', 'time_of_pickup', 'local_time_of_departure_from_destination', 'rental_pick_up_time', 'departing_time_of_return', 'arrival_time_of_outbound_flight', 'free_time_start', 'return_flight_departure_time', 'time_of_departure', 'time_of_show', 'origin_to_destination_departure_time', 'landing_time_at_destination', 'bus_leaving_time', 'rental_start_time', 'arrival_time', 'appointment_start_time', 'depart_time_out', 'time_of_return_arrival', 'restaurant_reservation_time', 'scheduled_time', 'additional_alarm_time', 'reservation_time', 'departing_time', 'dentist_visit_time', 'bus_departure_time', 'outbound_legflight_arrival_time', 'available_end_time', 'time_slot_start_time', 'time_of_visit', 'end_of_the_available_event', 'leaving_time', 'time_of_return_departure', 'time_to_reserve', 'time_flight_departs_from_origin', 'destination_to_origin_departure_time', 'to_destination_arrival_time', 'journey_start_time', 'departing_time_origin', 'alarm_time', 'new_set_time', 'return_legflight_arrival_time', 'requested_time_of_reservation', 'return_departure_time', 'rental_pickup_time', 'outbound_flight_departing_time', 'opening_time_of_the_event', 'outbound_departure_time', 'depart_time_return', 'inbound_arrival_time', 'time_to_see_stylist', 'starts_at', 'return_trip_flight_departure_time', 'car_retrieval_time', 'start_time_of_trip', 'destination_arrival_time', 'end_time', 'to_destination_departure_time', 'return_trip_flight_arrival_time', 'arrival_of_destination_to_origin_flight', 'appointment_time', 'when', 'doctor_appointment_time', 'start_time_of_journey', 'pick_up_time', 'origin_to_destination_arrival_time', 'new_alert_time', 'exact_time_of_appointment', 'time_of_departure_to_origin', 'inbound_departure_time', 'local_time_of_arrival_from_destination', 'pickup_time', 'time', 'departure_time_outbound_leg', 'return_legflight_departure_time', 'dining_time', 'time_of_outbound_arrival', 'current_alarm_time', 'time_flight_arrives_to_destination', 'outbound_flight_arrival_time', 'dental_appt_time', 'bus_time_departure', 'outbound_legflight_departure_time', 'begin_time', 'outbound_flight_departure_time', 'car_pickup_time', 'available_slot_start_time', 'availability_begin_time', 'time_of_appointment', 'scheduled_start_time', 'stylist_visit_time', 'returning_arrival_time', 'consult_time', 'event_begin_time', 'outbound_local_arrival_time', 'arrival_time_outbound_flight', 'time_of_day', 'show_time', 'movie_showing_time', 'availability_time_end', 'origin_flight_arrival_time', 'local_time_of_arrival_from_origin'}

date_pattern = re.compile(r"\d{1,2}(?:st|nd|rd|th)|(day|tomorrow)")

def is_date(s: str) -> bool:
    if re.search(date_pattern, s) is None:
        return False
    return True

def is_time_prefix(s: str) -> bool:
    time_prefixes = {'night', 'morning', 'evening', 'afternoon'}
    return any(s == cue for cue in time_prefixes)


def is_entity(s: str) -> bool:
    entity_cues = ['hotel', 'restaurant']
    return any(cue in s for cue in entity_cues)


def represents_time(s: str, value_separator: Optional[str] = None) -> bool:
    if value_separator is not None and value_separator in s:
        values = s.split(value_separator)
    else:
        values = [s]
    is_time = all(
        (
                # (re.match(r"([0-9]|10):([0-5])([0-9])", v) is not None) or
                (re.match(r"([0-9]|1[0-2]):([0-5])([0-9])$", v) is not None) or
                (re.match(r"([0-9]|1[0-2]):([0-5])([0-9])\s([ap]m|[AP]M)$", v) is not None)

        )
        for v in values
    )
    if is_time:
        return True
    else:
        return False

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
        target_slot_index_separator: str = ":"
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
        elif target_slot_index_separator in KNOWN_TARGET_INDEX_SEPARATORS:
            parse_without_context(
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
            raise ValueError(f"Unknown target slot index separator {target_slot_index_separator}")
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


def parse_categorical_slot(
        state: dict,
        dialogue_id: str,
        turn_index: str,
        service: str,
        slot: str,
        slot_value_pair: list[str, str],
        predicted_str: str,
        cat_values_mapping: dict,
        restore_categorical_case: bool = False
) -> bool:
    # Categorical
    recase = functools.partial(restore_case, restore_categorical_case=restore_categorical_case)
    # Invert the mapping to get the categorical value
    for categorical_value, categorical_value_idx in cat_values_mapping[slot].items():
        if categorical_value_idx == slot_value_pair[1].strip():
            recased_value = recase(value=categorical_value, service=service)
            assert isinstance(recased_value, str)
            state["slot_values"][slot] = [recased_value]
            return True
    else:
        logger.warning(
            f"Could not extract categorical value for slot {slot_value_pair[0].strip()} in "
            f"{predicted_str} in {dialogue_id}_{turn_index}. "
            f"Values defined for this slot were {cat_values_mapping[slot]}"
        )
        return False


def select_using_context(substrings: list[str], context: str, merge_index_candidates: list[int],
                         value_separator: Optional[str] = None, target_slot_index_separator: Optional[str] = ":"
                         ):
    partial_values = [substrings[idx-1].split(target_slot_index_separator)[1] for idx in merge_index_candidates]
    for i, partial_value in enumerate(partial_values):
        if is_time_prefix(partial_value):
            return merge_index_candidates[i]
    for merge_index in merge_index_candidates:
        if value_separator is not None and value_separator in substrings[merge_index - 1]:
            partial_value = substrings[merge_index - 1].split(value_separator)[-1]
        else:
            partial_value = substrings[merge_index - 1].split(target_slot_index_separator, 1)[1]
            if is_time_prefix(partial_value):
                return merge_index
        if value_separator is not None and value_separator in substrings[merge_index]:
            continuations = substrings[merge_index].split(value_separator)
        else:
            continuations = [substrings[merge_index]]
        if len(continuations) > 1 and continuations[0].count(target_slot_index_separator) > 1 and continuations[1].count(target_slot_index_separator) == 0:
            continue
        for continuation in continuations:
            if f"{partial_value}{continuation}".replace(" ", "").lower() in context.replace(" ", "").lower():
                return merge_index


def is_time_slot(slot_name: str) -> bool:
    return slot_name in time_slots


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

    def find_merge_index(target_slot_indices: list[int]) -> list[int]:
        if len(target_slot_indices) != len(set(target_slot_indices)):
            repeated_slot_index = Counter(target_slot_indices).most_common(1)[0][0]
            repeated_positions = [
                pos for pos in range(len(target_slot_indices)) if target_slot_indices[pos] == repeated_slot_index]
            if repeated_slot_index == target_slot_indices[0]:
                return [target_slot_indices.index(repeated_slot_index, 1)]
            return repeated_positions
        for i in range(1, len(target_slot_indices)):
            arr = [target_slot_indices[0]] + [
                target_slot_indices[idx]
                for idx in range(len(target_slot_indices))
                if idx != i and idx != 0
            ]
            if arr == sorted(arr):
                next_arr = [target_slot_indices[0]] + [target_slot_indices[idx] for idx in
                                                       range(1, len(target_slot_indices)) if idx != i + 1 and idx != 0]
                if next_arr == sorted(next_arr):
                    if len(arr) == len(set(arr)):
                        return [i, i + 1]
                    repeated_slot_index = Counter(arr).most_common(1)[0][0]
                    repeated_positions = [pos for pos in range(len(arr)) if arr[pos] == repeated_slot_index]
                    if repeated_slot_index == target_slot_indices[0]:
                        return [target_slot_indices.index(repeated_slot_index, 1)]
                    if i in repeated_positions:
                        return [i + 1]
                    else:
                        return [i]
                if len(next_arr) == len(target_slot_indices):
                    return [i, i - 1]
                else:
                    return [i]

        repeated_slot_index = Counter(target_slot_indices).most_common(1)[0][0]
        repeated_positions = [
            pos for pos in range(len(target_slot_indices))
            if target_slot_indices[pos] == repeated_slot_index
        ]
        if len(repeated_positions) == 1:
            for i in range(len(target_slot_indices)):
                if target_slot_indices[:i] != sorted(target_slot_indices[:i]):
                    # TODO: TRY I-1 IF THIS DOES NOT WORK
                    return [i-1, i]
        # if repeated_positions[0]== 0:
        #     return [1]
        return repeated_positions

    def find_categorical_slots(substrings: list[str]) -> list[bool]:
        # TODO: DONT HARDCODE :
        cat_pattern = r"([0-9]|1[0-6]):([0-9]|1[0-6])[a-l]$"
        is_cat = []
        for s in substrings:
            matches_pattern = re.match(cat_pattern, s) is not None
            is_cat.append(
                len(s) <= 6 and len(s) % 2 == 0 and matches_pattern
            )
        return is_cat

    def check_last_two_in_context(substrings: list[str], context: str, target_slot_index_separator: Optional[str] = ":",
                                  value_separator: Optional[str] = None) -> bool:
        if len(substrings) < 2:
            return False
        if any(find_categorical_slots(substrings[-2:])):
            return False
        # if len(substrings) == 2:
        #     # logger.warning(f"Only two last substrings {substrings}")
        #     return False

        penultimate_value = substrings[-2].split(target_slot_index_separator, 1)[1]
        if value_separator is not None and value_separator in penultimate_value:
            penultimate_value = penultimate_value.split(value_separator)[-1]
        trailing_values = [substrings[-1]]
        if value_separator is not None and value_separator in substrings[-1]:
            trailing_values = substrings[-1].split(value_separator)
            # while trailing_values and target_slot_index_separator not in trailing_values:
            #     trailing_values.pop()
            trailing_values = [v for v in trailing_values if target_slot_index_separator in v]
        for trailing_value in trailing_values:
            if trailing_value.count(target_slot_index_separator) > 1:
                return False
            if f"{penultimate_value}{trailing_value}".replace(" ", "").lower() in context.replace(" ", "").lower():
                return True
        return False


    def check_first_two_in_context(substrings: list[str], context: str, target_slot_index_separator: Optional[str] = ":",
                                  value_separator: Optional[str] = None) -> bool:
        if len(substrings) < 2:
            return False
        if any(find_categorical_slots([substrings[0], substrings[1]])):
            return False
        # if len(substrings) == 2:
        #     # logger.warning(f"Only two last substrings {substrings}")
        #     return False

        penultimate_value = substrings[0].split(target_slot_index_separator, 1)[1]
        if value_separator is not None and value_separator in penultimate_value:
            penultimate_value = penultimate_value.split(value_separator)[-1]
        trailing_values = [substrings[1]]
        if value_separator is not None and value_separator in substrings[-1]:
            trailing_values = substrings[1].split(value_separator)
            # while trailing_values and target_slot_index_separator not in trailing_values:
            #     trailing_values.pop()
            trailing_values = [v for v in trailing_values if target_slot_index_separator in v]
        for trailing_value in trailing_values:
            if trailing_value.count(target_slot_index_separator) > 1:
                return False
            if f"{penultimate_value}{trailing_value}".replace(" ", "").lower() in context.replace(" ", "").lower():
                return True
        return False



    def merge_substrings(
            substrings: list[str],
            context: str,
            slot_value_mapping: dict,
            target_slot_index_separator: Optional[str] = ":",
            value_separator: Optional[str] = None,
    ) -> list[str]:
        nonlocal dialogue_id
        merge_index_candidates = []
        target_slot_indices = [int(el.split(target_slot_index_separator)[0]) for el in substrings]
        # merge values for slot indices that are greater than the number of slots in the service
        invalid_slot_indices = [slot_idx for slot_idx in target_slot_indices if
                                slot_idx > (len(slot_value_mapping) // 2)]
        while invalid_slot_indices:
            this_invalid_index = invalid_slot_indices.pop()
            merge_index = target_slot_indices.index(this_invalid_index)
            substrings = substrings[:merge_index - 1] + \
                         [f"{substrings[merge_index - 1]} {substrings[merge_index]}"] + \
                         substrings[merge_index + 1:]
            target_slot_indices = [int(el.split(target_slot_index_separator)[0]) for el in substrings]
            invalid_slot_indices = [slot_idx for slot_idx in target_slot_indices if
                                    slot_idx > (len(slot_value_mapping) // 2)]
        if target_slot_indices[:-1] == sorted(target_slot_indices[:-1]) and target_slot_indices != sorted(
                target_slot_indices):
            if len(target_slot_indices[:-1]) == len(set(target_slot_indices[:-1])):
                if any(find_categorical_slots([substrings[-1]])):
                    substrings = substrings[:-3] + [f"{substrings[-3]} {substrings[-2]}"] + [substrings[-1]]
                    target_slot_indices = target_slot_indices[:-2] + [target_slot_indices[-1]]
                else:
                    merge_index_candidates = [len(substrings)-2 , len(substrings)-1]
            else:
                repeated_slot_index = Counter(target_slot_indices[:-1]).most_common(1)[0][0]
                merge_index_candidates = [pos for pos in range(len(target_slot_indices)) if
                                          target_slot_indices[pos] == repeated_slot_index]

        if check_last_two_in_context(substrings, context, target_slot_index_separator=target_slot_index_separator,
                                     value_separator=value_separator):
            penultimate_slot = slot_value_mapping[substrings[-2].split(target_slot_index_separator)[0].strip()]
            if is_time_slot(penultimate_slot):
                substrings = substrings[:-2] + [f"{substrings[-2]} {substrings[-1]}"]
                target_slot_indices = [int(el.split(target_slot_index_separator)[0]) for el in substrings]
        if check_first_two_in_context(substrings, context, target_slot_index_separator=target_slot_index_separator,
                                     value_separator=value_separator):
            substrings = [f"{substrings[0]} {substrings[1]}"] + substrings[2:]
            target_slot_indices = [int(el.split(target_slot_index_separator)[0]) for el in substrings]

        if target_slot_indices == sorted(target_slot_indices):
            if len(set(target_slot_indices)) == len(target_slot_indices):
                return substrings

        assert len(target_slot_indices) > 1
        while target_slot_indices != sorted(target_slot_indices) or len(target_slot_indices) != len(set(target_slot_indices)):
            if not merge_index_candidates:
                merge_index_candidates = find_merge_index(target_slot_indices)
            assert merge_index_candidates is not None
            if len(merge_index_candidates) == 1:
                merge_index = merge_index_candidates[0]
            else:
                try:
                    try:
                        cat_indicator = find_categorical_slots([substrings[idx] for idx in merge_index_candidates])
                    except IndexError:
                        merge_index_candidates = find_merge_index(target_slot_indices)
                        cat_indicator = find_categorical_slots([substrings[idx] for idx in merge_index_candidates])

                    merge_index_candidates = [
                        merge_index_candidates[i]
                        for i in range(len(cat_indicator))
                        if not cat_indicator[i]
                    ]
                    if len(merge_index_candidates) >= 2:
                        merge_index = select_using_context(
                            substrings,
                            context,
                            merge_index_candidates,
                            value_separator=value_separator,
                            target_slot_index_separator=target_slot_index_separator,
                        )
                        if merge_index is None:
                            merge_index_candidates = find_merge_index(target_slot_indices)
                            continue
                    else:
                        # all remaning indices are categorical
                        if not merge_index_candidates:
                            merge_index_candidates = find_merge_index(target_slot_indices)
                            continue
                        merge_index = merge_index_candidates[0]
                except AssertionError:
                    print(dialogue_id, turn_index)
                    raise AssertionError
            substrings = substrings[:merge_index - 1] + \
                         [f"{substrings[merge_index - 1]} {substrings[merge_index]}"] + \
                         substrings[merge_index + 1:]
            target_slot_indices = [int(el.split(target_slot_index_separator)[0]) for el in substrings]
            merge_index_candidates = find_merge_index(target_slot_indices)

        return substrings

    skip = 0
    trailing_categoricals = 0
    all_categoricals = False
    # TODO: PARSE CATEGORICALS FIRST AND SIMPLIFY MERGE_SUBSTRINGS
    for substring in reversed(substrings):
        if all(find_categorical_slots([substring])):
            trailing_categoricals -= 1
        else:
            break
    if trailing_categoricals < 0:
        extracted_categoricals = substrings[trailing_categoricals:]
        for pair in extracted_categoricals:
            slot_idx, slot_value = pair.strip().split(f"{target_slot_index_separator}", 1)
            slot_name = slot_mapping[slot_idx]
            parse_categorical_slot(state, dialogue_id, turn_index, service, slot_name, [slot_idx, slot_value], predicted_str, cat_values_mapping, restore_categorical_case=restore_categorical_case )
        substrings = substrings[:trailing_categoricals]
        if not substrings:
            all_categoricals = True
        # TODO: ADD THEM TO STATE HERE
    substrings = merge_substrings(substrings, context, slot_mapping,
                                  target_slot_index_separator=target_slot_index_separator,
                                  value_separator=value_separator)
    if not all_categoricals:
        assert substrings

    parsed_nocat_slots = []
    for i, pair in enumerate(substrings):
        if skip > 0:
            skip -= 1
            continue
        is_categorical = all(find_categorical_slots([pair]))
        slot_index_val_pair = pair.strip()
        pair = pair.strip().split(f"{target_slot_index_separator}", 1)  # slot value pair
        if len(pair) != 2:
            # String was not in expected format
            logger.warning(f"Could not extract slot values in {predicted_str} in {dialogue_id}_{turn_index}.")
            continue
        try:
            slot = slot_mapping[pair[0].strip()]
            if is_categorical:
                parse_categorical_slot(
                    state,
                    dialogue_id,
                    turn_index,
                    service,
                    slot,
                    pair,
                    predicted_str,
                    cat_values_mapping,
                    restore_categorical_case=restore_categorical_case
                )
                # parsed_slots.append(slot)
            else:
                if parsed_nocat_slots:
                    # TODO: CAREFULLY CHECK IF IT CAN BE MERGED INTO A PREVIOUS SLOT
                    last_parsed_value = state["slot_values"][parsed_nocat_slots[-1]][-1]
                    if is_time_slot(parsed_nocat_slots[-1]) and represents_time(slot_index_val_pair, value_separator=value_separator):
                            # skip as these are just consecutive time slots that should not be merged
                            if is_time_slot(slot_mapping[pair[0]]) and slot_mapping[pair[0]] not in parsed_nocat_slots and not is_time_prefix(last_parsed_value):
                                logger.warning(
                                    f"Skipping merge of slot {parsed_nocat_slots[-1]}, value {last_parsed_value} with "
                                    f"{slot_index_val_pair} since {pair[0]} is a different slot with the same semantics, "
                                    f"{slot_mapping[pair[0]]} "
                                )
                                assert slot_mapping[pair[0]] not in state["slot_values"]
                                if value_separator is not None and value_separator in pair[1]:
                                    value_list = value.split(value_separator)
                                    value_list = [v.strip() for v in value_list]
                                else:
                                    value_list = [pair[1].strip()]
                                state["slot_values"][slot_mapping[pair[0]]] = value_list
                                logger.warning(
                                    f"Added values {value_list} to slot {slot_mapping[pair[0]]} in the state instead"
                                )
                                continue
                            merged_value = f"{last_parsed_value} {substrings[i]}"
                            if value_separator is not None and value_separator in merged_value:
                                new_values = merged_value.split(value_separator)
                                if new_values[0].count(target_slot_index_separator) > 1:
                                    raise ArithmeticError
                                    continue
                                logger.warning(
                                    f"{dialogue_id}({turn_index}) "
                                    f"Merged multiple values {substrings[i]} with {last_parsed_value} for slot {parsed_nocat_slots[-1]} \n"
                                    f" Old values are {state['slot_values'][parsed_nocat_slots[-1]]}. \n New values are {new_values}."
                                )
                                state["slot_values"][parsed_nocat_slots[-1]] = new_values
                            else:
                                state["slot_values"][parsed_nocat_slots[-1]][-1] = merged_value
                                logger.warning(f"Merged {last_parsed_value} and {substrings[i]} for slot {parsed_nocat_slots[-1]} to obtain {merged_value} \n"
                                               f"State is {state['slot_values'][parsed_nocat_slots[-1]]} ")
                            # logger.warning(f"{dialogue_id}({turn_index}) Merged {last_parsed_value} and {substrings[i]} into a value")
                            continue
                # else:
                #     state["slot_values"][parsed_slots[-1]][-1] = f"{state['slot_values'][parsed_slots[-1]][-1]} {substrings[i].strip()}"
                parsed_nocat_slots.append(slot)
                # Non-categorical
                # Check if the next slot could potentially be part of the current slot
                value = pair[1]
                if value_separator is not None and value_separator in value:
                    value_list = value.split(value_separator)
                    value_list = [v.strip() for v in value_list]
                else:
                    value_list = [value.strip()]
                # j = i + 1
                # while j < len(substrings):
                #     # Check if the combined string exists in the context
                #     for idx, value in enumerate(value_list):
                #         if value_separator is None:
                #             possible_continuations = [substrings[j]]
                #         else:
                #             if value_separator in substrings[j]:
                #                 value_separator_idx = substrings[j].index(value_separator)
                #                 substrings[j] = substrings[j][:value_separator_idx]
                #             possible_continuations = [substrings[j]]
                # for continuation in possible_continuations:
                #     # Replace spaces to avoid issues with whitespace
                #     if (value + continuation).replace(" ", "").lower() not in context.replace(" ", "").lower():
                #         continue
                #     else:
                #         value_list[idx] += " " + continuation
                #         skip += 1
                #         break
                # else:
                #     break
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


def parse_without_context(
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
        target_slot_index_separator: str = ":"
):
    for i, pair in enumerate(substrings):
        pair = pair.strip().split(f"{target_slot_index_separator}", 1)  # slot value pair
        if len(pair) != 2:
            # String was not in expected format
            logger.warning(f"Could not extract slot values in {predicted_str} in {dialogue_id}_{turn_index}.")
            continue
        slot = slot_mapping[pair[0].strip()]
        if slot in cat_values_mapping:
            parse_categorical_slot(
                state,
                dialogue_id,
                turn_index,
                service,
                slot,
                pair,
                predicted_str,
                cat_values_mapping,
                restore_categorical_case=restore_categorical_case
            )
        else:
            value = pair[1]
            if value_separator is not None and value_separator in value:
                value_list = value.split(value_separator)
                value_list = [v.strip() for v in value_list]
            else:
                value_list = [value.strip()]
            state["slot_values"][slot] = value_list
            for value in value_list:
                if value.replace(" ", "").lower() not in context.replace(" ", "").lower():
                    # Replace spaces to avoid issues with whitespace
                    if value.strip() not in SPECIAL_VALUES:
                        logger.warning(
                            f"Predicted value {value.strip()} for slot {pair[0].strip()} "
                            f"not in context in {dialogue_id}_{turn_index}."
                        )


def populate_dialogue_state(
        predicted_data: dict,
        template_dialogue: dict,
        dialogue_id: str,
        schema: dict,
        model_name: str,
        preprocessed_references: dict,
        value_separator: Optional[str] = None,
        restore_categorical_case: bool = False,
        target_slot_index_separator: str = ":",
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
                                           restore_categorical_case=restore_categorical_case,
                                           target_slot_index_separator=target_slot_index_separator,
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
        logger.info("Could not find information about data processing config.")
    try:
        value_separator = data_processing_config.value_selection.value_separator
    except AttributeError:
        value_separator = None
        logger.info(
            "Could not find attributes 'value_selection.value_separator' in data processing config. "
            f"Defaulting to {value_separator}."
        )
    try:
        recase_categorical_values = data_processing_config.lowercase_model_targets
    except AttributeError:
        recase_categorical_values = False
        logger.info(
            "Could not find attribute lowercase_model_targets in data processing config. "
            f"Defaulting to {recase_categorical_values}"
        )
    try:
        target_slot_index_separator = data_processing_config.target_slot_index_separator
    except AttributeError:
        target_slot_index_separator = ":"
        logger.info(
            "Could not find attribute target_slot_index_separator in data processing config. "
            f"Defaulting to {target_slot_index_separator}"
        )

    if value_separator is not None and target_slot_index_separator == ':':
        logger.warning(
            "Parser algorithm is not correct when multiple values are in the target sequence."
            "Use at your own risk!"
        )

    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)
    only_files = ['dialogues_023.json']
    pattern = re.compile(r"dialogues_[0-9]+\.json")
    for file in output_dir.iterdir():
        if pattern.match(file.name):
            # if file.name not in only_files:
            #     continue
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
                populate_dialogue_state(
                    predicted_data,
                    blank_dialogue,
                    dialogue_id,
                    schema,
                    model_name,
                    preprocessed_references[dialogue_id],
                    value_separator=value_separator,
                    restore_categorical_case=recase_categorical_values,
                    target_slot_index_separator=target_slot_index_separator,
                )
            with open(output_dir.joinpath(file.name), "w") as f:
                json.dump(dialogue_templates, f, indent=4)
