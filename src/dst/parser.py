from __future__ import annotations

import functools
import json
import logging
import pathlib
import re
from collections import Counter
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from dst.parser_metadata import (
    CATEGORICAL_SPECIAL_VALUES,
    UNAMBIGUOUS_TARGET_SEPARATORS,
    lower_to_schema_case,
    time_slots,
)

logger = logging.getLogger(__name__)

date_pattern = re.compile(r"\d{1,2}(?:st|nd|rd|th)|(day|tomorrow)")
dialogue_id: str = ""
turn_index: str = ""
MAX_PARSED_SUBSTRINGS = 30
MAX_SUBSTRING_LEN = 150


def is_date(s: str) -> bool:
    if re.search(date_pattern, s) is None:
        return False
    return True


def is_time_prefix(s: str) -> bool:
    time_prefixes = {"night", "morning", "evening", "afternoon"}
    return any(s == cue for cue in time_prefixes)


def is_entity(s: str) -> bool:
    entity_cues = ["hotel", "restaurant"]
    return any(cue in s for cue in entity_cues)


def represents_time(s: str, value_separator: Optional[str] = None) -> bool:
    """Checks if `s` is likely a time value."""
    if value_separator is not None and value_separator in s:
        values = s.split(value_separator)
    else:
        values = [s]
    is_time = all(
        (
            (re.match(r"([0-9]|1[0-2]):([0-5])([0-9])$", v) is not None)
            or (
                re.match(r"([0-9]|1[0-2]):([0-5])([0-9])\s([ap]m|[AP]M)$", v)
                is not None
            )
        )
        for v in values
    )
    return is_time


def restore_case(
    value: str, service: str, restore_categorical_case: bool = True
) -> str:
    """Restore the case of a given categorical slot `value` to the original
    schema casing to ensure scoring is correct."""
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
    service: str,
    predicted_str: str,
    slot_mapping: dict,
    cat_values_mapping: dict,
    intent_mapping: dict,
    context: str,
    value_separator: Optional[str] = None,
    restore_categorical_case: bool = False,
    target_slot_index_separator: str = ":",
) -> dict:
    """ "Convert predicted string to a SGD state dictionary of the form::

    {
        'slot_values': dict[str, list[str]], mapping slot names to lists of values,
        'active_intent': str, the current turn active intent
        'requested_slots': list[str] of names of information requested by the user.
    }
    """
    state = {"slot_values": {}, "active_intent": "NONE", "requested_slots": []}
    # Expect [states] 0:value 1:1a ... [intents] i1 [req_slots] 2 ...
    match = re.search(r"\[states](.*)\[intents](.*)\[req_slots](.*)", predicted_str)
    if match is None:
        # String was not in expected format
        logger.warning(
            f"Could not parse predicted string {predicted_str} in {dialogue_id}_{turn_index}."
        )
        return state

    # Parse slot values
    if match.group(1).strip():
        pattern = rf"(?<!^)\s+(?=[0-9]+{target_slot_index_separator})"
        if value_separator == " || ":
            pattern = rf"(?<!^)\s+(?=[0-9]+{target_slot_index_separator})(?<!\|\| )"
        elif value_separator is not None:
            logger.error(
                f"State splitting pattern undefined for value separator {value_separator}"
            )
        substrings = re.compile(pattern).split(match.group(1).strip())
        if target_slot_index_separator == ":":
            # updates state  in-place
            parse_with_context(
                state,
                substrings,
                service,
                predicted_str,
                slot_mapping,
                cat_values_mapping,
                context,
                value_separator=value_separator,
                restore_categorical_case=restore_categorical_case,
                target_slot_index_separator=target_slot_index_separator,
            )
        elif target_slot_index_separator in UNAMBIGUOUS_TARGET_SEPARATORS:
            parse_without_context(
                state,
                substrings,
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
            raise ValueError(
                f"Unknown target slot index separator {target_slot_index_separator}"
            )
    # Parse intent
    intent = match.group(2).strip()
    if intent:
        try:
            state["active_intent"] = intent_mapping[intent]
        except KeyError:
            logger.warning(
                f"Could not extract intent in {predicted_str} in {dialogue_id}_{turn_index}."
            )

    # Parse requested slots
    requested = match.group(3).strip().split()
    for index in requested:
        try:
            state["requested_slots"].append(slot_mapping[index.strip()])
        except KeyError:
            logger.warning(
                f"Could not extract requested slot {index.strip()} in {predicted_str} in {dialogue_id}_{turn_index}."
            )
    return state


def parse_categorical_slot(
    state: dict,
    service: str,
    slot: str,
    slot_value_pair: list[str, str],
    predicted_str: str,
    cat_values_mapping: dict,
    restore_categorical_case: bool = False,
):
    # Categorical
    recase = functools.partial(
        restore_case, restore_categorical_case=restore_categorical_case
    )
    # Invert the mapping to get the categorical value
    if slot not in cat_values_mapping:
        logger.warning(
            f"{dialogue_id}({turn_index}): Could not find slot {slot} in categorical mapping. \n"
            f"Categorical slots for service {service} are {cat_values_mapping.keys()}."
        )
        return
    for categorical_value, categorical_value_idx in cat_values_mapping[slot].items():
        if categorical_value_idx == slot_value_pair[1].strip():
            recased_value = recase(value=categorical_value, service=service)
            assert isinstance(recased_value, str)
            state["slot_values"][slot] = [recased_value]
            break
    else:
        logger.warning(
            f"{dialogue_id}({turn_index}): "
            f"Could not lookup categorical value {slot_value_pair[1].strip()} "
            f"for slot {slot_value_pair[0].strip()} in {predicted_str}. \n"
            f"Values defined for this slot were {cat_values_mapping[slot]}"
        )


def select_using_context(
    substrings: list[str],
    context: str,
    merge_index_candidates: list[int],
    value_separator: Optional[str] = None,
    target_slot_index_separator: Optional[str] = ":",
):
    partial_values = [
        substrings[idx - 1].split(target_slot_index_separator)[1]
        for idx in merge_index_candidates
    ]
    for i, partial_value in enumerate(partial_values):
        if is_time_prefix(partial_value):
            return merge_index_candidates[i]
    for merge_index in merge_index_candidates:
        if (
            value_separator is not None
            and value_separator in substrings[merge_index - 1]
        ):
            partial_value = substrings[merge_index - 1].split(value_separator)[-1]
        else:
            partial_value = substrings[merge_index - 1].split(
                target_slot_index_separator, 1
            )[1]
            if is_time_prefix(partial_value):
                return merge_index
        if value_separator is not None and value_separator in substrings[merge_index]:
            continuations = substrings[merge_index].split(value_separator)
        else:
            continuations = [substrings[merge_index]]
        if (
            len(continuations) > 1
            and continuations[0].count(target_slot_index_separator) > 1
            and continuations[1].count(target_slot_index_separator) == 0
        ):
            continue
        for continuation in continuations:
            if (
                f"{partial_value}{continuation}".replace(" ", "").lower()
                in context.replace(" ", "").lower()
            ):
                return merge_index


def is_time_slot(slot_name: str) -> bool:
    return slot_name in time_slots


def preprocess_substrings(substrings: list[str]) -> list[str]:
    if len(substrings) > MAX_PARSED_SUBSTRINGS:
        logger.warning(
            f"{dialogue_id}({turn_index}) {len(substrings)} were parsed. Parsing only the first {MAX_PARSED_SUBSTRINGS}"
        )
        substrings = substrings[:MAX_PARSED_SUBSTRINGS]
    for i in range(len(substrings)):
        if len(substrings[i]) > MAX_SUBSTRING_LEN:
            logger.warning(
                f"{dialogue_id}({turn_index}) A substring len ({len(substrings[i])}) exceeded max value configured "
                f"({MAX_SUBSTRING_LEN}). Using only {MAX_SUBSTRING_LEN} to extract values."
            )
            substrings[i] = substrings[i][:MAX_SUBSTRING_LEN]
    return substrings


def find_categorical_slots(
    substrings: list[str], target_slot_index_separator: str = ":"
) -> list[bool]:
    """Determine whether each element of `substrings` is a categorical_slot_index - value pointer pair."""
    cat_pattern = rf"([0-9]|1[0-6]){target_slot_index_separator}([0-9]|1[0-6])[a-l]$"
    is_cat = []
    max_len = 5 + len(target_slot_index_separator)
    remainder = 0 if len(target_slot_index_separator) % 2 == 1 else 1
    for s in substrings:
        matches_pattern = re.match(cat_pattern, s) is not None
        is_cat.append(len(s) <= max_len and len(s) % 2 == remainder and matches_pattern)
    return is_cat


def parse_with_context(
    state: dict,
    substrings: list[str],
    service: str,
    predicted_str: str,
    slot_mapping: dict,
    cat_values_mapping: dict,
    context: str,
    value_separator: Optional[str] = None,
    restore_categorical_case: bool = False,
    target_slot_index_separator: Optional[str] = ":",
):
    """Using ":" to separate slot indices from their values in the target sequence makes parsing difficult because this
    is a symbol that appears in values. Therefore, 1:morning 11:15 is parsed as (1, morning), (11, 15) which is wrong.
    This algorithm assumes the following substrings could be part of the current slot values and uses the context to
    check whether this is the case. This algorithm is __NOT__ correct if the value separator is not `None`.
    """

    def get_merge_index_candidates(target_slot_indices: list[int]) -> list[int]:
        """By design, the target sequence slot indices are ordered. In most cases,
        when the splitting of the predicted string does not correctly work, the
        list is no longer sorted. This function select a list of potential merge
        candidates by checking the sorting properties of the target slot indices list.
        These will be used to merge back the substrings produced by the splitter output.
        """

        # substrings with repeated slot indices are candidates for merging
        if len(target_slot_indices) != len(set(target_slot_indices)):
            repeated_slot_index = Counter(target_slot_indices).most_common(1)[0][0]
            repeated_positions = [
                pos
                for pos in range(len(target_slot_indices))
                if target_slot_indices[pos] == repeated_slot_index
            ]
            # first slot index is always correct, so a slot with the same index
            # is a splitting mistake -> return the other index for merging
            if repeated_slot_index == target_slot_indices[0]:
                return [target_slot_indices.index(repeated_slot_index, 1)]
            return repeated_positions
        # determine if there are any slot indices that break target slot index array
        # sorting and return them
        for i in range(1, len(target_slot_indices)):
            arr = [target_slot_indices[0]] + [
                target_slot_indices[idx]
                for idx in range(len(target_slot_indices))
                if idx != i and idx != 0
            ]
            if arr == sorted(arr):
                next_arr = [target_slot_indices[0]] + [
                    target_slot_indices[idx]
                    for idx in range(1, len(target_slot_indices))
                    if idx != i + 1 and idx != 0
                ]
                if next_arr == sorted(next_arr):
                    if len(arr) == len(set(arr)):
                        return [i, i + 1]
                    # find duplicates as before and return indices of duplicated values
                    repeated_slot_index = Counter(arr).most_common(1)[0][0]
                    repeated_positions = [
                        pos
                        for pos in range(len(arr))
                        if arr[pos] == repeated_slot_index
                    ]
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
        # Fallback if omitting every index in turn does not yield a
        # sorted target sequence slot index  array
        repeated_slot_index = Counter(target_slot_indices).most_common(1)[0][0]
        repeated_positions = [
            pos
            for pos in range(len(target_slot_indices))
            if target_slot_indices[pos] == repeated_slot_index
        ]
        # see if there are duplicates otherwise return the index
        # that breaks sorting and the previous index
        if len(repeated_positions) == 1:
            for i in range(len(target_slot_indices)):
                if target_slot_indices[:i] != sorted(target_slot_indices[:i]):
                    return [i - 1, i]
        return repeated_positions

    def check_last_two_in_context(
        substrings: list[str],
        context: str,
        target_slot_index_separator: Optional[str] = ":",
        value_separator: Optional[str] = None,
    ) -> bool:
        """Heuristic check to see if the last two substrings appear together in the dialogue context."""
        if len(substrings) < 2:
            return False
        if any(
            find_categorical_slots(
                substrings[-2:], target_slot_index_separator=target_slot_index_separator
            )
        ):
            return False

        penultimate_value = substrings[-2].split(target_slot_index_separator, 1)[1]
        if value_separator is not None and value_separator in penultimate_value:
            penultimate_value = penultimate_value.split(value_separator)[-1]
        trailing_values = [substrings[-1]]
        if value_separator is not None and value_separator in substrings[-1]:
            trailing_values = substrings[-1].split(value_separator)
            trailing_values = [
                v for v in trailing_values if target_slot_index_separator in v
            ]
        for trailing_value in trailing_values:
            if trailing_value.count(target_slot_index_separator) > 1:
                return False
            if (
                f"{penultimate_value}{trailing_value}".replace(" ", "").lower()
                in context.replace(" ", "").lower()
            ):
                return True
        return False

    def check_first_two_in_context(
        substrings: list[str],
        context: str,
        target_slot_index_separator: Optional[str] = ":",
        value_separator: Optional[str] = None,
    ) -> bool:
        """ "Heuristic function to check whether the first two substrings appear together in context."""
        if len(substrings) < 2:
            return False
        if any(
            find_categorical_slots(
                [substrings[0], substrings[1]],
                target_slot_index_separator=target_slot_index_separator,
            )
        ):
            return False

        penultimate_value = substrings[0].split(target_slot_index_separator, 1)[1]
        if value_separator is not None and value_separator in penultimate_value:
            penultimate_value = penultimate_value.split(value_separator)[-1]
        trailing_values = [substrings[1]]
        if value_separator is not None and value_separator in substrings[-1]:
            trailing_values = substrings[1].split(value_separator)
            trailing_values = [
                v for v in trailing_values if target_slot_index_separator in v
            ]
        for trailing_value in trailing_values:
            if trailing_value.count(target_slot_index_separator) > 1:
                return False
            if (
                f"{penultimate_value}{trailing_value}".replace(" ", "").lower()
                in context.replace(" ", "").lower()
            ):
                return True
        return False

    def merge_substrings(
        substrings: list[str],
        context: str,
        slot_value_mapping: dict,
        target_slot_index_separator: Optional[str] = ":",
        value_separator: Optional[str] = None,
    ) -> list[str]:
        """Heuristic algorithm for merging substrings extracted from the model prediction."""
        if len(substrings) == 1:
            return substrings
        merge_index_candidates = []
        target_slot_indices = []
        exclude_indices = []
        for el_idx, el in enumerate(substrings):
            try:
                slot_index = int(el.split(target_slot_index_separator)[0])
                target_slot_indices.append(slot_index)
            except ValueError:
                logger.warning(
                    f"{dialogue_id}({turn_index}) Could not extract slot index for substring {el}"
                )
                exclude_indices.append(el_idx)
                continue
        substrings = [
            substrings[i] for i in range(len(substrings)) if i not in exclude_indices
        ]
        # predictions might not obey the indexing for the service
        this_service_max_slot_idx = len(slot_value_mapping) // 2 - 1
        invalid_slot_indices = [
            slot_idx
            for slot_idx in target_slot_indices
            if slot_idx > this_service_max_slot_idx
        ]
        # if all indices are wrong, we return the substrings
        if invalid_slot_indices == target_slot_indices:
            return substrings
        # if some are correct, we attempt to merge strings and return state
        while invalid_slot_indices:
            this_invalid_index = invalid_slot_indices.pop()
            merge_index = target_slot_indices.index(this_invalid_index)
            # unparsable sequence, probably missing index
            if merge_index == 0:
                break
            substrings = (
                substrings[: merge_index - 1]
                + [f"{substrings[merge_index - 1]} {substrings[merge_index]}"]
                + substrings[merge_index + 1 :]
            )
            target_slot_indices = [
                int(el.split(target_slot_index_separator)[0]) for el in substrings
            ]
            invalid_slot_indices = [
                slot_idx
                for slot_idx in target_slot_indices
                if slot_idx > this_service_max_slot_idx
            ]
            if len(substrings) == 1:
                break
        if target_slot_indices[:-1] == sorted(
            target_slot_indices[:-1]
        ) and target_slot_indices != sorted(target_slot_indices):
            if len(target_slot_indices[:-1]) == len(set(target_slot_indices[:-1])):
                if any(
                    find_categorical_slots(
                        [substrings[-1]],
                        target_slot_index_separator=target_slot_index_separator,
                    )
                ):
                    substrings = (
                        substrings[:-3]
                        + [f"{substrings[-3]} {substrings[-2]}"]
                        + [substrings[-1]]
                    )
                    target_slot_indices = target_slot_indices[:-2] + [
                        target_slot_indices[-1]
                    ]
                else:
                    merge_index_candidates = [len(substrings) - 2, len(substrings) - 1]
            else:
                repeated_slot_index = Counter(target_slot_indices[:-1]).most_common(1)[
                    0
                ][0]
                merge_index_candidates = [
                    pos
                    for pos in range(len(target_slot_indices))
                    if target_slot_indices[pos] == repeated_slot_index
                ]

        if check_last_two_in_context(
            substrings,
            context,
            target_slot_index_separator=target_slot_index_separator,
            value_separator=value_separator,
        ):
            penultimate_slot = slot_value_mapping[
                substrings[-2].split(target_slot_index_separator)[0].strip()
            ]
            if is_time_slot(penultimate_slot):
                substrings = substrings[:-2] + [f"{substrings[-2]} {substrings[-1]}"]
                target_slot_indices = [
                    int(el.split(target_slot_index_separator)[0]) for el in substrings
                ]
        if check_first_two_in_context(
            substrings,
            context,
            target_slot_index_separator=target_slot_index_separator,
            value_separator=value_separator,
        ):
            substrings = [f"{substrings[0]} {substrings[1]}"] + substrings[2:]
            target_slot_indices = [
                int(el.split(target_slot_index_separator)[0]) for el in substrings
            ]

        if target_slot_indices == sorted(target_slot_indices):
            if len(set(target_slot_indices)) == len(target_slot_indices):
                return substrings

        assert len(target_slot_indices) > 1
        max_tries = 10
        while target_slot_indices != sorted(target_slot_indices) or len(
            target_slot_indices
        ) != len(set(target_slot_indices)):
            if not merge_index_candidates:
                merge_index_candidates = get_merge_index_candidates(target_slot_indices)
            assert merge_index_candidates is not None
            if len(merge_index_candidates) == 1:
                merge_index = merge_index_candidates[0]
            else:
                try:
                    try:
                        cat_indicator = find_categorical_slots(
                            [substrings[idx] for idx in merge_index_candidates],
                            target_slot_index_separator=target_slot_index_separator,
                        )
                    except IndexError:
                        merge_index_candidates = get_merge_index_candidates(
                            target_slot_indices
                        )
                        cat_indicator = find_categorical_slots(
                            [substrings[idx] for idx in merge_index_candidates],
                            target_slot_index_separator=target_slot_index_separator,
                        )

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
                            merge_index_candidates = get_merge_index_candidates(
                                target_slot_indices
                            )
                            max_tries -= 1
                            if max_tries == 0:
                                break
                            continue
                    else:
                        # all remaning indices are categorical
                        if not merge_index_candidates:
                            merge_index_candidates = get_merge_index_candidates(
                                target_slot_indices
                            )
                            max_tries -= 1
                            if max_tries == 0:
                                break
                            continue
                        merge_index = merge_index_candidates[0]
                except AssertionError:
                    print(dialogue_id, turn_index)
                    raise AssertionError
            substrings = (
                substrings[: merge_index - 1]
                + [f"{substrings[merge_index - 1]} {substrings[merge_index]}"]
                + substrings[merge_index + 1 :]
            )
            target_slot_indices = [
                int(el.split(target_slot_index_separator)[0]) for el in substrings
            ]
            merge_index_candidates = get_merge_index_candidates(target_slot_indices)
            max_tries -= 1
            if max_tries == 0:
                break
        if max_tries == 0:
            logger.warning(
                f"{dialogue_id}({turn_index}) Substrings merging failed after 10 tries. \n"
                f"Substrings are {substrings}."
            )
        return substrings

    trailing_categoricals = 0
    all_categoricals = False
    substrings = preprocess_substrings(substrings)
    # TODO: PARSE CATEGORICALS FIRST AND SIMPLIFY MERGE_SUBSTRINGS
    for substring in reversed(substrings):
        if all(
            find_categorical_slots(
                [substring], target_slot_index_separator=target_slot_index_separator
            )
        ):
            trailing_categoricals -= 1
        else:
            break
    if trailing_categoricals < 0:
        extracted_categoricals = substrings[trailing_categoricals:]
        for pair in extracted_categoricals:
            slot_idx, slot_value = pair.strip().split(
                f"{target_slot_index_separator}", 1
            )
            try:
                slot_name = slot_mapping[slot_idx]
            except KeyError:
                logger.warning(
                    f"Invalid slot index {slot_idx} in service {service} where only {len(slot_mapping.items())//2} "
                    f"indices are defined."
                )
                continue
            parse_categorical_slot(
                state,
                service,
                slot_name,
                [slot_idx, slot_value],
                predicted_str,
                cat_values_mapping,
                restore_categorical_case=restore_categorical_case,
            )
        substrings = substrings[:trailing_categoricals]
        if not substrings:
            all_categoricals = True

    substrings = merge_substrings(
        substrings,
        context,
        slot_mapping,
        target_slot_index_separator=target_slot_index_separator,
        value_separator=value_separator,
    )
    if not all_categoricals:
        assert substrings

    parsed_nocat_slots = []
    for i, pair in enumerate(substrings):
        is_categorical = all(
            find_categorical_slots(
                [pair], target_slot_index_separator=target_slot_index_separator
            )
        )
        slot_index_val_pair = pair.strip()
        pair = pair.strip().split(
            f"{target_slot_index_separator}", 1
        )  # slot value pair
        if len(pair) != 2:
            # String was not in expected format
            logger.warning(
                f"Could not extract slot values in {predicted_str} in {dialogue_id}_{turn_index}."
            )
            continue
        try:
            slot = slot_mapping[pair[0].strip()]
            if is_categorical:
                parse_categorical_slot(
                    state,
                    service,
                    slot,
                    pair,
                    predicted_str,
                    cat_values_mapping,
                    restore_categorical_case=restore_categorical_case,
                )
                # parsed_slots.append(slot)
            else:
                if parsed_nocat_slots:
                    # TODO: CAREFULLY CHECK IF IT CAN BE MERGED INTO A PREVIOUS SLOT
                    last_parsed_value = state["slot_values"][parsed_nocat_slots[-1]][-1]
                    if is_time_slot(parsed_nocat_slots[-1]) and represents_time(
                        slot_index_val_pair, value_separator=value_separator
                    ):
                        # skip as these are just consecutive time slots that should not be merged
                        if (
                            is_time_slot(slot_mapping[pair[0]])
                            and slot_mapping[pair[0]] not in parsed_nocat_slots
                            and not is_time_prefix(last_parsed_value)
                        ):
                            logger.warning(
                                f"{dialogue_id}({turn_index}) Skipping merge of slot {parsed_nocat_slots[-1]}, "
                                f"value {last_parsed_value} with "
                                f"{slot_index_val_pair} since slot {pair[0]} is a different slot with the same "
                                f"semantics, {slot_mapping[pair[0]]} "
                            )
                            if (
                                value_separator is not None
                                and value_separator in pair[1]
                            ):
                                value_list = pair[1].split(value_separator)
                                value_list = [v.strip() for v in value_list]
                            else:
                                value_list = [pair[1].strip()]
                            if slot_mapping[pair[0]] not in state["slot_values"]:
                                state["slot_values"][slot_mapping[pair[0]]] = value_list
                            else:
                                logger.warning(
                                    f"{dialogue_id}({turn_index}) Found a second value for slot "
                                    f"{slot_mapping[pair[0]]} with index {pair[0]} in the same prediction. "
                                    f"Appending to value list."
                                )
                                state["slot_values"][slot_mapping[pair[0]]].extend(
                                    value_list
                                )
                            logger.warning(
                                f"Added values {value_list} to slot {slot_mapping[pair[0]]} in the state instead"
                            )
                            continue
                        merged_value = f"{last_parsed_value} {substrings[i]}"
                        if (
                            value_separator is not None
                            and value_separator in merged_value
                        ):
                            new_values = merged_value.split(value_separator)
                            if new_values[0].count(target_slot_index_separator) > 1:
                                logger.warning(
                                    f"First new value {new_values[0]} contained multiple target separators..."
                                )
                            logger.warning(
                                f"{dialogue_id}({turn_index}) "
                                f"Merged multiple values {substrings[i]} with {last_parsed_value} for slot "
                                f"{parsed_nocat_slots[-1]} \n"
                                f"Old values are {state['slot_values'][parsed_nocat_slots[-1]]}. \n"
                                f"New values are {new_values}."
                            )
                            state["slot_values"][parsed_nocat_slots[-1]] = new_values
                        else:
                            state["slot_values"][parsed_nocat_slots[-1]][
                                -1
                            ] = merged_value
                            logger.warning(
                                f"Merged {last_parsed_value} and {substrings[i]} for slot {parsed_nocat_slots[-1]} "
                                f"to obtain {merged_value} \n"
                                f"State is {state['slot_values'][parsed_nocat_slots[-1]]}."
                            )
                        continue
                parsed_nocat_slots.append(slot)
                value = pair[1]
                if value_separator is not None and value_separator in value:
                    value_list = value.split(value_separator)
                    value_list = [v.strip() for v in value_list]
                else:
                    value_list = [value.strip()]
                state["slot_values"][slot] = value_list
                for value in value_list:
                    if (
                        value.replace(" ", "").lower()
                        not in context.replace(" ", "").lower()
                    ):
                        # Replace spaces to avoid issues with whitespace
                        if value.strip() not in CATEGORICAL_SPECIAL_VALUES:
                            logger.warning(
                                f"Predicted value {value.strip()} for slot {pair[0].strip()} "
                                f"not in context in {dialogue_id}_{turn_index}."
                            )
        except KeyError:
            logger.warning(
                f"Could not extract slot {pair[0].strip()} in {predicted_str} in {dialogue_id}_{turn_index}."
            )


def parse_without_context(
    state: dict,
    substrings: list[str],
    service: str,
    predicted_str: str,
    slot_mapping: dict,
    cat_values_mapping: dict,
    context: str,
    value_separator: Optional[str] = None,
    restore_categorical_case: bool = False,
    target_slot_index_separator: str = ":",
):
    for i, pair in enumerate(substrings):
        pair = pair.strip().split(
            f"{target_slot_index_separator}", 1
        )  # slot value pair
        if len(pair) != 2:
            # String was not in expected format
            logger.warning(
                f"Could not extract slot values in {predicted_str} in {dialogue_id}_{turn_index}."
            )
            continue
        try:
            slot = slot_mapping[pair[0].strip()]
            if slot in cat_values_mapping:
                parse_categorical_slot(
                    state,
                    service,
                    slot,
                    pair,
                    predicted_str,
                    cat_values_mapping,
                    restore_categorical_case=restore_categorical_case,
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
                    if (
                        value.replace(" ", "").lower()
                        not in context.replace(" ", "").lower()
                    ):
                        # Replace spaces to avoid issues with whitespace
                        if value.strip() not in CATEGORICAL_SPECIAL_VALUES:
                            logger.warning(
                                f"Predicted value {value.strip()} for slot {pair[0].strip()} "
                                f"not in context in {dialogue_id}_{turn_index}."
                            )
        except KeyError:
            logger.warning(
                f"Could not extract slot {pair[0].strip()} in {predicted_str} in {dialogue_id}_{turn_index}."
            )


def populate_dialogue_state(
    predicted_data: dict,
    template_dialogue: dict,
    schema: dict,
    model_name: str,
    preprocessed_references: dict,
    value_separator: Optional[str] = None,
    restore_categorical_case: bool = False,
    target_slot_index_separator: str = ":",
):
    context = ""
    global turn_index
    for turn_index in predicted_data:
        # Loop over turns
        if int(turn_index) > 0:
            # Concatenate system utterance
            context += (
                template_dialogue["turns"][int(turn_index) * 2 - 1]["utterance"] + " "
            )
        template_turn = template_dialogue["turns"][
            int(turn_index) * 2
        ]  # skip system turns
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
            if "gpt2" in model_name.lower():
                try:
                    # Should contain the dialogue history
                    # We call replace() to avoid issues with extra whitespace
                    assert template_turn["utterance"].replace(
                        " ", ""
                    ) in predicted_str.replace(" ", "")
                except AssertionError:
                    logger.warning(
                        f"{predicted_str} in {dialogue_id}_{turn_index} does not match user utterance. Skipping."
                    )
                    raise AssertionError
            if "<EOS>" not in predicted_str:
                logger.warning(
                    f"No <EOS> token in {dialogue_id}_{turn_index}. Skipping."
                )
                continue

            # Extract string between <BOS> and <EOS>
            if "gpt2" in model_name.lower():
                predicted_str = (
                    re.search(r"<BOS>(.*)<EOS>", predicted_str).group(1).strip()
                )
            elif "t5" in model_name.lower():
                predicted_str = re.search(r"(.*)<EOS>", predicted_str).group(1).strip()
            else:
                raise ValueError("Unsupported model.")

            state = parse_predicted_string(
                service_name,
                predicted_str,
                ref_proc_turn["frames"][service_name]["slot_mapping"],
                ref_proc_turn["frames"][service_name]["cat_values_mapping"],
                ref_proc_turn["frames"][service_name]["intent_mapping"],
                context,
                value_separator=value_separator,
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
    experiment_config: DictConfig,
    value_separator: Optional[str] = None,
    recase_categorical_values: Optional[bool] = None,
    target_slot_index_separator: Optional[str] = None,
):
    model_name = experiment_config.decode.model_name_or_path
    try:
        data_processing_config = experiment_config.data.preprocessing
        train_data_paths = list(data_processing_config.keys())
    except AttributeError:
        data_processing_config = OmegaConf.create()
        train_data_paths = []
        logger.info("Could not find information about data processing config.")

    if value_separator is None:
        try:
            value_separators = []
            for shard in train_data_paths:
                value_separator = data_processing_config[
                    shard
                ].value_selection.value_separator
                value_separators.append(value_separator)
            assert len(set(value_separators)) == 1
            value_separator = value_separators[0]
        except AttributeError:
            value_separator = None
            logger.info(
                "Could not find attributes 'value_selection.value_separator' in data processing config "
                "Defaulting to None."
            )
        except AssertionError:
            value_separator = None
            logger.info(
                "Either attribute value_selection.value_separator was not set or different"
                "data shards had different settings. Defaulting to None."
            )
    if recase_categorical_values is None:
        try:
            recase_settings = []
            for shard in train_data_paths:
                recase_categorical_values = data_processing_config[
                    shard
                ].lowercase_model_targets
                recase_settings.append(recase_categorical_values)
            assert len(set(recase_settings)) == 1
            recase_categorical_values = recase_settings[0]
        except AttributeError:
            recase_categorical_values = False
            logger.info(
                "Could not find attribute lowercase_model_targets in data processing config. "
                f"Defaulting to {recase_categorical_values}"
            )
        except AssertionError:
            recase_categorical_values = False
            logger.info(
                "Either attribute lowercase_model_targets was not set or different"
                "data shards had different settings. Casing will not be altered."
            )
    if target_slot_index_separator is None:
        try:
            slot_index_separator_settings = []
            for shard in train_data_paths:
                target_slot_index_separator = data_processing_config[
                    shard
                ].target_slot_index_separator
                slot_index_separator_settings.append(target_slot_index_separator)
            target_slot_index_separator = slot_index_separator_settings[0]
            assert len(set(slot_index_separator_settings)) == 1
        except (IndexError, AttributeError):
            target_slot_index_separator = ":"
            logger.info(
                "Could not find attribute target_slot_index_separator in data processing config. "
                f"Defaulting to {target_slot_index_separator}"
            )
        except AssertionError:
            target_slot_index_separator = ":"
            logger.info(
                "Either attribute target_slot_index_separator was not set or different"
                "data shards had different settings. Defaulting to `:`."
            )

    if value_separator is not None and target_slot_index_separator == ":":
        logger.warning(
            "Parser algorithm is not correct when multiple values are in the target sequence."
            "Use at your own risk!"
        )

    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)
    only_files = ["dialogues_025.json"]  # noqa
    pattern = re.compile(r"dialogues_[0-9]+\.json")
    for file in output_dir.iterdir():
        if pattern.match(file.name):
            # if file.name not in only_files:
            #     continue
            logger.info(f"Parsing file {file}.")
            with open(file, "r") as f:
                dialogue_templates = json.load(f)
            for blank_dialogue in dialogue_templates:
                global dialogue_id
                dialogue_id = blank_dialogue["dialogue_id"]
                logger.info(f"Parsing dialogue {dialogue_id}")
                try:
                    predicted_data = predictions[dialogue_id]
                except KeyError:
                    logging.warning(
                        f"Could not find dialogue {dialogue_id} in predicted states."
                    )
                    raise KeyError
                populate_dialogue_state(
                    predicted_data,
                    blank_dialogue,
                    schema,
                    model_name,
                    preprocessed_references[dialogue_id],
                    value_separator=value_separator,
                    restore_categorical_case=recase_categorical_values,
                    target_slot_index_separator=target_slot_index_separator,
                )
            with open(output_dir.joinpath(file.name), "w") as f:
                json.dump(dialogue_templates, f, indent=4)
