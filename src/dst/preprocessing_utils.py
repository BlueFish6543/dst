from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Generator, Optional, Union

from omegaconf import DictConfig

from dst.sgd_utils import infer_schema_variant_from_path
from dst.utils import load_json, to_json

logger = logging.getLogger(__name__)


def _paraphrased_description_iterator(
    ref_schemas: list[dict], variant_or_synthetic_schemas: dict[str, list[dict]]
) -> Generator[dict]:
    """A generator that returns dictionary of the form::

            {
                'type': Literal['service', 'slots', 'intents']
                'reference_description': str, SGD description
                "paraphrases": dict[str, str] mapping the variant IDs (keys of `variant_or_synthetic_schemas`) to the
                    description paraphrase for that variant
            }

    Parameters
    ----------
    ref_schemas
        List of service schemas.
    variant_or_synthetic_schemas
        Mapping from variant identifier to list of service schemas for that variant

    """
    for service_index in range(len(ref_schemas)):
        ref_service_schema = ref_schemas[service_index]
        var_service_schemas = {
            variant: variant_or_synthetic_schemas[variant][service_index]
            for variant in variant_or_synthetic_schemas
        }
        description_types = ["slots", "intents"]
        try:
            assert all(
                (
                    ref_service_schema["service_name"]
                    == var_service_schemas[variant]["service_name"][:-1]
                    for variant in variant_or_synthetic_schemas
                )
            )
        except AssertionError:
            assert all(
                (
                    ref_service_schema["service_name"]
                    == var_service_schemas[variant]["service_name"]
                    for variant in variant_or_synthetic_schemas
                )
            ) or all(
                (
                    ref_service_schema["service_name"][:-1]
                    == var_service_schemas[variant]["service_name"][:-1]
                    for variant in variant_or_synthetic_schemas
                )
            )

        yield {
            "type": "service",
            "reference_description": ref_service_schema["description"],
            "paraphrases": {
                var_id: var_service_schemas[var_id]["description"]
                for var_id in var_service_schemas
            },
            "service_name": ref_service_schema["service_name"],
        }
        for desc_type in description_types:
            assert all(
                (
                    len(ref_service_schema[desc_type])
                    == len(var_service_schemas[variant][desc_type])
                    for variant in variant_or_synthetic_schemas
                )
            )
            for element_schema_index in range(len(ref_service_schema[desc_type])):
                ref_element_description = ref_service_schema[desc_type][
                    element_schema_index
                ]["description"]
                variant_element_descriptions = {}
                for var_id, variant_service_schema in var_service_schemas.items():
                    variant_element_descriptions[var_id] = variant_service_schema[
                        desc_type
                    ][element_schema_index]["description"]
                yield {
                    "type": desc_type,
                    "reference_description": ref_element_description,
                    "paraphrases": variant_element_descriptions,
                    "service_name": ref_service_schema["service_name"],
                }


def reformat_schemas(
    ref_schemas: list[dict],
    variant_or_synthetic_schemas: dict[str, list[dict]],
) -> dict[Union[str, list[dict], dict[str, list[str]]]]:
    """Maps the SGD schemas to the format::

            {
                'sgd_description': Union[list[dict], dict[str, list[dict]]
            }

    where the ``dict`` objects contain the paraphrases of `sgd_description` as values of
    the `surface_form` key. The mapping is to a dictionary containing service names as keys
    only in situations where the same description appears in different service in the
    reference schema.

    Parameters
    ----------
    ref_schemas:
        List of SGD service schemas, containing the descriptions which are the keys of the
        output lookup table.
    variant_or_synthetic_schemas:
        A mapping from schema variant IDs (str) to list of paraphrased service schemas.
    """

    schema_variants = sorted(
        list(variant_or_synthetic_schemas.keys()), key=lambda x: int(x[-1:])
    )

    descriptions_to_service = defaultdict(list)
    reformatted_schemas = {}
    for description in _paraphrased_description_iterator(
        ref_schemas, variant_or_synthetic_schemas
    ):
        ref_description = description["reference_description"]
        ref_service = description["service_name"]
        descriptions_to_service[ref_description].append(description["service_name"])
        variant_descriptions = []
        for var_id in schema_variants:
            variant_descriptions.append(
                {"surface_form": description["paraphrases"][var_id]}
            )
        if ref_description not in reformatted_schemas:
            reformatted_schemas[ref_description] = variant_descriptions
        # the same description may appear in the schema across multiple
        # services
        else:
            service_to_paraphrases = {ref_service: variant_descriptions}
            descriptions_to_service[ref_description] = [
                s for s in descriptions_to_service[ref_description] if s != ref_service
            ]
            if isinstance(reformatted_schemas[ref_description], dict):
                previous_descriptions = reformatted_schemas.pop(ref_description)
                service_to_paraphrases.update(previous_descriptions)
            else:
                previous_descriptions = reformatted_schemas.pop(ref_description)
                previous_service = descriptions_to_service[ref_description].pop()
                service_to_paraphrases[previous_service] = previous_descriptions
            reformatted_schemas[ref_description] = service_to_paraphrases
    return reformatted_schemas


def _switch_descr_paraphrases_keys(
    turn_examples_index: dict[str, Union[list[dict], dict[str, list[dict]]]],
    orig_service_schemas: list[dict],
    variant_schemas: dict[str, dict],
) -> dict[str, list[dict]]:
    """
    Switch the keys of turn examples index from SGD description to the SGD-X variant
    currently processed.


    Parameters
    ----------
    turn_examples_index
        A mapping of the form::

            {
                'sgd_description': Union[list[dict], dict[str, list[dict]]
            }
        where the ``dict`` objects contain the paraphrases of `sgd_description` as values of
        the `surface_form` key. The mapping is to a dictionary containing service names as keys
        only in situations where the same description appears in different service in the
        reference schem
    orig_service_schemas
        SGD service schemas.
    variant_schemas
        The SGD-X schemas of the variant currently processed (variant ID as key).

    Returns
    -------
    remapped_turn_examples_index
        This maps SGD-X descriptions to examples turns collected from the corpus. Unlike the input index,
        where for a few keys the values are mappings from service names to paraphrase lists because the
        same description appears across multiple SGD services, all the values of the remapped index are lists
        of turn examples because the SGD-X descriptions are assumed unique for all variants.
    """
    sgd_to_sgdx_descr = reformat_schemas(orig_service_schemas, variant_schemas)
    assert any(isinstance(sgd_to_sgdx_descr[d], dict) for d in sgd_to_sgdx_descr)
    remapped_turn_examples_index = {}
    for sgd_description in turn_examples_index:
        if isinstance(sgd_to_sgdx_descr[sgd_description], dict):
            for sgd_service in sgd_to_sgdx_descr[sgd_description]:
                assert int(sgd_service.split("_")[-1]) < 10
                this_service_paraphrases = sgd_to_sgdx_descr[sgd_description][
                    sgd_service
                ]
                assert len(this_service_paraphrases) == 1
                sgdx_description = this_service_paraphrases[0]["surface_form"]
                remapped_turn_examples_index[sgdx_description] = turn_examples_index[
                    sgd_description
                ][sgd_service]
        else:
            paraphrases = sgd_to_sgdx_descr[sgd_description]
            assert len(paraphrases) == 1
            remapped_turn_examples_index[
                paraphrases[0]["surface_form"]
            ] = turn_examples_index[sgd_description]

    return remapped_turn_examples_index


def load_turn_examples(
    shard_path: str, service_schemas: list[dict], config: DictConfig, split: str
) -> Union[dict[str, Union[dict[str, list[dict]], list[dict]]], None]:
    """

    Parameters
    ----------
    service_schemas:
        Service schema for the split processed.
    config:
        Data preprocessing configuration.
    split:
        Split processed.

    Returns
    -------
    turn_examples_index
        A lookup table mapping the descriptions of the schema to be processed
        to list/service mappings of description paraphrases.
    """

    preproc_config = config.preprocessing
    shard_variant = infer_schema_variant_from_path(str(shard_path))

    if hasattr(preproc_config, "descriptions"):
        if "turn" in preproc_config.descriptions.components:
            turns_path = getattr(preproc_config.descriptions.turns_path, split)
            if turns_path is None:
                logger.warning(
                    "Could not find path to turns for enriching descriptions. Only "
                    "schema descriptions will be used!"
                )
            else:
                variant_name = infer_schema_variant_from_path(turns_path)
                turns_path = Path(turns_path)
                descr_paraphrases = load_json(turns_path)
                if preproc_config.descriptions.turns_format == "schema":
                    descr_paraphrases = reformat_schemas(
                        service_schemas, {variant_name: descr_paraphrases}
                    )
                    to_json(
                        descr_paraphrases,
                        Path(".").resolve().joinpath("check_desc_paraphrase_map.json"),
                        sort_keys=False,
                    )
                    return descr_paraphrases
                assert (
                    preproc_config.descriptions.turns_format == "lookup"
                ), "Unknown description paraphrases format!"

                # turn examples lookup table keys are SGD descriptions, so they
                # have to be mapped to SGD-X descriptions for pre-processing to
                # work on SGD-X variants
                if shard_variant != "original":
                    orig_schema_path = Path(
                        preproc_config.descriptions.sgd_dir
                    ).joinpath(split, "schema.json")
                    orig_service_schemas = load_json(orig_schema_path)
                    variant_schemas = {shard_variant: service_schemas}
                    descr_paraphrases = _switch_descr_paraphrases_keys(
                        descr_paraphrases, orig_service_schemas, variant_schemas
                    )
                return descr_paraphrases
    return


def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def _preprocess_intent_name(value: str) -> str:
    """Splits and intent type into its component words.

    Example
    -------
        "FindRestaurant" -> "find restaurant"
    """

    intent_words = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", value)
    return " ".join([w.lower() for w in intent_words])


def _preprocess_slot_name(value: str) -> str:
    """Splits and slot type into its component words.

    Example
    -------
        "has_live_music" -> "has live music"
    """
    return value.replace("_", " ")


def preprocess_element_name(value: str) -> str:
    """Splits intent- or slot-types to individual words"""

    if is_camel_case(value):
        return _preprocess_intent_name(value)
    return _preprocess_slot_name(value)


def postprocess_description(
    description: str, excluded_end_symbols: Optional[list[str]] = None
) -> str:
    """Removes the full stop from the end of the description as the schema
    SGD descriptions are not punctuated."""

    description = description.strip()
    if excluded_end_symbols is None:
        return description
    while description[-1] in excluded_end_symbols:
        description = description[:-1]
    return description
