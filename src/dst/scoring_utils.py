from __future__ import annotations

import glob
import json
import pathlib
from typing import Optional

from absl import logging
from omegaconf import DictConfig

PER_FRAME_OUTPUT_FILENAME = "metrics_and_dialogues.json"


def get_dataset_as_dict(file_path_patterns, decoded_only: Optional[list[str]] = None):
    """Read the DSTC8 json dialog data as dictionary with dialog ID as keys.

    Parameters
    ----------
    decoded_only
        Used for code testing with few dialogues. Should contain valid dialogue IDs.
    """
    dataset_dict = {}
    if isinstance(file_path_patterns, list):
        list_fp = file_path_patterns
    else:
        list_fp = sorted(glob.glob(file_path_patterns))
    for fp in list_fp:
        if PER_FRAME_OUTPUT_FILENAME in fp or "belief" in fp:
            continue
        logging.info("Loading file: %s", fp)
        with open(fp, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dial_id = dial["dialogue_id"]
                    if decoded_only is not None and dial_id not in decoded_only:
                        continue
                    dataset_dict[dial_id] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
    return dataset_dict


def get_service_set(schema_path):
    """Get the set of all services present in a schema."""
    service_set = set()
    with open(schema_path, "r") as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
    return service_set


def get_in_domain_services(schema_path_1, schema_path_2):
    """Get the set of common services between two schemas."""
    return get_service_set(schema_path_1) & get_service_set(schema_path_2)


def load_hyps_and_refs(
    hyp_path: pathlib.Path, ref_path: pathlib.Path, decoded_only: list[str] = None
):
    hyp_data = get_dataset_as_dict(
        str(hyp_path.joinpath("dialogues_*.json")), decoded_only=decoded_only
    )
    ref_data = get_dataset_as_dict(
        str(ref_path.joinpath("dialogues_*.json")), decoded_only=decoded_only
    )
    data = {"dataset_hyp": hyp_data, "dataset_ref": ref_data}
    return data


def setup_evaluator_inputs(
    hyp_path: pathlib.Path, inference_config: DictConfig
) -> dict:
    """Helper function for calling evaluation from training script.

    Returns
    -------
    A mapping containing the positional arguments of the official SGD evaluation script.
    """
    hyps_refs = load_hyps_and_refs(
        hyp_path.resolve(),
        pathlib.Path(inference_config.ref_path),
        decoded_only=inference_config.decode_only
        if inference_config.decode_only
        else None,
    )
    eval_schema_path = inference_config.ref_schema_path
    with open(eval_schema_path, "r") as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service
    in_domain_services = get_in_domain_services(
        eval_schema_path,
        inference_config.orig_train_schema_path,
    )
    logging.info(f"In domain services: {in_domain_services}")
    return {
        "eval_services": eval_services,
        "in_domain_services": in_domain_services,
        "dataset_hyp": hyps_refs["dataset_hyp"],
        "dataset_ref": hyps_refs["dataset_ref"],
    }
