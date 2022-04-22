import glob
import json
import pathlib

from absl import logging

PER_FRAME_OUTPUT_FILENAME = "metrics_and_dialogues.json"


def get_dataset_as_dict(file_path_patterns):
    """Read the DSTC8 json dialog data as dictionary with dialog ID as keys."""
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
                    dataset_dict[dial["dialogue_id"]] = dial
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


def get_evaluator_inputs(hyp_path: pathlib.Path, ref_path: pathlib.Path):

    hyp_data = get_dataset_as_dict(
        str(hyp_path.joinpath("dialogues_*.json")),
    )
    ref_data = get_dataset_as_dict(
        str(ref_path.joinpath("dialogues_*.json")),
    )
    data = {"dataset_hyp": hyp_data, "dataset_ref": ref_data}
    return data
