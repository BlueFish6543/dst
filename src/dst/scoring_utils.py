import glob
import json

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
        if PER_FRAME_OUTPUT_FILENAME in fp or 'belief' in fp:
            continue
        logging.info("Loading file: %s", fp)
        with open(fp, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dataset_dict[dial["dialogue_id"]] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
    return dataset_dict