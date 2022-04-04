# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate predictions JSON file, w.r.t. ground truth file."""

from __future__ import absolute_import, division, print_function

import json
import os

from absl import app, flags, logging

from dst.evaluation import ALL_SERVICES, get_metrics
from dst.scoring_utils import get_dataset_as_dict

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "prediction_dir",
    None,
    "Directory in which all JSON files combined are predictions of the"
    " evaluation set on a single model checkpoint. We evaluate these JSON files"
    " by DSTC8 metrics.",
)
flags.DEFINE_string(
    "raw_data_dir",
    None,
    "Directory for the downloaded data, which contains the dialogue files"
    " and schema files of all datasets (train, dev, test)",
)
flags.DEFINE_enum(
    "eval_set",
    None,
    ["train", "dev", "test", "test_small", "dev_small"],
    "Dataset split for evaluation.",
)
flags.DEFINE_string(
    "output_metric_file",
    None,
    "Single JSON output file containing aggregated evaluation metrics results"
    " for all predictions files in FLAGS.prediction_dir.",
)
flags.DEFINE_boolean(
    "joint_acc_across_turn",
    False,
    "Whether to compute joint accuracy across turn instead of across service. "
    "Should be set to True when conducting multiwoz style evaluation.",
)
flags.DEFINE_boolean(
    "use_fuzzy_match",
    True,
    "Whether to use fuzzy string matching when comparing non-categorical slot "
    "values. Should be set to False when conducting multiwoz style evaluation.",
)

# Name of the file containing all predictions and their corresponding frame
# metrics.
PER_FRAME_OUTPUT_FILENAME = "metrics_and_dialogues.json"


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


def main(_):
    logging.set_verbosity(logging.INFO)

    in_domain_services = get_in_domain_services(
        os.path.join(FLAGS.raw_data_dir, FLAGS.eval_set, "schema.json"),
        os.path.join(FLAGS.raw_data_dir, "train", "schema.json"),
    )
    with open(
        os.path.join(FLAGS.raw_data_dir, FLAGS.eval_set, "schema.json"), "r"
    ) as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service

    dataset_ref = get_dataset_as_dict(
        os.path.join(FLAGS.raw_data_dir, FLAGS.eval_set, "dialogues_*.json")
    )
    dataset_hyp = get_dataset_as_dict(os.path.join(FLAGS.prediction_dir, "*.json"))

    all_metric_aggregate, _ = get_metrics(
        dataset_ref,
        dataset_hyp,
        eval_services,
        in_domain_services,
        use_fuzzy_match=FLAGS.use_fuzzy_match,
        joint_acc_across_turn=FLAGS.joint_acc_across_turn,
    )
    logging.info("Dialog metrics: %s", str(all_metric_aggregate[ALL_SERVICES]))

    # Write the aggregated metrics values.
    with open(FLAGS.output_metric_file, "w") as f:
        json.dump(
            all_metric_aggregate, f, indent=2, separators=(",", ": "), sort_keys=True
        )
    # Write the per-frame metrics values with the corrresponding dialogue frames.
    with open(os.path.join(FLAGS.prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))


if __name__ == "__main__":
    flags.mark_flag_as_required("prediction_dir")
    flags.mark_flag_as_required("raw_data_dir")
    flags.mark_flag_as_required("eval_set")
    flags.mark_flag_as_required("output_metric_file")
    app.run(main)
