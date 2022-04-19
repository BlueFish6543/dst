from __future__ import annotations

import json
import math
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Callable

import numpy as np
from prettyprinter import pprint

from dst.utils import aggregate_values, default_to_regular


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


def load_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def nested_defaultdict(default_factory: Callable, depth: int = 1):
    """Creates a nested default dictionary of arbitrary depth with a specified callable as leaf."""
    if not depth:
        return default_factory()
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


def check_processing(
    scores: dict[str, dict[str, dict[str, list[list[float]]]]],
    scores_reduced: dict[str, dict[str, list[list[list[float]]]]],
    schema_variants: list[str],
    model: str,
    split: str,
):
    for var_idx, variant in enumerate(schema_variants):
        for model_step in range(len(scores[model][variant][split])):
            assert math.isclose(
                sum(scores_reduced[model][split][model_step][var_idx]),
                sum(scores[model][variant][split][model_step]),
            )


def get_metric_sensitivity(scores: np.ndarray) -> float:

    scores = scores.T
    n_schemas = scores.shape[1]
    assert n_schemas == 5
    mean = np.mean(scores, axis=1, keepdims=True)
    std = np.sqrt(np.sum((scores - mean) ** 2, axis=1, keepdims=True) / (n_schemas - 1))
    return np.nanmean(std / mean)


def main():
    split = "test"
    METRICS_SOURCE_DIR = "metrics"
    HYPS_SOURCE_DIR = "hyps"
    MODEL_INPUT_DATA_VERSION = "version_3"
    SCHEMA_VARIANTS = ["v1", "v2", "v3", "v4", "v5"]
    # SCHEMA_VARIANTS = ["v1", "v2",]

    # MODELS = ['d3st', 'pegasus_schema_aug']
    MODELS = ["seed_20220303_d3st_sgd_x_oracle"]
    basic_metrics = [
        "joint_goal_accuracy",
        "joint_cat_accuracy",
        "joint_noncat_accuracy",
    ]
    METRICS = {m: basic_metrics for m in MODELS}

    frame_metric_paths = nested_defaultdict(list, depth=3)
    for model in MODELS:
        for variant in SCHEMA_VARIANTS:
            this_model_schema_variant_paths = list(
                Path("..")
                .resolve()
                .joinpath(
                    HYPS_SOURCE_DIR, model, variant, split, MODEL_INPUT_DATA_VERSION
                )
                .glob("model*")
            )
            this_model_schema_variant_paths = sorted(
                this_model_schema_variant_paths,
                key=lambda pth: int(pth.name.split(".")[1]),
            )
            print(
                f"Paths for model {model}, schema variant {variant}, {this_model_schema_variant_paths}"
            )
            frame_metric_paths[model][variant][split].extend(
                [
                    p.joinpath("metrics_and_dialogues.json")
                    for p in this_model_schema_variant_paths
                ]
            )

    frame_metrics = nested_defaultdict(list, depth=3)
    for model in MODELS:
        for variant in SCHEMA_VARIANTS:
            frame_metrics[model][variant][split] = [
                load_json(pth) for pth in frame_metric_paths[model][variant][split]
            ]

    # Metric to use
    metric = "joint_goal_accuracy"
    orig_train_schema_path = (
        Path("..").resolve().joinpath("data/raw/sgd/train/schema.json")
    )
    orig_test_schema_path = (
        Path("..").resolve().joinpath("data/raw/sgd/test/schema.json")
    )
    in_domain_services = get_in_domain_services(
        orig_train_schema_path, orig_test_schema_path
    )
    # Retrieve scores for all models, schema variant and optimization step given a split and input data version
    all_scores = nested_defaultdict(list, depth=3)
    seen_scores = nested_defaultdict(list, depth=3)
    unseen_scores = nested_defaultdict(list, depth=3)
    all_scores_reduced = nested_defaultdict(list, depth=2)
    seen_scores_reduced = nested_defaultdict(list, depth=2)
    unseen_scores_reduced = nested_defaultdict(list, depth=2)
    for model in MODELS:
        (
            all_scores_across_varaints,
            seen_scores_across_variants,
            unseen_scores_across_variants,
        ) = ([], [], [])
        for variant in SCHEMA_VARIANTS:
            for step_idx, this_step_frame_metrics in enumerate(
                frame_metrics[model][variant][split]
            ):
                (
                    this_step_idx_all_scores,
                    this_step_idx_seen_scores,
                    this_step_idx_unseen_scores,
                ) = ([], [], [])
                for dialogue_id, dialogue in this_step_frame_metrics.items():
                    for turn in dialogue["turns"]:
                        if turn["speaker"] == "USER":
                            for frame in turn["frames"]:
                                this_step_idx_all_scores.append(
                                    frame["metrics"][metric]
                                )
                                if frame["service"][:-1] in in_domain_services:
                                    this_step_idx_seen_scores.append(
                                        frame["metrics"][metric]
                                    )
                                else:
                                    this_step_idx_unseen_scores.append(
                                        frame["metrics"][metric]
                                    )
                assert len(this_step_idx_unseen_scores) + len(
                    this_step_idx_seen_scores
                ) == len(this_step_idx_all_scores)
                all_scores[model][variant][split].append(this_step_idx_all_scores)
                seen_scores[model][variant][split].append(this_step_idx_seen_scores)
                unseen_scores[model][variant][split].append(this_step_idx_unseen_scores)
            if not all_scores_across_varaints:
                for model_step in range(len(all_scores[model][variant][split])):
                    all_scores_across_varaints.append(
                        [all_scores[model][variant][split][model_step]]
                    )
            else:
                for model_step in range(len(all_scores[model][variant][split])):
                    all_scores_across_varaints[model_step].append(
                        all_scores[model][variant][split][model_step]
                    )
            if not seen_scores_across_variants:
                for model_step in range(len(seen_scores[model][variant][split])):
                    seen_scores_across_variants.append(
                        [seen_scores[model][variant][split][model_step]]
                    )
            else:
                for model_step in range(len(seen_scores[model][variant][split])):
                    seen_scores_across_variants[model_step].append(
                        seen_scores[model][variant][split][model_step]
                    )
            if not unseen_scores_across_variants:
                for model_step in range(len(unseen_scores[model][variant][split])):
                    unseen_scores_across_variants.append(
                        [unseen_scores[model][variant][split][model_step]]
                    )
            else:
                for model_step in range(len(unseen_scores[model][variant][split])):
                    unseen_scores_across_variants[model_step].append(
                        unseen_scores[model][variant][split][model_step]
                    )
        print("Model", model)
        variant_aggregated_scores = deepcopy(all_scores[model])
        aggregate_values(variant_aggregated_scores, "mean")
        print("Variant scores")
        pprint(default_to_regular(variant_aggregated_scores))
        all_scores_reduced[model][split] = all_scores_across_varaints
        seen_scores_reduced[model][split] = seen_scores_across_variants
        unseen_scores_reduced[model][split] = unseen_scores_across_variants
        check_processing(all_scores, all_scores_reduced, SCHEMA_VARIANTS, model, split)
        check_processing(
            seen_scores, seen_scores_reduced, SCHEMA_VARIANTS, model, split
        )
        check_processing(
            unseen_scores, unseen_scores_reduced, SCHEMA_VARIANTS, model, split
        )

    # convert reduced scores to 2-D tensor containing metrics for all variants for each model optimisation step
    all_scores_arrays = nested_defaultdict(list, depth=2)
    seen_scores_arrays = nested_defaultdict(list, depth=2)
    unseen_scores_arrays = nested_defaultdict(list, depth=2)
    for model in MODELS:
        all_scores_arrays[model][split] = [
            np.asarray(all_variant_scores)
            for all_variant_scores in all_scores_reduced[model][split]
        ]
        seen_scores_arrays[model][split] = [
            np.asarray(all_variants_seen_scores)
            for all_variants_seen_scores in seen_scores_reduced[model][split]
        ]
        unseen_scores_arrays[model][split] = [
            np.asarray(all_variants_unseen_scores)
            for all_variants_unseen_scores in unseen_scores_reduced[model][split]
        ]

    # calculate JGA 1-5
    all_jga_avg = nested_defaultdict(list, depth=2)
    seen_jga_avg = nested_defaultdict(list, depth=2)
    unseen_jga_avg = nested_defaultdict(list, depth=2)
    print(f"Reporting metric: {metric}]")
    for model in MODELS:
        print("Model", model)
        all_jga_avg[model][split] = [
            np.mean(arr) for arr in all_scores_arrays[model][split]
        ]
        seen_jga_avg[model][split] = [
            np.mean(arr) for arr in seen_scores_arrays[model][split]
        ]
        unseen_jga_avg[model][split] = [
            np.mean(arr) for arr in unseen_scores_arrays[model][split]
        ]
        print(
            f"Average JGA for schema variants {SCHEMA_VARIANTS} on all services. Model {model}, split {split}.",
            all_jga_avg[model][split],
        )
        print(
            f"Average JGA for schema variants {SCHEMA_VARIANTS} on seen services. Model {model}, split {split}.",
            seen_jga_avg[model][split],
        )
        print(
            f"Average JGA for schema variants {SCHEMA_VARIANTS} on unseen services. Model {model}, split {split}.",
            unseen_jga_avg[model][split],
        )

    # calculate SS
    all_ss = nested_defaultdict(list, depth=2)
    seen_ss = nested_defaultdict(list, depth=2)
    unseen_ss = nested_defaultdict(list, depth=2)

    for model in MODELS:
        all_ss[model][split] = [
            get_metric_sensitivity(arr) for arr in all_scores_arrays[model][split]
        ]
        seen_ss[model][split] = [
            get_metric_sensitivity(arr) for arr in seen_scores_arrays[model][split]
        ]
        unseen_ss[model][split] = [
            get_metric_sensitivity(arr) for arr in unseen_scores_arrays[model][split]
        ]
        print(
            f"Schema sensitivity for schema variants {SCHEMA_VARIANTS} on all services. Model {model}, split {split}.",
            all_ss[model][split],
        )
        print(
            f"Schema sensitivity for schema variants {SCHEMA_VARIANTS} on seen services. Model {model}, split {split}.",
            seen_ss[model][split],
        )
        print(
            f"Schema sensitivity for schema variants {SCHEMA_VARIANTS} on unseen services. Model {model}, split {split}.",
            unseen_ss[model][split],
        )


if __name__ == "__main__":
    main()
