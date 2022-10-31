from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np

from dst.utils import (
    append_to_values,
    average_nested_dicts,
    default_to_regular,
    nested_defaultdict,
    safeget,
)

logger = logging.getLogger(__name__)

SGD_VARIANTS = ("original", "v1", "v2", "v3", "v4", "v5")
SEEN_SERVICES = {
    "test": frozenset(
        {"Hotels_2", "Movies_1", "RideSharing_2", "Services_1", "Travel_1", "Weather_1"}
    ),
}
UNSEEN_SERVICES = {
    "test": frozenset(
        {
            "Alarm_1",
            "Buses_3",
            "Events_3",
            "Flights_4",
            "Homes_2",
            "Hotels_4",
            "Media_3",
            "Messaging_1",
            "Movies_3",
            "Music_3",
            "Payment_1",
            "RentalCars_3",
            "Restaurants_2",
            "Services_4",
            "Trains_1",
        }
    )
}
ALL_SERVICES = {
    "test": frozenset(set(SEEN_SERVICES["test"]).union(set(UNSEEN_SERVICES["test"]))),
}


def load_metrics(
    paths: list[Path], split: str, schema_variants: tuple[str] = SGD_VARIANTS
) -> list[dict]:
    """Load aggregated metrics output by the SGD evaluator, for each path. For each path, the function will recurse
    into the directories provided  as follows:

            - It will first look for directories specified in `schema_variants`. If such directories are found, \
            then the metrics for each decoding data format are loaded, and stored in the output dictionary as follows::

                {
                    'decoding_data_version': {'sgd_variant': metrics_dict, ...}, ...
                }

            where `metrics_dict` is the metrics dictionary output the by the SGD evaluator. `decoding_data_version`
            is a string indicating which prompt format was used for decoding and 'sgd_variant' is a member of
            `schema_variants`.

            - For all directories that are not in `schema_variants`, the loading function expects that the
            `schema_variants` directories are rooted in them, so the output dictionary will contain

                {
                    'dir_name':
                        {
                            'decoding_data_version': {'sgd_variant': metrics_dict, ...}, ...
                        }
                }

                for each directory name `dir_name` that is found by iterating through the contents of the directory
                rooted at path.

    Parameters
    ----------
    paths
        A list of paths for the metrics to be loaded.
    split
        Which split has been evaluated. Should be one of 'train', 'test', 'dev'.
    schema_variants
        Which SGD variants were evaluated. Defaults to ['original', 'v1', 'v2', 'v3', 'v4', 'v5']


    Returns
    -------
    A nested dict containing the metrics, with structure as described above.
     """

    metrics = []

    def load_metrics_file(metrics_path: Path) -> dict:
        metrics_files = list(metrics_path.glob("*.json"))
        if len(metrics_files) == 0:
            return
        if len(metrics_files) > 1:
            metric_file = metrics_files[-1]
            logging.info(
                f"Found multiple metrics files for path {metrics_path}, selected {metric_file}"
            )
        else:
            metric_file = metrics_files[0]
        with open(metric_file, "r") as f:
            data = json.load(f)
        return data

    for path in paths:
        this_pth_metrics = nested_defaultdict(dict, depth=3)
        for variant_dir in path.iterdir():
            variant = variant_dir.name
            if variant in schema_variants:
                if variant_dir.joinpath(split).exists():
                    versions = [p.name for p in variant_dir.joinpath(split).iterdir()]
                else:
                    versions = []
                for version in versions:
                    metrics_path = variant_dir.joinpath(split, version)
                    m = load_metrics_file(metrics_path)
                    if m is not None:
                        this_pth_metrics[version][variant] = m
            else:
                for v in schema_variants:
                    versions = [
                        p.name for p in variant_dir.joinpath(v, split).iterdir()
                    ]
                    for ver in versions:
                        metrics_path = variant_dir.joinpath(v, split, ver)
                        m = load_metrics_file(metrics_path)
                        if m is not None:
                            this_pth_metrics[ver][variant][v] = m
        metrics.append(default_to_regular(this_pth_metrics))
    return metrics


def get_version(ver: str) -> int:
    """Convert a string of format ``version_{digit}`` into an integer."""
    return int(ver.split("_")[1])


def get_average_metric_value(
    average_metrics: dict,
    experiment_name: str,
    decoding_data_version: str,
    decoding_strategy: str,
    domain_or_service: str,
    metric: str = "joint_goal_accuracy",
    schema_variants: tuple[str] = SGD_VARIANTS,
    return_list: bool = False,
) -> dict[str, Union[float, list[float]]]:
    """Retrieve a value of a metric from the nested dictionary data structure generated by `load_metrics` function."""

    def _format_service_for_variant(domain_or_service: str) -> str:
        nonlocal var  # SGD variant
        if "_" in domain_or_service and var != "original":
            return f"{domain_or_service}{var[-1]}"
        return domain_or_service

    metric_by_variant = {}
    for var in schema_variants:
        if get_version(decoding_data_version) > 7:
            store_key = [
                experiment_name,
                decoding_data_version,
                decoding_strategy,
                var,
                _format_service_for_variant(domain_or_service),
                metric,
            ]
        else:
            store_key = [
                experiment_name,
                decoding_data_version,
                var,
                _format_service_for_variant(domain_or_service),
                metric,
            ]
        value = safeget(average_metrics, *store_key)
        if return_list:
            metric_by_variant[var] = [value]
        else:
            metric_by_variant[var] = value
    return metric_by_variant


def rank_services_by_performance(
    average_metrics: dict,
    experiment_name: str,
    decoding_data_version: str,
    decoding_strategy: str,
    metric: str = "joint_goal_accuracy",
    schema_variants: tuple[str] = SGD_VARIANTS,
    split: str = "test",
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Rank the services according to the performance on a given metric.

    Returns
    _______
    ranked_results
        A nested dictionary with format::

            {
                {'service_type': {'variant': [(service, metric_value], ...}, ...}
            }

        where `variant` is a variant specified in `schema_variants` or ``'sgd_x_average'``.
        `service_type` is one of ``'seen'``, ``'unseen'`` or ``'combined'``.
    """

    assert split == "test", "This function is not implemented for dev set metrics"

    metric_values = {"seen": {}, "unseen": {}, "combined": {}}
    ranked_results = deepcopy(metric_values)

    service_order = []
    for service in ALL_SERVICES[split]:
        service_order.append(service)
        this_service_avg_metrics = get_average_metric_value(
            average_metrics,
            experiment_name,
            decoding_data_version,
            decoding_strategy,
            service,
            metric=metric,
            schema_variants=schema_variants,
            return_list=True,
        )
        sgdx_average = (
            sum(
                this_service_avg_metrics[var][0]
                for var in schema_variants
                if var != "original"
            )
            / 5
        )
        this_service_avg_metrics["sgd_x_average"] = [sgdx_average]
        append_to_values(metric_values["combined"], this_service_avg_metrics)
        if service in SEEN_SERVICES[split]:
            append_to_values(metric_values["seen"], this_service_avg_metrics)
        else:
            append_to_values(metric_values["unseen"], this_service_avg_metrics)

    for service_type in metric_values:
        for variant in list(schema_variants) + ["sgd_x_average"]:
            this_variant_metric_vals = metric_values[service_type][variant]
            sort_indices = np.argsort(this_variant_metric_vals)
            ranked_services = [
                (service_order[i], this_variant_metric_vals[i]) for i in sort_indices
            ]
            ranked_results[service_type][variant] = ranked_services

    return ranked_results


if __name__ == "__main__":
    model = "d3st_concat_corpus_description_oracle"
    decoding_data_version = "version_9"
    decoding_strategy = "random_turn_example"

    keywords = [
        "v15_oracle",
        "v15_corpus_description_oracle",
        "d3st_concat_corpus_description_oracle",
        "concept_concat_corpus_description_oracle",
    ]
    split = "test"
    METRICS_PATH = "/scratches/neuron/dev/d3st/metrics"

    def get_matching_dirs(keyword: str, path: str):
        matches = []
        for dir_ in Path(path).iterdir():
            if keyword in dir_.name:
                print(dir_.name)
                matches.append(dir_)
        return matches

    metrics = {}
    for keyword in keywords:
        metrics[keyword] = load_metrics(
            get_matching_dirs(keyword, METRICS_PATH),
            split,
        )
    average_metrics = {}
    for key in metrics:
        average_metrics[key] = average_nested_dicts(metrics[key], max_depth=10)
    ranked_services = rank_services_by_performance(
        average_metrics, model, decoding_data_version, decoding_strategy
    )
