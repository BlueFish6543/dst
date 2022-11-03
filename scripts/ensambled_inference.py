from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

import click

from dst.evaluation import get_metrics, save_metrics
from dst.scoring_utils import (
    get_dataset_as_dict,
    get_eval_services,
    get_in_domain_services,
)
from dst.sgd_utils import dialogue_iterator, split_iterator, turn_iterator
from dst.utils import aggregate_values, append_to_values, nested_defaultdict, to_json

logger = logging.getLogger(__name__)

_SPLIT = "test"
_METRICS_ROOT = "metrics"

service = ""


def frame_sanity_check(frame: dict):
    assert not frame["slots"]
    state = frame["state"]
    for key in state:
        assert not state[key]


def convert_values_to_perc(metrics: dict) -> dict:
    def update_percentage(metric_raw_data: dict, container: dict, key: str):

        for task_name, task_metrics in metric_raw_data.items():
            frame_count = task_metrics["frame_count"]
            for metric_name, value in task_metrics.items():
                if metric_name == "frame_count":
                    continue
                container[variant][key][task_name][metric_name] = round(
                    100 * value / frame_count, 4
                )
                container[variant][key][task_name][f"{metric_name}_counts"] = int(value)
                container[variant][key][task_name]["frame_count"] = int(frame_count)

    def service_metrics_container():
        return nested_defaultdict(dict, depth=2)

    special_keys = ["seen", "unseen", "combined"]
    metrics_perc = defaultdict(
        service_metrics_container,
        {
            variant: defaultdict(
                service_metrics_container,
                {key: nested_defaultdict(dict, depth=2) for key in special_keys},
            )
            for variant in metrics
        },
    )
    for special_key in special_keys:
        for variant in metrics:
            combined_metrics = metrics[variant].pop(special_key)
            update_percentage(combined_metrics, metrics_perc, special_key)

    for variant, service_to_metrics in metrics.items():
        for service, this_service_metrics in service_to_metrics.items():
            assert "_" in service
            update_percentage(this_service_metrics, metrics_perc, service)
    return metrics_perc


def get_current_shard(paths: list[str]) -> str:
    shards = [Path(p).name for p in paths]
    assert len(set(shards)) == 1
    return shards[0]


def get_split_iterators(
    ensemble_dirs: tuple[str],
    hyps_source_dir: str,
    ref_dir: str,
    model: str,
    template_dir: Path,
    variant: str,
    version: str,
    step: str,
) -> List[Iterable]:
    prediction_iterators = [
        split_iterator(_SPLIT, data_pckg_or_path=template_dir),
        split_iterator(_SPLIT, data_pckg_or_path=ref_dir),
    ]
    for dir_ in ensemble_dirs:
        this_member_hyps_path = (
            Path(".")
            .resolve()
            .joinpath(
                hyps_source_dir, model, dir_, variant, _SPLIT, version, f"model.{step}"
            )
        )
        prediction_iterators.append(
            split_iterator(_SPLIT, data_pckg_or_path=this_member_hyps_path)
        )
    return prediction_iterators


def _ensemble_frames(ref_frame: dict, frames: list[dict]) -> dict[str, dict]:
    """Ensemble multiple frames by selecting the most common prediction for
    active intent, requested slots and slot values.

    Parameters
    ----------
    ref_frame
        Reference frame.
    frames
        Frames to be ensembled.

    Returns
    -------
    A dictionary with structure::

        {
            'state': dict, containing the ensemble predicted state of the dialogue in SGD format
            'statistics': dict, containing statistics gathered during ensembling
        }
    """

    def ensemble_intents(ref_frame: dict, frames: list[dict]) -> dict:
        predicted_intents = [f["state"]["active_intent"] for f in frames]
        ref_intent = ref_frame["state"]["active_intent"]
        most_common_intent = Counter(predicted_intents).most_common()[0][0]
        return {
            "result": Counter(predicted_intents).most_common()[0][0],
            "statistics": {
                "most_common_correct": [int(most_common_intent == ref_intent)],
                "gold_in_ensemble_predictions": [int(ref_intent in predicted_intents)],
                "frame_count": [1],
            },
        }

    def ensemble_requested_slots(ref_frame: dict, frames: list[dict]) -> dict:

        req_slot_dicts = [f["state"]["requested_slots"] for f in frames]
        req_slot_hashes = [
            " ".join(sorted(req_slot_dict)) for req_slot_dict in req_slot_dicts
        ]
        most_common_hash = Counter(req_slot_hashes).most_common()[0][0]
        ref_hash = " ".join(sorted(ref_frame["state"]["requested_slots"]))

        return {
            "result": req_slot_dicts[req_slot_hashes.index(most_common_hash)],
            "statistics": {
                "most_common_correct": [int(most_common_hash == ref_hash)],
                "gold_in_ensemble_predictions": [int(ref_hash in req_slot_hashes)],
                "frame_count": [1],
            },
        }

    def ensemble_slot_values(ref_frame: dict, frames: list[dict]) -> dict:
        def create_hash(slots_values: dict) -> str:
            this_dict_hash = ""
            for slot, values in sorted(slots_values.items()):
                val_string = " ".join(sorted([v.lower() for v in values])).strip()
                this_dict_hash += f"{slot} {val_string} "
            return this_dict_hash.strip()

        slot_value_dicts = [f["state"]["slot_values"] for f in frames]
        slot_value_hashes = []
        for svp_dict in slot_value_dicts:
            slot_value_hashes.append(create_hash(svp_dict))
        ref_hash = create_hash(ref_frame["state"]["slot_values"])
        most_common_hash = Counter(slot_value_hashes).most_common()[0][0]
        return {
            "result": slot_value_dicts[slot_value_hashes.index(most_common_hash)],
            "statistics": {
                "most_common_correct": [int(most_common_hash == ref_hash)],
                "gold_in_ensemble_predictions": [int(ref_hash in slot_value_hashes)],
                "frame_count": [1],
            },
        }

    statistics, state = {}, {}
    task_2_ensembled_preds = {
        "active_intent": ensemble_intents(ref_frame, frames),
        "requested_slots": ensemble_requested_slots(ref_frame, frames),
        "slot_values": ensemble_slot_values(ref_frame, frames),
    }
    for task, preds_and_stats in task_2_ensembled_preds.items():
        statistics[task] = preds_and_stats.pop("statistics")
        state[task] = preds_and_stats.pop("result")
    return {"statistics": statistics, "state": state}


def ensemble_turn_predictions(
    ref_dial_iterator: Iterable,
    template_dial_iterator: Iterable,
    ensemble_dials_iterators: list[Iterable],
    in_domain_services: set[str],
) -> dict:
    """Ensemble predictions of multiple models for a given turn.

    Parameters
    ---------
    ref_dial_iterator
        Iterator which yields the reference turns.
    template_dial_iterator
        Iterator which yields the reference turns without annotations.
        These are populated with ensembled predictions.
    ensemble_dials_iterators
        Iterators which yield tuples of turns which are populated with
        predictions instead of gold annotations.
    in_domain_services
        Services that appear both in the train set and the evaluation set.

    Returns
    -------
    statistics
    # TODO: ADD DETAIL HERE.

    """

    statistics = {"combined": {}, "seen": {}, "unseen": {}}
    for template_turn in template_dial_iterator:
        ref_turn = next(ref_dial_iterator)
        ensemble_preds = next(zip(*ensemble_dials_iterators))  # type: tuple[dict]
        for template_frame in turn_iterator(template_turn):
            frame_sanity_check(template_frame)
            global service
            service = template_frame["service"]
            hyp_frames = [
                next(turn_iterator(pred, service=service)) for pred in ensemble_preds
            ]
            # ref frame is necessary checking how the model behaves when
            # prompted with different descriptions
            ref_frame = next(turn_iterator(ref_turn, service=service))
            ensembled_frame = _ensemble_frames(ref_frame, hyp_frames)
            statistics[service] = ensembled_frame.pop("statistics")
            seen_key = "seen" if service in in_domain_services else "unseen"
            append_to_values(statistics["combined"], statistics[service])
            append_to_values(statistics[seen_key], statistics[service])
            template_frame.update(ensembled_frame)
    remove_key = [key for key in statistics if not statistics[key]]
    for k in remove_key:
        statistics.pop(k)
    return statistics


@click.command()
@click.option(
    "-ver",
    "--version",
    "version",
    default=None,
    type=str,
    required=True,
    help="The data version on which the model was trained on. Should be in the format version_*.",
)
@click.option(
    "-vars",
    "--variants",
    "schema_variants",
    default=("original", "v1", "v2", "v3", "v4", "v5"),
    multiple=True,
    help="Which variants will be used for computing SGD-X metrics",
)
@click.option(
    "-h",
    "--hyps_source_dir",
    "hyps_source_dir",
    default="hyps",
    type=str,
    help="Absolute to the path where the hypothesis for the models specified with -m/--models "
    "option are located",
)
@click.option(
    "-mod",
    "--models",
    "models",
    multiple=True,
    required=True,
    help="Names of the experiments for predictions are to be ensembled. This should be "
    "a subset of the names of the directories listed under -h/--hyps_source_dir option.",
)
@click.option(
    "-scheme",
    "--scheme",
    "ensemble_dirs",
    multiple=True,
    required=True,
    default=("v1_examples", "v2_examples", "v3_examples", "v4_examples", "v5_examples"),
    help="The directories where the parsed predictions for each ensemble member are stored. These"
    "folders should be located in each experiment directory listed through -mod/--models",
)
@click.option(
    "-s",
    "--steps",
    "model_steps",
    multiple=True,
    required=True,
    type=int,
    help="The step number identifying the checkpoints which should be parsed and scored, for each model specified"
    "with -mod/--models argument",
)
@click.option(
    "-templates",
    "--template_dir",
    "template_dir",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the directory containing SGD-formatted files without annotations for evaluation sets of the variants "
    "to be evaluated",
)
# necessary for scoring
@click.option(
    "-ref",
    "--ref_dir",
    "ref_dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory where the SGD-format reference files for the evaluation set are saved."
    "This should contain an original subdirectory where the SGD train/dev/test data is"
    "saved and v1-5 subdirectories containing the SGD-X data.",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def ensemble_inference(
    version: str,
    hyps_source_dir: str,
    schema_variants: tuple[str],
    models: tuple[str],
    model_steps: tuple[str],
    template_dir: Path,
    ensemble_dirs: tuple[str],
    ref_dir: Path,
    log_level: int,
):

    handlers = [
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)
    assert len(models) == len(model_steps), "Please specify the step for each model"
    ensemble_name = f"ensemble.1-{len(ensemble_dirs)}"
    # ensemble predictions and save SGD-formatted files
    for model, step in zip(models, model_steps):
        metrics = nested_defaultdict(list, depth=5)
        logger.info(f"Ensembling predictions for model {model}")
        for variant in schema_variants:
            logger.info(f"Ensembling predictions for schema variant {variant}")
            ensemble_metrics_dir = (
                Path(".")
                .resolve()
                .joinpath(_METRICS_ROOT, model, ensemble_name, variant, _SPLIT, version)
            )
            ensemble_hyps_dir = (
                Path(".")
                .resolve()
                .joinpath(
                    hyps_source_dir,
                    model,
                    ensemble_name,
                    variant,
                    _SPLIT,
                    version,
                    f"model.{step}",
                )
            )
            if not ensemble_hyps_dir.exists():
                ensemble_hyps_dir.mkdir(parents=True, exist_ok=True)
            if not ensemble_metrics_dir.exists():
                ensemble_metrics_dir.mkdir(parents=True, exist_ok=True)
            # mapping from dial ID to SGD formatted dialogue required for scoring
            ensembled_predictions = {}  # type: dict[str, dict]
            eval_schema_path = ref_dir.joinpath(variant, _SPLIT, "schema.json")
            train_schema_path = ref_dir.joinpath(variant, "train", "schema.json")
            in_domain_services = get_in_domain_services(
                eval_schema_path, train_schema_path
            )
            eval_services = get_eval_services(eval_schema_path)
            split_iterators = get_split_iterators(
                ensemble_dirs,
                hyps_source_dir,
                ref_dir.joinpath(variant),
                model,
                template_dir.joinpath(variant, _SPLIT),
                variant,
                version,
                step,
            )
            for paths_and_dialogues in zip(*split_iterators):
                paths, dialogues = list(zip(*paths_and_dialogues))
                template_dialogue, ref_dialogue, *hyp_dialogues = dialogues
                # first dialogue is a blank copy where we store the ensembled predictions
                assert len({d["dialogue_id"] for d in dialogues}) == 1
                # keep a reference to the template dialogue
                ensembled_predictions[
                    template_dialogue["dialogue_id"]
                ] = template_dialogue
                template_dial_iterator = dialogue_iterator(
                    ensembled_predictions[template_dialogue["dialogue_id"]],
                    system=False,
                )
                ref_dial_iterator = dialogue_iterator(ref_dialogue, system=False)
                ensemble_dials_iterators = [
                    dialogue_iterator(d, system=False) for d in hyp_dialogues
                ]
                stats = ensemble_turn_predictions(
                    ref_dial_iterator,
                    template_dial_iterator,
                    ensemble_dials_iterators,
                    in_domain_services,
                )
                append_to_values(metrics, {f"{variant}": stats})
            all_metrics_aggregate, _ = get_metrics(
                get_dataset_as_dict(
                    str(eval_schema_path.parent.joinpath("dialogues_*.json"))
                ),
                ensembled_predictions,
                eval_services,
                in_domain_services,
            )
            save_metrics(
                step,
                ensemble_metrics_dir,
                ensemble_hyps_dir,
                {"dataset_hyp": ensembled_predictions},
                all_metrics_aggregate,
            )
        logger.info("Aggregating metrics")
        aggregate_values(metrics, "sum")
        metrics = convert_values_to_perc(metrics)
        to_json(
            metrics,
            Path(".")
            .resolve()
            .joinpath(
                hyps_source_dir, model, ensemble_name, _SPLIT, version, "metrics.json"
            ),
        )


if __name__ == "__main__":
    ensemble_inference()
