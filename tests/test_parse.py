import json
import logging
import pathlib
from distutils.dir_util import copy_tree
from operator import itemgetter

import pytest
from omegaconf import DictConfig, OmegaConf

from dst.parser import parse
from dst.scoring_utils import get_dataset_as_dict
from dst.utils import Schema

logger = logging.getLogger(__name__)

PROCESSED_REFERENCES_ROOT_PATH = pathlib.Path(
    "/scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed"
)
DIALOGUE_TEMPLATES_ROOT_PATH = pathlib.Path(
    "/scratches/neuron/dev/d3st/data/interim/blank_dialogue_templates/"
)
SCHEMA_PATH_ROOT = pathlib.Path(
    "/scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw"
)
ROOT_TEST_OUTPUTS = pathlib.Path(
    "/scratches/neuron/dev/d3st/tests/outputs"
)
if not ROOT_TEST_OUTPUTS.exists():
    ROOT_TEST_OUTPUTS.mkdir(exist_ok=False, parents=True)


def setup_inputs(schema_variant: str, split: str, data_version: str) -> dict:
    schema_path = SCHEMA_PATH_ROOT.joinpath(schema_variant, split, "schema.json")
    references_path = PROCESSED_REFERENCES_ROOT_PATH.joinpath(
        schema_variant, split, data_version, "data.json"
    )
    with open(references_path, 'r') as f:
        references = json.load(f)
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return {
        'schema': schema,
        'schema_obj': Schema(schema_path),
        'preprocessed_references': references,
    }


def patch_experiment_config(
        variant: str,
        split: str,
        model_input_data_version: str,
        model_name_or_path: str
) -> DictConfig:

    preprocessing_config = OmegaConf.load(
        PROCESSED_REFERENCES_ROOT_PATH.joinpath(
            variant, split, model_input_data_version, "preprocessing_config.yaml"
        )
    )
    conf = OmegaConf.create()
    conf.decode = {}
    conf.data = {}
    conf.data.version = int(model_input_data_version.split("_")[1])
    conf.data = preprocessing_config
    conf.decode.model_name_or_path = model_name_or_path
    return conf


def reformat_references(preprocessed_references: dict) -> dict:
    """Reformat preprocessed references from a given split to model predictions format."""

    predictions = {}
    for dialogue_id in preprocessed_references:
        predictions[dialogue_id] = {}
        for idx, turn in enumerate(preprocessed_references[dialogue_id]):
            predictions[dialogue_id][str(idx)] = {}
            predictions[dialogue_id][str(idx)]["utterance"] = turn["user_utterance"]
            for service in turn["frames"]:
                expected_output = turn["frames"][service]["expected_output"]
                predictions[dialogue_id][str(idx)][service] = {'predicted_str': f" {expected_output} <EOS>"}
    return predictions


def get_metrics_inputs(hyp_path: pathlib.Path, ref_path: pathlib.Path):

    hyp_data = get_dataset_as_dict(
        str(hyp_path.joinpath("dialogues_*.json")),
    )
    ref_data = get_dataset_as_dict(
        str(ref_path.joinpath("dialogues_*.json")),
    )
    data = {"dataset_hyp": hyp_data, "dataset_ref": ref_data}
    return data


MODEL_INPUT_DATA_VERSION = ['version_5']
VARIANTS = ['original', 'v1', 'v2', 'v3', 'v4', 'v5']
# VARIANTS= ['original']
SPLITS = ['train', 'dev', 'test']
# SPLITS = ['dev']

@pytest.mark.parametrize("setup_tmp_directories", [ROOT_TEST_OUTPUTS, ], indirect=True)
@pytest.mark.parametrize("model_name_or_path", ['google/t5-v1_1-base', ], ids="model_name_or_path={}".format)
@pytest.mark.parametrize("variant", VARIANTS, ids="variant={}".format)
@pytest.mark.parametrize("split", SPLITS, ids="split={}".format)
@pytest.mark.parametrize(
    "model_input_data_version",
    MODEL_INPUT_DATA_VERSION,
    ids="model_input_data_version={}".format,
)
def test_parse(setup_tmp_directories, split: str, variant: str, model_input_data_version: str, model_name_or_path: str):

    parser_inputs = setup_inputs(variant, split, model_input_data_version)
    experiment_config = patch_experiment_config(
        variant,
        split,
        model_input_data_version,
        model_name_or_path
    )
    dialogue_templates_dir = DIALOGUE_TEMPLATES_ROOT_PATH.joinpath(variant, split)
    parser_output_dir = ROOT_TEST_OUTPUTS.joinpath(
        variant, split, model_input_data_version
    )
    copy_tree(str(dialogue_templates_dir), str(parser_output_dir))
    preprocessed_references = parser_inputs["preprocessed_references"]
    predictions = reformat_references(preprocessed_references)
    parse(
        parser_inputs["schema"],
        predictions,
        preprocessed_references,
        parser_output_dir,
        experiment_config
    )
    test_data = get_metrics_inputs(
        parser_output_dir,
        SCHEMA_PATH_ROOT.joinpath(variant, split)
    )
    hyps, refs = test_data['dataset_hyp'], test_data['dataset_ref']
    schema = parser_inputs["schema_obj"]
    for dial_id in refs:
        hyp_dialogue = hyps[dial_id]
        ref_dialogue = refs[dial_id]
        assert hyp_dialogue["dialogue_id"] == ref_dialogue["dialogue_id"]
        for hyp_turn, ref_turn in zip(hyp_dialogue["turns"], ref_dialogue["turns"]):
            hyp_turn["frames"].sort(key=itemgetter("service"))
            ref_turn["frames"].sort(key=itemgetter("service"))
            for hyp_frame, ref_frame in zip(hyp_turn["frames"], ref_turn["frames"]):
                assert hyp_frame["service"] == ref_frame["service"]
                if "state" not in ref_frame:
                    assert "state" not in hyp_frame
                else:
                    assert hyp_frame["state"]["active_intent"] == ref_frame["state"]["active_intent"]
                    assert sorted(hyp_frame["state"]["requested_slots"]) == sorted(ref_frame["state"]["requested_slots"])
                    assert ref_frame["state"]["slot_values"].keys() == hyp_frame["state"]["slot_values"].keys()

                    for ref_slot in ref_frame["state"]["slot_values"]:
                        ref_slot_values = ref_frame["state"]["slot_values"][ref_slot]
                        hyp_slot_values = hyp_frame["state"]["slot_values"][ref_slot]
                        if model_input_data_version == 'version_1':
                            assert set(hyp_slot_values).issubset(ref_slot_values)
                        else:
                            if ref_slot in schema.get_service_schema(ref_frame["service"]).all_categorical_slots:
                                assert sorted(hyp_slot_values) == sorted(ref_slot_values)
                            else:
                                lowercased_refs = [r.lower() for r in ref_slot_values]
                                if model_input_data_version == 'version_5':
                                    assert sorted(hyp_slot_values) == sorted(lowercased_refs)
                                else:
                                    assert set(sorted(hyp_slot_values)).issubset(sorted(lowercased_refs))
