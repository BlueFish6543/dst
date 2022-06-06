#!/bin/bash

if [ -z ${SHARDS+x} ]; then
  echo "SHARDS has not been specified, all variants and the original schema will be parsed"
  SHARDS="original v1 v2 v3 v4 v5"
fi
if [ -z ${EXPERIMENTS+x} ]; then
  echo "Please pass the names of the experiments to be parsed and scored as a space separated string to the evaluation command by prepending EXPERIMENTS=\"experiment_1 experiment_2 \"  variable."
  exit
fi
if [ -z ${STEPS+x} ]; then
  echo "Please specify a space separated string indicating which steps are to be decoded by prepending STEPS= to the evaluation command. Should have the same len as EXPERIMENTS"
  exit
fi
if [ -z ${SPLIT+x} ]; then
  echo "SPLIT variable was not specified, defaulting to SPLIT=test"
  SPLIT="test"
fi
if [ -z ${VERSIONS+x} ]; then
  echo "Please specify the input data version for each experiment. For example prepend VERSION=\"1 1\" if you are parsing and scoring two experiments trained with input data version 1"
  exit
fi

schema_variants=($SHARDS)
sgd_step=($STEPS)
experiments=($EXPERIMENTS)
input_versions=($VERSIONS)
NUMBER_OF_EXPERIMENTS="${#experiments[@]}"
NUMBER_OF_VERSIONS="${#input_versions[@]}"
NUMBER_OF_STEPS="${#sgd_step[@]}"

if [ "$NUMBER_OF_EXPERIMENTS" -ne "$NUMBER_OF_VERSIONS" ]; then
  echo "Number of models is $NUMBER_OF_EXPERIMENTS but only got $NUMBER_OF_VERSIONS in the data versions array so cannot determine version. Aborting."
  exit
fi
if [ "$NUMBER_OF_EXPERIMENTS" -ne "$NUMBER_OF_VERSIONS" ]; then
  echo "Number of models is $NUMBER_OF_EXPERIMENTS but only got $NUMBER_OF_STEPS elements specifying which step to decode for each model. Please specify which step to decode for each experiment.Aborting."
  exit
fi

HYPS_BASE_DIR=hyps
echo "Base directory where hypotheses are found is $HYPS_BASE_DIR"
for i in "${!experiments[@]}"; do
  experiment="${experiments[i]}"
  step="${sgd_step[i]}"
  input_version="${input_versions[i]}"
  echo "Parsing predictions of model $experiment at step $step"
  echo "Converting decoder output to SGD format ..."
  for variant in "${schema_variants[@]}"; do
    echo "Parsing schema variant: $variant"
    python -m scripts.parse \
      --belief_path "$HYPS_BASE_DIR"/"$experiment"/"$variant"/"$SPLIT"/version_"$input_version"/model."$step" \
      --schema_path /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw/"$variant"/"$SPLIT"/schema.json \
      --template_dir data/interim/blank_dialogue_templates/"$variant"/"$SPLIT" \
      --test_data /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed/"$variant"/"$SPLIT"/version_"$input_version"/data.json -vvv
    mkdir -p "metrics/$experiment/$variant/$SPLIT/version_$input_version"
  done
  echo "Scoring hypotheses..."
  for variant in "${schema_variants[@]}"; do
    echo "Scoring model $experiment at step $step, schema variant $variant... "
    python -m scripts.score \
      --prediction_dir "$HYPS_BASE_DIR"/"$experiment"/"$variant"/"$SPLIT"/version_"$input_version"/model."$step" \
      --raw_data_dir /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw/"$variant" \
      --eval_set "$SPLIT" \
      --output_metric_file metrics/"$experiment"/"$variant"/"$SPLIT"/version_"$input_version"/model_"$step"_metrics.json
  done
done
