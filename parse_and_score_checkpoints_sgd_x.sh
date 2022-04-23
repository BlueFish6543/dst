#!/bin/bash

declare -a sgd_step=(1840000)
declare -a schema_variants=("original" "v1" "v2" "v3" "v4" "v5")

if [ -z ${EXPERIMENT+x} ]; then
  echo "Please pass the experiment name to the evaluation command by prepending EXPERIMENT=my_experiment_name variable."
  exit
fi

if [ -z ${SPLIT+x} ]; then
  echo "Please pass the split to the evaluation command by prepending SPLIT=my_split_name variable. The split name
  should be one of 'dev', 'dev_small' or 'test'."
  exit
fi

if [ -z ${VERSION+x} ]; then
  echo "Please pass an integer representing the data version to the evaluation command. For example prepend VERSION=1 to
  test data preprocessed in folder */version_1/"
  exit
fi


HYPS_BASE_DIR=hyps
echo "Base directory where hypotheses are found is $HYPS_BASE_DIR"
echo "Converting decoder output to SGD format ..."
for step in "${sgd_step[@]}"; do
  echo "Parsing predictions of model at step $step"
  for variant in "${schema_variants[@]}"; do
    echo "Parsing schema variant: $variant"
    python -m scripts.parse \
      --belief_path "$HYPS_BASE_DIR"/"$EXPERIMENT"/"$variant"/"$SPLIT"/version_"$VERSION"/model."$step" \
      --schema_path /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw/"$variant"/"$SPLIT"/schema.json \
      --template_dir data/interim/blank_dialogue_templates/"$variant"/"$SPLIT" \
      --test_data /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed/"$variant"/"$SPLIT"/version_"$VERSION"/data.json -vvv
    mkdir -p "metrics/$EXPERIMENT/$variant/$SPLIT/version_$VERSION"
  done
done
echo "Scoring hypotheses..."
for step in "${sgd_step[@]}"; do
    for variant in "${schema_variants[@]}"; do
      echo "Scoring model at step $step, schema variant $variant"
      python -m scripts.score \
        --prediction_dir "$HYPS_BASE_DIR"/"$EXPERIMENT"/"$variant"/"$SPLIT"/version_"$VERSION"/model."$step" \
        --raw_data_dir /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw/"$variant" \
        --eval_set "$SPLIT" \
        --output_metric_file metrics/"$EXPERIMENT"/"$variant"/"$SPLIT"/version_"$VERSION"/model_"$step"_metrics.json
    done
done
