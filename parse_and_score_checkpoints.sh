#!/bin/bash

declare -a sgd_step=(1400000 1440000 1480000 1520000 1560000)

if [ -z ${EXPERIMENT+x} ]; then
  echo "Please pass the experiment name to the evaluation command by prepending EXPERIMENT=my_experiment_name variable."
  exit
fi

echo "Converting decoder output to SGD format ..."
for step in "${sgd_step[@]}"; do
  python -m scripts.parse \
    -d decode/"$EXPERIMENT"/model."$step" \
    -s data/raw/sgd/test/schema.json \
    -t data/interim/sgd/test \
    -j data/preprocessed/sgd/"$EXPERIMENT"/test.json
done
echo "Scoring hypotheses..."
mkdir -p "metrics/$EXPERIMENT"
for step in "${sgd_step[@]}"; do
  python -m scripts.score \
    --prediction_dir decode/"$EXPERIMENT"/model."$step" \
    --raw_data_dir data/raw/sgd \
    --eval_set test \
    --output_metric_file metrics/"$EXPERIMENT"/model_"$step"_metrics.json
done