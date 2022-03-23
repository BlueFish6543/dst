#!/bin/bash

declare -a sgd_step=(1160000  1320000  1520000  280000  440000  640000  800000 1000000  120000   1360000  1560000  320000  480000  680000  840000 1040000  1200000  1400000  160000   360000  520000  720000  880000 1080000  1240000  1440000  200000   40000   560000  760000  920000 1120000  1280000  1480000  240000   400000  600000  80000   960000)

if [ -z ${EXPERIMENT+x} ]; then
  echo "Please pass the experiment name to the evaluation command by prepending EXPERIMENT=my_experiment_name variable."
  exit
fi

if [ -z ${SPLIT+x} ]; then
  echo "Please pass the split to the evaluation command by prepending SPLIT=my_split_name variable. The split name
  should be one of 'dev', 'dev_small' or 'test'."
  exit
fi

HYPS_BASE_DIR=decode
echo "Base directory where hypotheses are found is $HYPS_BASE_DIR"
echo "Converting decoder output to SGD format ..."
for step in "${sgd_step[@]}"; do
  python -m scripts.parse \
    -d "$HYPS_BASE_DIR"/"$EXPERIMENT"/"$SPLIT"/model."$step" \
    -s data/raw/sgd/"$SPLIT"/schema.json \
    -t data/interim/sgd/"$SPLIT" \
    -j data/preprocessed/sgd/"$EXPERIMENT"/"$SPLIT".json
done
echo "Scoring hypotheses..."
mkdir -p "metrics/$EXPERIMENT/$SPLIT"
for step in "${sgd_step[@]}"; do
  python -m scripts.score \
    --prediction_dir "$HYPS_BASE_DIR"/"$EXPERIMENT"/"$SPLIT"/model."$step" \
    --raw_data_dir data/raw/sgd \
    --eval_set "$SPLIT" \
    --output_metric_file metrics/"$EXPERIMENT"/"$SPLIT"/model_"$step"_metrics.json
done