#!/bin/bash

if [ -z ${VERSION+x} ]; then
  echo "Please pass an integer representing the data version to the evaluation command. For example prepend VERSION=1 to
  test data preprocessed in folder */version_1/"
  exit
fi

python -m scripts.train \
  -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json \
  -d /scratches/neuron/dev/d3st/data/preprocessed/original/dev/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/original/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v1/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v2/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v3/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v4/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v5/train/version_$VERSION/data.json \
  -a /scratches/neuron/dev/d3st/configs/main_experiments/training/neuron_train_arguments_qcpg_v15_run1.yaml \
  --ref_dir /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw/original/dev \
  --template_dir /scratches/neuron/dev/d3st/data/interim/blank_dialogue_templates/original/dev \
  --do_inference -vvv

python -m scripts.train \
  -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json \
  -d /scratches/neuron/dev/d3st/data/preprocessed/original/dev/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/original/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v1/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v2/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v3/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v4/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v5/train/version_$VERSION/data.json \
  -a /scratches/neuron/dev/d3st/configs/main_experiments/training/neuron_train_arguments_qcpg_v15_run2.yaml \
  --ref_dir /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw/original/dev \
  --template_dir /scratches/neuron/dev/d3st/data/interim/blank_dialogue_templates/original/dev \
  --do_inference -vvv

python -m scripts.train \
  -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json \
  -d /scratches/neuron/dev/d3st/data/preprocessed/original/dev/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/original/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v1/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v2/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v3/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v4/train/version_$VERSION/data.json \
  -t /scratches/neuron/dev/d3st/data/preprocessed/qcpg/v5/train/version_$VERSION/data.json \
  -a /scratches/neuron/dev/d3st/configs/main_experiments/training/neuron_train_arguments_qcpg_v15_run3.yaml \
  --ref_dir /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/raw/original/dev \
  --template_dir /scratches/neuron/dev/d3st/data/interim/blank_dialogue_templates/original/dev \
  --do_inference -vvv
