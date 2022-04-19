#!/bin/bash


# full test decoding

python -m scripts.batch_decode -t /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed/v1/test/version_3/data.json -a /scratches/neuron/dev/d3st/configs/decode_arguments.yaml -c /scratches/neuron/dev/d3st/models/d3st_retrain/version_3/model.1000000 -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json -vvv --test
python -m scripts.batch_decode -t /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed/v2/test/version_3/data.json -a /scratches/neuron/dev/d3st/configs/decode_arguments.yaml -c /scratches/neuron/dev/d3st/models/d3st_retrain/version_3/model.1000000 -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json -vvv --test
python -m scripts.batch_decode -t /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed/v3/test/version_3/data.json -a /scratches/neuron/dev/d3st/configs/decode_arguments.yaml -c /scratches/neuron/dev/d3st/models/d3st_retrain/version_3/model.1000000 -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json -vvv --test
python -m scripts.batch_decode -t /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed/v4/test/version_3/data.json -a /scratches/neuron/dev/d3st/configs/decode_arguments.yaml -c /scratches/neuron/dev/d3st/models/d3st_retrain/version_3/model.1000000 -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json -vvv --test
python -m scripts.batch_decode -t /scratches/neuron/dev/robust_paraphrases/dstc8-schema-guided-dialogue/sgd_x/data/preprocessed/v5/test/version_3/data.json -a /scratches/neuron/dev/d3st/configs/decode_arguments.yaml -c /scratches/neuron/dev/d3st/models/d3st_retrain/version_3/model.1000000 -s /scratches/neuron/dev/d3st/data/raw/sgd/train/schema.json -vvv --test
