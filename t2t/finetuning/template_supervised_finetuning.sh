#!bin/sh
python finetuning.py \
    --dataset-dir=../training_examples \
    --dataset-name=template_examples.tsv \
    --model-name=template_bart
