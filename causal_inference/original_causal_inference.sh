#!/bin/sh
ENDPOINT=ade

ALBERT_OUTPUT_DIR=../albert_finetuning/albert_output
ALBERT_OUTPUT_NAME=gpt_supervised #Can be changed to 'gpt_supervised','template_supervised' and 'control'

DATA_DIR=../preprocessing/original_proc
CAUSAL_OUTPUT_DIR=causal_results/${ALBERT_OUTPUT_NAME}

python causal_inference.py \
  --probability-file=$ALBERT_OUTPUT_DIR/$ALBERT_OUTPUT_NAME/test_results.tsv \
  --feature-file=$DATA_DIR/feature.tsv \
  --out-dir=$CAUSAL_OUTPUT_DIR \
  --endpoint=$ENDPOINT

