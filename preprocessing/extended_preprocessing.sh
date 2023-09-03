#!/bin/sh
python extended_preprocessing.py \
    --dataset-dir=../dataset \
    --dataset-name=dataset.csv \
    --out-dir=extended_proc \
    
python report_encoding.py --proc-dir=extended_proc --label-file=labels.tsv --out-dir=extended_proc
