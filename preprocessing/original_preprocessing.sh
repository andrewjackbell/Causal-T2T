#!/bin/sh
python original_preprocessing.py \
    --dataset-dir=../dataset \
    --dataset-name=dataset.csv \
    --out-dir=original_proc \
    
python report_encoding.py --proc-dir=original_proc --label-file=all.tsv --out-dir=original_proc
