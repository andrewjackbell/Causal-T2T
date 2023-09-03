torchrun --nproc_per_node 2 llama_t2t.py \
    --report-dir=../../../preprocessing/extended_proc \
    --out-dir=llama_extended_generations \
    --model-dir=llama-2-13b-chat/ \
    --tokenizer-path=tokenizer.model \
    --temperature=0 \
    --top-p=0.2 \
    --max-seq-len=512 \
    --max-batch-size=20
