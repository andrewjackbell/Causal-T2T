# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import torch
import gc
from llama import Llama
from tqdm import tqdm
import pandas
import argparse

from os import path

def extract_sentence(output_string):
    output_string = " "+output_string
    split = output_string.split('"')
    if len(split)==1:
        print(f"No delimiter found: \n {output_string}\n")
        return "No information given"
    elif len(split)==2:
        print(f"Full delimitation not found: \n {output_string}\n")
        return split[1]
    else:
        return split[1]


def generate(
    report_dir: str,
    out_dir: str,
    model_dir: str,
    tokenizer_path: str,
    temperature: float,
    top_p: float,
    max_seq_len: int,
    max_batch_size: int,
    max_gen_len: Optional[int],
):
    
    with open(path.join(report_dir,'encoded_reports.tsv'),'r') as f:
        df = pandas.read_csv(f,sep='\t')
    
    reports = df['report']
    labels = df['label']

    system_message = """Your role is a table-to-text generator for pharmacovigilance reports.
You take as input a tabular report and give as output a single concise sentence which describes the report. 
Include every term provided in the output sentence. You must always use quotation marks around 
the summary sentence the delimit it. Here is an example: 

Input: <age>18-39</age><dose>larger than 100 MG</dose><psd>amikacin</psd><gender>Male</patient><p>70-90</patient weight><indication>infection</indication><outcome>death</outcome>	

Output: "An 18-39 year old male patient took amikacin to treat infection, leading to death."

""".replace("\n"," ").replace("  "," ")
    count=0
    batches = []
    while count<len(reports):
        diff = len(reports)-count
        if diff>=max_batch_size:
            diff=max_batch_size 
        prompts = [[{"role":"system","content":system_message.strip()}, {"role":"user","content": report.strip()}] for report in reports[count:count+diff]]
        count = count+max_batch_size
        batches.append(prompts)

    generator = Llama.build(
        ckpt_dir=model_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    sentences = []
    for batch in tqdm(batches):
        results = generator.chat_completion(
            batch,  
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        sentences+=[extract_sentence(result['generation']['content']) for result in results]
        del batch
        del results
        gc.collect()
        torch.cuda.empty_cache()

    dict = {"sentence":sentences, "labels":labels}

    full_dataframe = pandas.DataFrame(dict)

    train_dev = full_dataframe.sample(frac=0.8)
    train = train_dev.sample(frac=0.8)
    dev = train_dev.drop(train.index)
    test = full_dataframe.drop(train_dev.index)


    with open(path.join(out_dir,'/train.tsv'), 'w+') as write_tsv:
        write_tsv.write(train.to_csv(sep='\t', index=False))
    with open(path.join(out_dir,'/dev.tsv'), 'w+') as write_tsv:
        write_tsv.write(dev.to_csv(sep='\t', index=False))
    with open(path.join(out_dir,'/test.tsv'), 'w+') as write_tsv:
        write_tsv.write(test.to_csv(sep='\t', index=False))
    with open(path.join(out_dir,'/all.tsv'), 'w+') as write_tsv:
        write_tsv.write(full_dataframe.to_csv(sep='\t', index=False))
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--report-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--tokenizer-path', type=str)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--max-seq-length', type=int)
    parser.add_argument('--max-batch-size', type=int)

    args = parser.parse_args()

    generate(args.report_dir,args.out_dir,args.model_dir,args.tokenizer_path,args.temperature,args.top_p,args.max_seq_length,args.max_batch_size)

if __name__ == "__main__":
    main()
    

