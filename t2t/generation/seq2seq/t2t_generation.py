import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
logging.disable(logging.WARNING)
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
#from optimum.pipelines import pipeline
from transformers.pipelines import pipeline
import pandas
import torch
from os import path
from os import makedirs
import argparse

def generate_sentences(data_dir, model_dir, model_name, out_dir):

	if not path.exists(out_dir): 
		makedirs(out_dir) 

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	with open(path.join(data_dir,'encoded_reports.tsv'),'r') as f:
		df=pandas.read_csv(f,sep='\t',nrows=10)

	encoded_records = list(df['report'])
	labels = list(df['label'])

	tokenizer = BartTokenizer.from_pretrained(path.join(model_dir,model_name))
	model = BartForConditionalGeneration.from_pretrained(path.join(model_dir,model_name)).to(device)

	pipe = pipeline("text2text-generation",device=device, model=model,tokenizer=tokenizer)
	results = pipe(encoded_records,max_length=300)
	results_extracted=[o['generated_text'] for o in results]

	dict = {"sentence":results_extracted, "labels":labels}

	full_dataframe = pandas.DataFrame(dict)
	df1 = full_dataframe

	train_dev = df1.sample(frac=0.8)
	train = train_dev.sample(frac=0.8)
	dev = train_dev.drop(train.index)
	test = df1.drop(train_dev.index)

	with open(out_dir + '/train.tsv', 'w+') as write_tsv:
		write_tsv.write(train.to_csv(sep='\t', index=False))
	with open(out_dir + '/dev.tsv', 'w+') as write_tsv:
		write_tsv.write(dev.to_csv(sep='\t', index=False))
	with open(out_dir + '/test.tsv', 'w+') as write_tsv:
		write_tsv.write(test.to_csv(sep='\t', index=False))
	with open(out_dir + '/all.tsv', 'w+') as write_tsv:
		write_tsv.write(df1.to_csv(sep='\t', index=False))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--out-dir', type=str)

    args = parser.parse_args()

    generate_sentences(args.data_dir,args.model_dir,args.model_name,args.out_dir)

if __name__ == "__main__":
    main()