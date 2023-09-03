#This file used the 'feature.tsv' and 'all.tsv' files to create a dataset of pairs.
#The first element is the tabular data encoded into text, the second is the sentence generated from the template

from os import path
from os import listdir
import pandas
import argparse

def create_dataset(proc_dir, example_count):

    with open(path.join(proc_dir,'feature.tsv')) as f:
        features = pandas.read_csv(f,sep='\t',nrows=int(example_count))

    encoded_records = []
    for index, row in features.iterrows():
        encoded = ""
        for column in features.columns:
            if (column!='ade'):
                encoded += f"<{column}>{row[column]}</{column}>"
        encoded_records.append(encoded.strip())


    with open(path.join(proc_dir,'all.tsv')) as f:
        processed = pandas.read_csv(f,sep='\t')

    sentences= list(processed['sentence'])
    labels = list(processed['label'])

    with open('template_examples.tsv','w') as f:
        f.write("table"+"\t"+"text\n")
        for i in range(len(encoded_records)):
            row = encoded_records[i]+"\t"+sentences[i]+"\n"
            f.write(row)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proc-dir', type=str)
    parser.add_argument('--example-count',type=str)

    args = parser.parse_args()

    create_dataset(args.proc_dir,args.example_count)

if __name__ == "__main__":
    main()

