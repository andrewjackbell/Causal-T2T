#This file used the 'feature.tsv' and 'labels.tsv' files that were made by the generalised preprocessing
#It outputs a tsv file, whose first column is the encoded report and the second is the label (1 or 0)
#This tsv file will be used by the T2T model to generate input for ALBERT (in the same format as in inferbet)


from os import path
from os import listdir
import pandas
import argparse


def encode_reports(proc_dir, label_file, out_dir):

	with open(path.join(proc_dir,'feature.tsv')) as f:
		features = pandas.read_csv(f,sep='\t')
	with open(path.join(proc_dir,label_file)) as f:
		label_df = pandas.read_csv(f,sep='\t') 

	labels = list(label_df['label'])

	encoded_records = []
	for index, row in features.iterrows():
		encoded = ""
		for column in features.columns:
			if (column!='ade' and row[column]!=" "):
				encoded += f"<{column}>{row[column]}</{column}>"
		encoded_records.append(encoded.strip())

	print(len(encoded_records))
	with open(path.join(out_dir,'encoded_reports.tsv'),'w') as f:
		f.write("report"+"\t"+"label\n")
		for i in range(len(encoded_records)):
			row = encoded_records[i]+"\t"+str(labels[i])+"\n"
			f.write(row)
 
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proc-dir', type=str)
    parser.add_argument('--label-file',type=str)
    parser.add_argument('--out-dir', type=str)

    args = parser.parse_args()

    encode_reports(args.proc_dir,args.label_file,args.out_dir)


if __name__ == "__main__":
    main()     






