from os import path
import pandas
import argparse
from rouge import Rouge


def compare_data(data_dir_1, data_dir_2):

    with open(path.join(data_dir_1,"all.tsv")) as f:
        template_df = pandas.read_csv(f, delimiter='\t')

    with open(path.join(data_dir_2,"all.tsv")) as f:
        generated_df = pandas.read_csv(f, delimiter='\t')

    list_gold = list(template_df['sentence'])
    list_generated = list(generated_df['sentence'])

    # Compute Exact Match (EM)
    em_score = sum([1 for i in range(len(list_gold)) if list_gold[i] == list_generated[i]]) / len(list_gold)

    em_raw = sum([1 for i in range(len(list_gold)) if list_gold[i]!=list_generated[i]])
    rouge = Rouge()
    scores = rouge.get_scores(list_generated, list_gold, avg=True)

    rouge_1_score = scores['rouge-1']['f']
    rouge_2_score = scores['rouge-2']['f']
    rouge_L_score = scores['rouge-l']['f']


    print('Exact Match score:', em_score)
    print("Number of mismatches", em_raw)
    print('ROUGE_L score:', rouge_L_score)
    print('ROUGE_1 score:',rouge_1_score)
    print('ROUGE_2 score:',rouge_2_score)

    average = (rouge_1_score+rouge_2_score+rouge_L_score)/3

    print('ROUGE_ALL:',average)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir-1', type=str)
    parser.add_argument('--data-dir-2', type=str)
    args = parser.parse_args()

    compare_data(args.data_dir_1,args.data_dir_2)

if __name__ == "__main__":
    main()


