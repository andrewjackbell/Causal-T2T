from os import path
import pandas
import numpy as np
import argparse

def results_table(result_dir,result_name):
    with open(path.join(result_dir,result_name,'root.csv'),'r') as f:
        df = pandas.read_csv(f)

    maxes= {}
    idxs = {}
    for index,row in df.iterrows():
        feature = row["Feature"]
        z_value = row["z score"]
        value = row["value"]
        if 'po' not in value.lower() and 'hepatic' not in value.lower():
            if feature in maxes:
                if maxes[feature] < z_value:

                    maxes[feature] = z_value
                    idxs[feature] = index
            else:
                maxes[feature] = z_value
                idxs[feature] = index

    indexes = [idx for feature,idx in idxs.items()]
    
    selected_rows = (df.loc[indexes])[['Feature','value','probability of do value','probability of not do value','z score']]
    print(selected_rows.to_latex(index=False,float_format="%.2f"))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--result-name', type=str)
    args = parser.parse_args()

    results_table(args.result_dir,args.result_name)

if __name__ == "__main__":
    main()