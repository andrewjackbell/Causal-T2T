import pandas
import os
import numpy
import re
import argparse

def filter_cases(dict,index_dict):
    to_remove = ["3807044-9","3262811-8","3806869-3","3822806-X","3821696-9"]
    for key in to_remove:
        del dict[key]
    return dict

def normalise_drugs(dict,index_dict):
    psd_index = index_dict['psd_index']
    ssd_index = index_dict['ssd_index']
    for key in dict:
        case = dict[key]
        psd = case[psd_index]
        ssd_terms = case[ssd_index].split(", ")
        ssd_non_duplicates = [term for term in ssd_terms if not term.lower() in psd.lower()]
        if len(ssd_non_duplicates)==0:
            ssd = " "
        elif len(ssd_non_duplicates)<3:
            ssd=", ".join(ssd_non_duplicates)
        else: #Number of drugs limited to 3 or else sentences become too long
            ssd=", ".join(ssd_non_duplicates[:3])
        case[ssd_index]=ssd
        dict[key]=case
    return dict

def normalization(dict,index_dict):
    for key in dict: #This loop checks each case for empty terms
        case = dict[key]
        empty_tests=['unk','unknown','()','other','see']
        for _,index in index_dict.items(): #This loop checks each feature of the case
            terms = case[index].replace(';',',').split(', ')
            relevant_terms = [term for term in terms if not any([test in term.lower() for test in empty_tests])]
            if len(relevant_terms)==0:
                terms_string = " "
            else:
                terms_string = ", ".join(relevant_terms)

            case[index]=terms_string
        dict[key]=case
    
    return dict


def dose_unify(dict, index_dict):
    dose_index = index_dict['dose_index']
    for key in dict:
        if (dict[key][dose_index] == ' '):
            continue
        if 'MG' in str(dict[key][dose_index]):
            res = re.findall(r'\d+\.*\d*', dict[key][dose_index].split('MG')[0])
            if (len(res) == 0):
                value = dict[key]
                value[dose_index] = ' '
                continue
            if (float(max(res)) > 100):
                value = dict[key]
                value[dose_index] = 'larger than 100 MG'
            else:
                value = dict[key]
                value[dose_index] = 'equal or smaller than 100 MG'
            dict[key] = value
        elif 'mg' in str(dict[key][dose_index]):
            res = re.findall(r'\d+\.*\d*', dict[key][dose_index].split('mg')[0])
            if (len(res) == 0):
                value = dict[key]
                value[dose_index] = ' '
                continue
            if (float(max(res)) > 100):
                value = dict[key]
                value[dose_index] = 'larger than 100 MG'
            else:
                value = dict[key]
                value[dose_index] = 'equal or smaller than 100 MG'
            dict[key] = value
        elif 'MILLIGRAM' in str(dict[key][dose_index]):
            res = re.findall(r'\d+\.*\d*', dict[key][dose_index].split('MILLIGRAM')[0])
            if (len(res) == 0):
                value = dict[key]
                value[dose_index] = ' '
                continue
            if (float(max(res)) > 100):
                value = dict[key]
                value[dose_index] = 'larger than 100 MG'
            else:
                value = dict[key]
                value[dose_index] = 'equal or smaller than 100 MG'
            dict[key] = value
        elif 'MICROGRAM' in str(dict[key][dose_index]):
            value = dict[key]
            value[dose_index] = 'equal or smaller than 100 MG'
            dict[key] = value
        elif 'UG' in str(dict[key][dose_index]):
            res = re.findall(r'\d+\.*\d* UG', dict[key][dose_index])
            if len(res) > 0:
                value = dict[key]
                value[dose_index] = 'equal or smaller than 100 MG'
                dict[key] = value
        elif 'MCG' in str(dict[key][dose_index]):
            value = dict[key]
            value[dose_index] = 'equal or smaller than 100 MG'
            dict[key] = value
        elif 'GRAM' in str(dict[key][dose_index]):
            value = dict[key]
            value[dose_index] = 'larger than 100 MG'
            dict[key] = value
        elif 'G' in str(dict[key][dose_index]):
            res1 = re.findall(r'\d+\.*\d* G', dict[key][dose_index])
            res2 = re.findall(r'\d+\.*\d*G', dict[key][dose_index])
            if len(res1) > 0 or len(res2) > 0:
                value = dict[key]
                value[dose_index] = 'larger than 100 MG'
                dict[key] = value
    return dict


def age_unify(dict, index_dict):
    def age_divide(age_year):
        if age_year < 18:
            age = 'younger than 18'
        elif age_year < 40:
            age = '18-39'
        elif age_year < 65:
            age = '40-64'
        elif age_year >= 65:
            age = 'older than 65'
        return age
    age_index = index_dict['age_index']
    age_unit_index = age_index + 1
    for key in dict:
        type = dict[key][age_unit_index]
        if type == ' ':
            continue
        if type == 'Day':
            age_year = float(dict[key][age_index]) / 365
            age = age_divide(age_year)
        elif type == 'Month':
            age_year = float(dict[key][age_index]) / 12
            age = age_divide(age_year)
        elif type == 'Week':
            age_year = float(dict[key][age_index]) / 52
            age = age_divide(age_year)
        elif type == 'Decade':
            age_year = float(dict[key][age_index]) * 10
            age = age_divide(age_year)
        elif type == 'Hour':
            age_year = float(dict[key][age_index]) / 8760
            age = age_divide(age_year)
        elif type == 'Year':
            age_year = float(dict[key][age_index])
            age = age_divide(age_year)

        case = dict[key]
        case[age_index] = age
        dict[key] = case
    return dict


#Route_unify takes the dictionary of cases and removes all the non-relevant routes such as unkown. 
#The first word is extracted and used from the relevant routes.
def route_unify(dict,index_dict):
    route_index = index_dict['route_index']
    for key in dict:
        case = dict[key]
        case_route = str(case[route_index]).lower()
        words = case_route.split()
        if not words:
            case[route_index]=" "
            dict[key]=case
            continue
        first_word = words[0]
        if (first_word):
            case[route_index]=first_word
        else:
            case[route_index]=" "
        dict[key]=case
    return dict

#Weight_unify normalises the weights so they are all in kg and then groups into one of four classes
def weight_unify(dict, index_dict):
    def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
    def is_unit(s):
        return s=="kilogram" or s=="pound"
    def weight_divide(weight_kg):
        if weight_kg<50:
            weight_string = "less than 50"
        elif weight_kg<71:
            weight_string = "50-70"
        elif weight_kg<91:
            weight_string = "71-90"
        else:
            weight_string = "heavier than 90"
        
        return weight_string


    weight_index=  index_dict['weight_index']
    weight_unit_index = index_dict['weight_unit_index']

    for key in dict:
        case = dict[key]
        case_weight = str(case[weight_index]).lower()
        case_unit = str(case[weight_unit_index]).lower()
        if is_number(case_weight) and is_unit(case_unit):
            
            weight_int = int(float(case_weight))
            if case_unit=="pound":
                weight_int = int(weight_int/2.205)    

            weight_string = weight_divide(weight_int)
            unit_string = "kilogram"
        else:
            weight_string = " "
            unit_string = " "

        case[weight_index] = weight_string
        case[weight_unit_index] = unit_string
        dict[key]=case
    return dict

def generate_dataset(dict,target_list,index_dict):
    dose_index = index_dict['dose_index']
    age_index = index_dict['age_index']
    psd_index = index_dict['psd_index']
    ssd_index = index_dict['ssd_index']
    gender_index = index_dict['gender_index']
    indication_index = index_dict['indication_index']
    outcome_index= index_dict['outcome_index']
    route_index = index_dict['route_index']
    weight_index = index_dict['weight_index']
    target_index = index_dict['target_index']

    feature_rows = []
    labels = []
    for key in dict:
        case = dict[key]
        dose = case[dose_index].replace('\n', '')
        age = case[age_index].replace('\n', '')
        psd = case[psd_index].replace('\n', '')
        ssd = case[ssd_index].replace('\n','')
        gender = case[gender_index].replace('\n', '')
        indication = case[indication_index].replace('\n', '')
        outcome = case[outcome_index].replace('\n', '')
        weight = case[weight_index].replace('\n', '')
        route = case[route_index].replace('\n', '')
        target = case[target_index].replace('\n', '')

        #Determining the y label for each case based on whether the case contains liver failure as an ADE
        label = ''
        for target_term in target_list:
            for term in target.split(', '):
                if term == target_term:
                    label = '1'
                    break
            if not label == '1':
                label = '0'

        feature_dict = {}
        feature_dict['patient age'] = age
        feature_dict['patient gender'] = gender
        feature_dict['patient weight'] = weight
        feature_dict['primary suspect drug'] = psd
        feature_dict['secondary suspect drug'] = ssd
        feature_dict['dose'] = dose
        feature_dict['indication'] = indication
        feature_dict['outcome'] = outcome
        feature_dict['route'] = route

        feature_rows.append(feature_dict)
        labels.append(label)

    feature_df = pandas.DataFrame(feature_rows)
    label_df = pandas.DataFrame(labels,columns=['label'])

    return feature_df,label_df

        

def preprocess(dataset_dir, dataset_name, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    article_read = pandas.read_csv(dataset_dir + '/' + dataset_name, delimiter=',', dtype=object)
    dataset = article_read.to_numpy()
    target_list = ['Hepatic failure and associated disorders', 'Acute hepatic failure',
                   'Acute on chronic liver failure', 'Chronic hepatic failure', 'Hepatic failure',
                   'Hepatorenal failure', 'Hepatorenal syndrome', 'Subacute hepatic failure']
    dict = {}
    for case in dataset:
        if str(case[0]) not in dict:
            dict[str(case[0])] = case.copy()
    index_dict = {'psd_index':2,'oad_index':3,'route_index':4,'dose_index':5,
              'ade_index':6,'target_index':6,'outcome_index':7,'gender_index':10,
              'age_index':11,'age_unit_index':12,'weight_index':13,'weight_unit_index':14,
              'indication_index':15,'ssd_index':20,'concomitant_index':21,'interacting_index':22}
    
    #Apply normalisation
    dict = filter_cases(dict,index_dict) #Explicit function to remove cases with 00:00:00, since this is done in the original 
    dict = normalization(dict,index_dict) #Changes terms such as {''} or {'unknown'} to a space {' '}
    dict = normalise_drugs(dict,index_dict) #Removes SSDs that are already in PSD
    dict = dose_unify(dict, index_dict) #As in InferBERT
    dict = age_unify(dict, index_dict) #As in InferBERT
    dict = weight_unify(dict,index_dict) #Normalises weights to kg and puts in one for four categories
    dict = route_unify(dict,index_dict)

    feature_df, label_df = generate_dataset(dict,target_list,index_dict)
    with open(out_dir + '/feature.tsv', 'w', newline='') as write_tsv:
        write_tsv.write(feature_df.to_csv(sep='\t', index=False))
    with open(out_dir + '/labels.tsv', 'w', newline='') as write_tsv:
        write_tsv.write(label_df.to_csv(sep='\t', index=False))
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--out-dir', type=str)

    args = parser.parse_args()

    preprocess(args.dataset_dir,args.dataset_name,args.out_dir)



if __name__ == "__main__":
    main()     
