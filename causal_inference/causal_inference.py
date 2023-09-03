import csv
import scipy.stats as st
import statistics
import math
import os
import argparse
from random import sample


class Causal_cell():
    def __init__(self, Feature, value, z_score, probability_do, probability_not_do, p_value, support, condition_list):
        self.Feature = Feature
        self.value = value
        self.z_score = z_score
        self.probability_do = probability_do
        self.probability_not_do = probability_not_do
        self.probability_difference = probability_do - probability_not_do
        self.p_value = p_value
        self.support = support
        self.condition_list = condition_list

    def set_condition_list(self, condition_list):
        self.condition_list = condition_list

    def get_variable_list1(self):
        variable_list = [self.Feature, self.value, self.z_score, self.probability_do, self.probability_not_do,
                         self.probability_difference, self.p_value, self.support]
        return variable_list

    def get_variable_list2(self):
        variable_list = [self.Feature, self.value, self.z_score, self.probability_do, self.probability_not_do,
                         self.probability_difference, self.p_value, self.support, self.condition_list]
        return variable_list

class Feature_info: #Each instance of this class represents one feature, e.g primary suspect drug.
    def __init__(self,name,list_index):
        self.name = name
        self.list_index=list_index
        self.values_dict = {} #Key is the feat. value and the dict value is a list of all the line nos. where it's found
        self.not_empty_dict = {} #Key is a line no. and value is true iff the feature is not empty on that line
        self.gap_dict = {}
        self.in_dict = {}
        self.out_dict = {}
    
    def update_info(self, index, value):
        #index is the line no. in feature file
        #value is the value of self's feature on the line no. given by index
        if value!=' ':
            self.not_empty_dict[index]=True
        items = value.split(', ')
        for item in items:
            if item in self.values_dict:
                self.values_dict[item].append(index)
            else:
                self.values_dict[item]=[index]

def causal_tree(endpoint_name,feature_file, probability_file, threshold, condition_list):

    features_list = []
    probability_dict={}
    endpoint_index=None
    with open(feature_file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for index, line in enumerate(tsvreader):
            if index == 0: #First line contains names of features
                for i,feature_name in enumerate(line):
                    if feature_name==endpoint_name:
                        endpoint_index = i
                    new_feature_info = Feature_info(feature_name,i)
                    features_list.append(new_feature_info)
            elif not all([term in line for term in condition_list]): #Line not processed if not all conditions are met
                continue
            else:
                for i,value in enumerate(line):
                    if i==endpoint_index:
                        continue
                    this_feature_info = features_list[i]
                    this_feature_info.update_info(index,value)

    with open(probability_file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for i,line in enumerate(tsvreader):
            probability_dict[i + 1] = line

    def compute_gap(value_dict, p_dict, not_empty_dict):
        dict = {}
        dict_in = {}
        dict_out = {}

        def compute_Z_score(in_list, out_list):
            L1 = []
            for term in in_list:
                L1.append(float(term[1]))
            L2 = []
            for term in out_list:
                L2.append(float(term[1]))
            s1 = statistics.variance(L1)
            l1 = s1 / len(L1)
            s2 = statistics.variance(L2)
            l2 = s2 / len(L2)
            l = l1 + l2
            m = math.sqrt(l)
            m1 = statistics.mean(L1)
            m2 = statistics.mean(L2)
            return (m1 - m2) / m

        def compute_expecation(list):
            L = []
            for term in list:
                L.append(float(term[1]))
            if not len(L) == 0:
                return sum(L) / len(L)
            else:
                return 0

        for key in value_dict.keys():
            l1 = value_dict[key]
            In = []
            Out = []
            for index in p_dict.keys():
                if index in not_empty_dict:
                    if index in l1:
                        In.append(p_dict[index])
                    else:
                        Out.append(p_dict[index])
            if len(In) >= threshold and len(Out) >= threshold:
                dict[key] = compute_Z_score(In, Out)
                dict_in[key] = compute_expecation(In)
                dict_out[key] = compute_expecation(Out)
        return dict, dict_in, dict_out

    total_dict = {}

    for i,this_feature_info in enumerate(features_list):
        values_dict = this_feature_info.values_dict
        not_empty_dict = this_feature_info.not_empty_dict
        gap_dict, in_dict, out_dict = compute_gap(values_dict, probability_dict, not_empty_dict)
        this_feature_info.gap_dict=gap_dict
        this_feature_info.in_dict=in_dict
        this_feature_info.out_dict=out_dict
        total_dict.update(**gap_dict)

    cell_list = []
    for key in total_dict:
        z_score = total_dict[key]
        p_values = 1 - st.norm.cdf(z_score)
        if not p_values <= 0.05:
            continue
        
        for i,this_feature_info in enumerate(features_list):
            this_gap_dict = this_feature_info.gap_dict
            if key in this_gap_dict: 
                this_values_dict = this_feature_info.values_dict
                this_in_dict = this_feature_info.in_dict
                this_out_dict = this_feature_info.out_dict
                name = this_feature_info.name
                cell = Causal_cell(name, key, total_dict[key], this_in_dict[key], this_out_dict[key], p_values,
                               len(this_values_dict[key]), condition_list)
        cell_list.append(cell)
    return cell_list


def causal_inference(endpoint_name,probability_file, feature_file, out_dir, threshold=100):
    condition_list = []
    cell_list = causal_tree(endpoint_name,feature_file, probability_file, threshold, condition_list)

    def myFunc(e):
        return e.get_variable_list1()[2]

    cell_list.sort(reverse=True, key=myFunc)
    # write csv file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    title = ['Feature', 'value', 'z score', 'probability of do value', 'probability of not do value',
             'probability difference', 'p value', 'support']
    with open(out_dir + '/root' + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(title)
        for cell in cell_list:
            a_list = cell.get_variable_list1()
            spamwriter.writerow(a_list)


def generate_causal_tree(endpoint_name,probability_file, feature_file, out_dir, threshold=100):
    def myFunc(e):
        return e.get_variable_list1()[2]

    condition_list = []
    cell_list = causal_tree(endpoint_name,feature_file, probability_file, threshold, condition_list)
    cell_list.sort(reverse=True, key=myFunc)
    stack = cell_list
    total_cell_list = []

    while not stack == []:
        one_cell = stack[0]
        condition_term = one_cell.get_variable_list2()[1]
        condition_list = one_cell.get_variable_list2()[8].copy()
        if one_cell.get_variable_list2()[3] >= 0.5:
            condition_list.append(condition_term)
            if len(condition_list) > 3:
                stack.pop(0)
                continue
            cell_list = causal_tree(endpoint_name,feature_file, probability_file, threshold, condition_list)
            cell_list.sort(reverse=True, key=myFunc)
            total_cell_list.append(one_cell)
            stack.pop(0)
            stack = stack + cell_list
        else:
            stack.pop(0)

    title = ['level', 'route', 'Feature', 'value', 'z score', 'probability of do value', 'probability of not do value',
             'probability difference', 'p value', 'support']
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(out_dir + '/causal_tree.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(title)
        current_level = 0
        for cell in total_cell_list:
            a_list = cell.get_variable_list2()
            route = '->'.join(a_list[-1])
            line = []
            line.append(len(a_list[-1]))
            line.append(route)
            line = line + cell.get_variable_list1()
            if current_level < len(a_list[-1]):
                current_level = len(a_list[-1])
                spamwriter.writerow([])
            spamwriter.writerow(line)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--probability-file', type=str)
    parser.add_argument('--feature-file', type=str)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--endpoint', type=str)

    args = parser.parse_args()
    causal_inference(args.endpoint,args.probability_file, args.feature_file, args.out_dir)
    generate_causal_tree(args.endpoint,args.probability_file, args.feature_file, args.out_dir)


if __name__ == "__main__":
    main()
