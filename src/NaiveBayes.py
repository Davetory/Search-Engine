
import csv
import json
from math import log

class NaiveBayes:
    def __init__(self, dict, info):
        self.dict = dict
        self.info = info
        self.train_dict = {}
        self.prob_table = {}
        self.c_tf = {}
        self.total_docs = 0
        with open(info, 'rb') as reader:
            docs_list = json.load(reader)
            for docs in docs_list:
                self.total_docs += 1



   
    def fit(self, X_train, y_train):
        with open(X_train, 'r') as tsv_in:
            tsv_reader = csv.reader(tsv_in, delimiter='\t')

            for record in tsv_reader:
                self.train_dict[record[1]] = [[], []]
                with open(self.info, "rb") as reader:
                    docs_list = json.load(reader)
                    for docs in record[2].split(','):
                        if docs in docs_list:
                            self.train_dict[record[1]][0].append(docs)


        with open(y_train, 'r', encoding="utf-8") as tsv_in_2:
            tsv_reader_2 = csv.reader(tsv_in_2, delimiter='\t')

            for record in tsv_reader_2:
                self.train_dict[record[1]][1]= record[2].split(',')

        for c in self.train_dict:
            self.c_tf[c] = 0
            with open(self.dict, 'rb') as reader_2:
                index = json.load(reader_2)
                for term in self.train_dict[c][1]:
                    if term in index:
                        self.c_tf[c] += (index[term] + 1)
                
                for term in self.train_dict[c][1]:
                    if term in index:
                        self.prob_table[term] = ((index[term] + 1) / self.c_tf[c])


    def predict(self, query):
        result = {}
        existed = False
        for c in self.train_dict:
            prob_category = log(len(self.train_dict[c][0]) / self.total_docs)
            prob_query = 0
            for word in query.lower().split(' '):
                if word in self.train_dict[c][1]:
                    existed = True
                    prob_query += log(self.prob_table[word])
                else:
                    prob_query += log(1 / self.c_tf[c])


            res_prob = prob_category + prob_query
            result[c] = res_prob
        if not existed:
            print("\nCannot find the respective topics from training data. Query might be non-relevant.\n")
            return "None"
        return max(result, key=result.get)
            

  