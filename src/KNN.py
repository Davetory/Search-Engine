import csv
import json

from collections import Counter

class KNN:
    def __init__(self, index, info, k):
        self.index = index
        self.train_dict = {}
        self.info = info
        self.k = int(k)
        self.total_docs = 0
        with open(info, 'rb') as reader:
            docs_list = json.load(reader)
            self.total_docs = len(docs_list)
        
    def fit(self, X_train):
        with open(X_train, 'r', encoding='utf-8') as tsv_in:
            tsv_reader = csv.reader(tsv_in, delimiter='\t')

            for record in tsv_reader:
                self.train_dict[record[1]] = [[], []]
                with open(self.info, "rb") as reader:
                    docs_list = json.load(reader)
                    for docs in record[2].split(','):
                        if docs in docs_list:
                            self.train_dict[record[1]][0].append(docs)

 
    def predict(self, scores):
        neighbor_class = {}
        existed = False
        nearest_neighbors = Counter(scores).most_common(self.k)
        for neighbor in nearest_neighbors:
            for category in self.train_dict:
                if neighbor[0] in self.train_dict[category][0]:
                    existed = True
                    if category in neighbor_class:
                        neighbor_class[category] += 1
                    else:
                        neighbor_class[category] = 1
        if not existed:
            print("\nCannot find the respective topics from training data. Query might be non-relevant.\n")
            return "None"
        return max(neighbor_class, key=neighbor_class.get)
        
    



                

                