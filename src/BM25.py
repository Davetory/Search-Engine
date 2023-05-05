
import json
from math import log


class BM25Rank:
    def __init__(self, index, info, k, b):
        self.index = index
        self.info = info
        self.k = k
        self.b = b
        self.total_docs = 0
        self.total_length = 0
        self.scores = {}
        with open(self.info, 'rb') as reader:
            docs_list = json.load(reader)
            for docs in docs_list:
                self.total_docs += 1
                self.total_length += docs_list[docs]

    def rank(self, query):
        existed = False
        with open(self.info, 'r') as reader:
            docs_list = json.load(reader)
            for term in query.split(' '):
                existed = self.RSV(term, docs_list, existed)
        if not existed:
            return None
        return self.scores

        
    def RSV(self, term, docs_list, existed): 
        avdl = self.total_length / self.total_docs
        idf = 0
        tf = 0
        t = term.lower()
        with open(self.index, 'r') as reader:
            posting_list = json.load(reader)

            if t in posting_list:
                existed = True
                for document in posting_list[t]:
                    dl = docs_list[document]
                    tf = posting_list[t][document][0]
                    idf = log(self.total_docs / len(posting_list[t]))
                    self.scores[document] = idf * (((self.k+1)*tf) / (self.k*((1-self.b) + self.b*(dl/avdl))+tf))
        return existed









                

                