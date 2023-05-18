import sys
import psutil
import json
import shelve
import time
import re
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from sortedcontainers import SortedDict


class InvertedIndex:
    """ Inverted Index constructor """
    def __init__(self, input, outputDict, outputPL, stopwords):
        self.input = input
        self.outputDict= outputDict
        self.outputPL = outputPL
        self.stopwords_list = set()
        with open(stopwords, "r") as reader:
            for word in reader:
                self.stopwords_list.add(word)
        self.porter_stemmer = PorterStemmer()

        
    def index_document(self):
        with open(self.input, "rb") as reader:
            tempPL = SortedDict()
            i = 0
            for text in reader:
                i += 1
                written = False
                document = json.loads(text)
                content = document['contents']
                id = document['id']
                title = document['title']
                stemmedTokens = self.preprocess(content)
                self.constructPL(title, id, stemmedTokens, tempPL)

                if i % 1000 == 0 or psutil.virtual_memory().percent >= 70:
                    self.write_PL_to_db(tempPL)
                    tempPL = SortedDict()
                    written = True

                print(i)
                if i == 1001:
                    break
            
            #Write the leftover generated PL to database if there are under 1000 documents in the last run
            if written == False:
                sh = shelve.open(self.outputPL)
                self.mergePL(sh, tempPL)
                sh.close()

        self.write_dict_to_db()
        return 0


        
    def constructPL(self, title, id, termList, plDict):
        i = 0
        for term in termList:
            if term in plDict:
                if id in plDict[term]:
                    plDict[term][id][0] += 1
                    plDict[term][id][1].append(i)
                else:
                    plDict[term][id] = [1, [i], title]
            else:
                plDict[term] = SortedDict()
                plDict[term][id] = [1, [i], title]
            i += 1
    
    def write_PL_to_db(self, toWritePL):
        sh = shelve.open(self.outputPL)
        self.mergePL(sh, toWritePL)
        sh.close()

    def write_dict_to_db(self):
        shPL = shelve.open(self.outputPL, 'r')
        shDict = shelve.open(self.outputDict)
        for key in shPL:
            shDict[key] = len(shPL[key])
        shDict.close()
        shPL.close()
        
            
    def mergePL(self, targetPL, secondPL):
 
        for term in secondPL:
            for id in secondPL[term]:
                temp = targetPL.get(term, SortedDict())
                temp[id] = secondPL[term][id]
                targetPL[term] = temp


    def query(self, term):
        shDict = shelve.open(self.outputDict, 'r')
        if term in shDict:
            print("Document Frequency: ", shDict[term])
            shPL = shelve.open(self.outputPL, 'r')
            print("Info about the input term with DocID, Term Frequency, Term Postions and DocTitle respectively: ")
            for k, v in shPL[term].items():
                print("DocID: ", k, " - ", v)
                print("Document highlight: ")
                with open(self.input, "rb") as reader:
                    for text in reader:
                        document = json.loads(text)
                        if document['id'] == k:
                            content = document['contents']
                            stemmedTokens = self.preprocess(content)
                            targetPos = v[1][0]
                            print(stemmedTokens[max(targetPos-5, 0) : min(targetPos+5, len(content)-1)])
                            break

        else:
            print("The input term cannot be found.")

    def preprocess(self, text):
        tokenizedWords = self.tokenize(text)
        sys.argv[1]
        stoppedTokens = self.stopwords_removal(tokenizedWords)
        stemmedTokens = self.stem(stoppedTokens)
        return stemmedTokens

    def tokenize(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        return re.compile(r'\w+').findall(soup.get_text())

    def stopwords_removal(self, wordList):
        if sys.argv[1] == 1:
            return [word for word in wordList if word not in self.stopwords_list]
        return wordList
    def stem(self, wordList):
        if sys.argv[2] == 1:
            return [self.porter_stemmer.stem(word) for word in wordList]
        return wordList


def main():
    index = InvertedIndex("trec_corpus_5000.jsonl", "dictionary.db", "posting_lists.db", "stopwords.txt")
    index.index_document()

if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Elapsed time:",elapsed_time, "seconds")