SEARCH ENGINE SIMULATION

#Overview
	The search engine takes in a query string and search its database (a corpus)
	to return relevant visualized results such as BM25 scores, document similarities,
	relevant topics, information about relevant documents... 

#Python Version: 3.10.7

#Files:
	. final_project.py
	. BM25.py
	. KNN.py
	. NaiveBayes.py
	. trec_corpus_20220301_plain.json
	. dictionary_new.json
	. posting_list_new.json
	. docs_info_new.json
	. train_topics_keywords.tsv
	. train_topics_reldocs.tsv
	
#Packages:
	. pip install sortedcontainers
	. pip install numpy
	. pip install plotly==5.11.0
	. pip install -U scikit-learn
	. pip install networkx[default]
	. Go to https://graphviz.org/download/ -> Download the appropriate version of Graphviz for the running environment 
	-> Add the bin folder path to All-user envirnonment variable "Path"

#Execution: 
	-- python final_project.py
	. Instruction details are displayed within the program
	. Type "ZZEND" to exit the program
	
	
