import json
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from BM25 import BM25Rank
from NaiveBayes import NaiveBayes
from KNN import KNN


def create_node_trace(G):
    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    node_color = []

    for i, node in enumerate(G.nodes(data=True)):

        x, y = node[1]['pos']
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[1]['text'])
        node_hovertext.append(node[1]['hovertext'])
        node_color.append(node[1]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_hovertext,
        marker=dict(
            showscale=True,
            color=node_color,
            colorbar=dict(
                title='BM25 Score',
                titleside="top",
                tickmode="array",
                tickvals=[min(node_color), sum(node_color)/len(node_color), max(node_color)],
                ticktext=["Low", "Moderate", "High"],
                ticks="outside"
            ),
            colorscale='Sunsetdark',
            size=18,
            line_width=0.5,
        ),
        text=node_text,
        textfont=dict(
            family="serif bold",
            size=14,
            color="cadetblue"
        ),
        textposition="top center",
        visible=False
    )

    return node_trace

def create_edge_trace(G):
    edge_pos = []
    edge_color = []
    
    for edge in G.edges(data=True):
        
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_pos.append([[x0, x1, None], [y0, y1, None]])
        edge_color.append(G.nodes[edge[0]]['color'])


    edge_traces = []
    for i in range(len(edge_pos)):
        
        line_width = 2

        trace = go.Scatter(
            x=edge_pos[i][0], y=edge_pos[i][1],

            marker = dict(line=dict(width=line_width, color=edge_color)),
            mode='lines+markers',
            visible=False
        )
        edge_traces.append(trace)

    return edge_traces


def get_interactive_slider_similarity_graph(square_matrix, slider_values, corpus_dict, query, topic, scores, yaxisrange=None, xaxisrange=None):
    # Modify scores shallow copy
    bm25_scores = scores.copy()
    total_scores = sum(bm25_scores.values())
    for doc in bm25_scores:
        bm25_scores[doc] = bm25_scores[doc] / total_scores


    # Create figure with plotly
    fig = go.Figure()

    slider_dict = {}
    
    total_n_traces = 0
    
    G = nx.Graph()
    create_graph(square_matrix, G)
    
    # for each possible value in the slider, create and store traces (i.e., plots)
    for i, step_value in enumerate(slider_values):
        Gx = G.copy()

        Gx.remove_edges_from([(a, b) for a, b, attrs in Gx.edges(data=True) if attrs["weight"] < step_value])

        node_pos = nx.nx_pydot.graphviz_layout(Gx)

        for node in Gx.nodes(data=True):
            
            node[1]['pos'] = node_pos[node[0]]

            node[1]['color'] = bm25_scores[node[0]]

            node[1]['text'] = "DocID: " + node[0] 

            # node text on hover 
            docTitle = corpus_dict[node[0]][0]
            context = retrieve_context(query, corpus_dict[node[0]][1])
            node[1]['hovertext'] = "<b>Title</b>: \"" + docTitle + "\"<br><b>Context</b>: " + context + "<br><b>BM25 Score</b>: " + str(bm25_scores[node[0]])

        edge_traces = create_edge_trace(Gx)
        node_trace = create_node_trace(Gx) 

        slider_dict[step_value] = edge_traces + [node_trace]
        
        total_n_traces += len(slider_dict[step_value])

        # make sure that the first slider value is active for visualization
        if i == 0:
            for trace in slider_dict[step_value]:
                trace.visible = True

                
    steps = []
    for step_value in slider_values:
        n_traces_before_adding_new = len(fig.data)
        fig.add_traces(slider_dict[step_value])

        step = dict(
            method="update",
            args=[{"visible": [False] * total_n_traces}],
            label=str(round(step_value, 3)),
        )

        n_traces_for_step_value = len(slider_dict[step_value])
        for i in range(n_traces_before_adding_new, n_traces_before_adding_new + n_traces_for_step_value):
            step["args"][0]["visible"][i] = True
        
        steps.append(step)

    # create slider with list of step objects
    slider = [dict(
        active=0,
        steps=steps
    )]

    # add slider to figure and create layout
    fig.update_layout(
        title_text= "<b>Query</b>: \"" + query + "\"<br><b>Topic</b>: " + topic,
        title_x=0.5,
        autosize=False,
        sliders=slider,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=15, l=35, r=5, t=65),
        xaxis=dict(range=xaxisrange, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=yaxisrange, showgrid=False, zeroline=False, showticklabels=False),
        width=1200, height=700,
    )


    return fig

def create_graph(d, g):
   for a, b in d.items():
        g.add_node(a)
        for node, val in b.items():
            g.add_edge(a, node, weight=val)

def retrieve_context(query, text):
        i = 0
        context = "\""
        content = re.compile(r'\w+').findall(text.lower())
        for termx in query.split(' '):
            term = termx.lower()
            if term in content:
                i += 1
                targetPos = content.index(term)
                startWord = content[max(targetPos-4, 0)]
                endWord = content[min(targetPos+4, len(content)-1)]

                targetIndex = 0
                if (text.lower().find(term) > 0):
                    targetWord_match = re.search(r'[^a-z]' + term + r'[^a-z]', text.lower())
                    if not targetWord_match == None:
                        targetIndex = targetWord_match.start() + 1

                startIndex = targetIndex
                endIndex = targetIndex + len(term) - 1

                while startIndex > 0:
                    startIndex -= 1
                    if text[startIndex : (startIndex+len(startWord))].lower() == startWord.lower():
                        break
                while endIndex < len(text):
                    endIndex += 1
                    if text[endIndex : (endIndex+len(endWord))].lower() == endWord.lower():
                        endIndex = endIndex+len(endWord)
                        break

                context += text[(startIndex) : targetIndex] + "<b>" + text[targetIndex : targetIndex+len(term)] + "</b>" + text[(targetIndex+len(term)) : endIndex] + "..."
                if i > 1:
                    context += "<br>"
                if i == 3:
                    break
        context += "\""

        return context

def filter(text):
    text = re.sub(r'[\n]+', ' ', text)
    text = text.encode("ascii", "ignore")
    text = text.decode()
    return text

def visualize(query, scores, topic, k):
    target = dict()
    for docs in Counter(scores).most_common(int(k)):
        target[str(docs[0])] = scores[docs[0]]

    corpus_dict = dict()
    with open("trec_corpus_20220301_plain.json", 'rb') as json_file:
        for line in json_file:
            doc = json.loads(line)
            if str(doc['id']) in target:
                content = filter(doc["plain"])
                corpus_dict[str(doc['id'])] = list()
                corpus_dict[str(doc['id'])].append(doc['title'])
                corpus_dict[str(doc['id'])].append(content)
            if len(corpus_dict) == len(target):
                break
    
    id_mappings = list(corpus_dict.keys())
    corpus_content = list()
    for content in corpus_dict.values():
        corpus_content.append(content[1])

    try:
        tfidf = TfidfVectorizer().fit_transform(corpus_content)  
        cosine_similarity_matrix = tfidf * tfidf.T 
        cosine_similarity_2darray = cosine_similarity_matrix.toarray()
        cosine_similarity_matrix_dict = {}
    
        for i in range(len(id_mappings)):
            cosine_similarity_matrix_dict[id_mappings[i]] = {}
            for j in range(len(id_mappings)):
                cosine_similarity_matrix_dict[id_mappings[i]][id_mappings[j]] = cosine_similarity_2darray[i][j]


        # define threshold values
        slider_steps = np.arange(0.1, 0.8, 0.1)
    
        # get the slider figure
        fig = get_interactive_slider_similarity_graph(
            cosine_similarity_matrix_dict,
            slider_steps,
            corpus_dict,
            query,
            topic,
            target
        )

        print("\nVisualization DONE. The graph should pop up in seconds.\n")
        time.sleep(3)
        # plot the figure
        fig.show()
    except:
        print("\nUnable to proceed with the given query. Terms might be non-existent in the database.\n")

def exitProgram(query):
    if query.lower() == "ZZEND".lower():
            exit()

def bm25():
    query = input("Enter a query (\"ZZEND\" to exit): ")
    exitProgram(query)

    confirm = input("\nThe input query is \"" + query + "\". Confirm? (Y/N) ")
    while confirm.lower() != "Y".lower() or confirm.lower() == "N".lower():
        query = input("Enter a query: ")
        exitProgram(query)
        confirm = input("\nThe input query is \"" + query + "\". Confirm? (Y/N) ")

    print("\nThe input query is \"" + query + "\".\n")
    time.sleep(1.5)
    print("Please wait for up to several minutes for query processing.\n")

    bm25 = BM25Rank("posting_list_new.json", "docs_info_new.json", 0.75, 1.5)
    scores = bm25.rank(query)
    return scores, query
    
def main():

    while (True):
        scores, query = bm25()
        while scores == None:
            print("\nThe query terms might be non-existent in the database. Try another query!\n")
            scores = bm25()
        print("Finished ranking. Please choose a model for text classification (topic).\n")
        time.sleep(1.5)

        model = input("Enter 1 for NaiveBayes / 2 for KNN / any other to proceed without text classification: ")
        time.sleep(1.5)

        k = input("\nEnter number of documents k for training (more than 50 is not recommended): ")
        while not k.isdigit():
            print("Please enter an integer value for k.")
            k = input("\nEnter number of documents k for training (more than 50 is not recommended): ")
        time.sleep(2)

        print("\nAssigning topics to query. Please wait for up to several minutes for text classification.")
        topic = ""

        if (model == "1"):
            NB = NaiveBayes("dictionary_new.json", "docs_info_new.json")
            NB.fit("train_topics_reldocs.tsv", "train_topics_keywords.tsv")
            topic = NB.predict(query)
        if (model == "2"):
            knn = KNN("posting_list_new.json", "docs_info_new.json", k)
            knn.fit("train_topics_reldocs.tsv")
            topic = knn.predict(scores)
        print("\nText Classification DONE.")
        time.sleep(1)
    
        print("\nPlease wait for up to several minutes for result visualization.")
        visualize(query, scores, topic, k)
        
    
if __name__ == "__main__":
    main()