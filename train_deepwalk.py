import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd

from src.Embedding.Graph.deepWalk import deepwalk

## Read data 
# Read training data
df_train = pd.read_csv('data/train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]
# Read test data
df_test = pd.read_csv('data/test.csv', dtype={'author': np.int64})
n_test = df_test.shape[0]

# Read graph
G = nx.read_edgelist('data/coauthorship.edgelist', delimiter=' ', nodetype=int)


# Create various useful dictionnaries to access data lines
abs_nodeID_Graph=dict(enumerate(G.nodes))
nodeID_abs_Graph=dict([(b,a) for a,b in enumerate(G.nodes)])

abs_nodeID_Train=dict(df_train["author"])
nodeID_abs_Train=dict([(b,a) for a,b in abs_nodeID_Train.items()])

abs_nodeID_Test=dict(df_test["author"])
nodeID_abs_Test=dict([(b,a) for a,b in abs_nodeID_Test.items()])

abs_hindex_Train=dict(df_train["hindex"])



## Training DeepWalk
n_dim = 128
n_walks = 50
walk_length = 10
model = deepwalk(G, n_walks, walk_length, n_dim) 
model.save("DeepWalk/Models/model_"+str(n_walks)+'_'+str(walk_length)+'_'+str(n_dim))

## Creating embedding matrix : every line contains nodeID and its embedding vector
n=G.number_of_nodes()
DeepWalk_embeddings = np.empty(shape=(n, n_dim+1))

for node in G.nodes:
    DeepWalk_embeddings[nodeID_abs_Graph[node]][0]=node
    DeepWalk_embeddings[nodeID_abs_Graph[node]][1:]=model.wv.get_vector(node)
np.save("DeepWalk/embeddings.npy", DeepWalk_embeddings)

## Creating improved embedding matrix : some metrics are added before the vector (degree, degree^2, mean degree of neighbours, clustering coefficient,...)
n=G.number_of_nodes()
print("Starting")
core_n=nx.core_number(G)
print("Core number OK")
degrees=nx.degree(G)
print("Degree OK")
surr_mean_deg=nx.average_neighbor_degree(G)
print("Neighb degree OK")
coef_clust=nx.clustering(G)
print("Clustering OK")
deg_cent=nx.degree_centrality(G)
print("Degree centrality OK")

DeepWalk_embeddings_i = np.empty(shape=(n, n_dim+7))
print("Filling embeddings")
for node in G.nodes:
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][0]=node
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][1]=core_n[node]
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][2]=degrees[node]
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][3]=degrees[node]*degrees[node]
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][4]=surr_mean_deg[node]
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][5]=coef_clust[node]
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][6]=deg_cent[node]
    DeepWalk_embeddings_i[nodeID_abs_Graph[node]][7:]=model.wv.get_vector(node)
np.save("DeepWalk/embeddings_improved.npy", DeepWalk_embeddings_i)