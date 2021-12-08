import networkx as nx
from DeepWalk.deepwalk import deepwalk

import numpy as np
from gensim.models import Word2Vec

num_walks = 20
walk_length = 30
n_dim = 150
epochs = 25

pathModel = "data/model/model_" + str(n_dim) + "d_" + str(walk_length) + "l_" + str(num_walks)
pathModelLoad = "model_10_10_128"

print("Loading of the graph...")
G = nx.read_edgelist('data/coauthorship.edgelist', delimiter=' ', nodetype=int)
n = len(G.nodes())
print("Number of nodes:", n)

model = deepwalk(G, num_walks, walk_length, n_dim, epochs)
#model = Word2Vec.load(pathModelLoad) 
model.save(pathModel)

DeepWalk_embeddings = np.zeros(shape=(n, n_dim+1))

print("Filling embeddings")
for i, node in enumerate(G.nodes()):
    #try:
    try :
        index = model.wv.key_to_index[node]
        DeepWalk_embeddings[i, 0] = node
        DeepWalk_embeddings[i, 1:]=model.wv[index]

    except KeyError:
        DeepWalk_embeddings[i, 0] = node

np.save("data/Model/DeepWalk_embeddings_" + str(n_dim) + "d_" + str(walk_length) + "l_" + str(num_walks) + ".npy",DeepWalk_embeddings)