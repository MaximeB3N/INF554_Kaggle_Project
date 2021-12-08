"""
This scripts implements the DeepWalk algorithm, to generate an embedding vector for all authors (dimension 128)
In the last part, the vector is augmented with various metrics that seemed relevant
"""

import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd


## DeepWalk

def random_walk(G, node, walk_length):
    # Simulates a random walk of length "walk_length" starting from node "node"
    walk=[node]
    for _ in range(walk_length):
        node=np.random.choice(list(G.neighbors(node)))
        walk.append(node)
    return walk

def generate_walks(G, num_walks, walk_length):
    # Runs "num_walks" random walks from each node
    walks = []
    for x in G.nodes():
        for _ in range(num_walks):
            walks.append(random_walk(G,x,walk_length))
    np.random.shuffle(walks)
    return walks

def deepwalk(G, num_walks, walk_length, n_dim):
    # Simulates walks and uses the Skipgram model to learn node representations
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)
    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)
    return model


