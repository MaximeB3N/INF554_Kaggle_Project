import os
import numpy as np
from gensim.models import Word2Vec


def random_walk(G, x, walk_length):
    # Simulates a random walk of length "walk_length" starting from node "node"
    walk = [x]
    for i in range(walk_length):
        walk.append(np.random.choice(list(G.neighbors(walk[i]))))

    walk = [str(node) for node in walk]
    return walk

def generate_walks(G, num_walks, walk_length):
    # Runs "num_walks" random walks from each node
    walks = []
    for x in G.nodes():
        for _ in range(num_walks):
            walks.append(random_walk(G,x,walk_length))
        
    np.random.shuffle(walks)
    return walks

def deepwalk(G, num_walks, walk_length, n_dim, epochs):
    # Simulates walks and uses the Skipgram model to learn node representations
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)
    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=os.cpu_count(), hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=epochs)
    return model