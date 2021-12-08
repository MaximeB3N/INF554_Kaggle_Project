"""
This script is used to generate the final embedding matrix : 
for each author, we have a line {authorID, embedded_author}
Where embedded_author contains several metrics extracted from the graph,
the DeepWalk embedding of the author,
the embedding of the author's set of articles, if found
"""


import numpy as np
import networkx as nx

# Reading vectors extracted from every papers abstracts
vectors=np.load("Global/paper_vectors.npy")
id_abstracts=vectors[:,0].astype(np.int64)

## Various useful dictionnaries
num_id_abstracts=dict([(a,b) for a,b in enumerate(id_abstracts)])
id_abstracts_num=dict([(b,a) for a,b in enumerate(id_abstracts)])

## Getting relation between authors and their papers
with open("data/author_papers.txt") as f:
    authors_papers=f.readlines()

# Creating a matrix that will contain {author ID, vector} with the vector being the normalized sum of all its papers vectors
n_dim_abstracts_vectors=vectors.shape[1]-1
authors_vectors=np.zeros((len(authors_papers), n_dim_abstracts_vectors+1), dtype=np.float64)

# Processing author-paper relation
papers=authors_papers[0].split("\n")[0].split(":")[1].split("-")
s=0
for i,author in enumerate(authors_papers):
    papers=author.split("\n")[0].split(":")[1].split("-")
    vector=np.zeros(n_dim_abstracts_vectors)
    no_fail=False
    for p in papers:
        try:
            vector+=vectors[id_abstracts_num[int(p)], 1:]
            no_fail=True
        except KeyError:
            pass
    if (not no_fail):
        s+=1
    authors_vectors[i][0]=int(author.split(":")[0])
    if (np.linalg.norm(vector)>0):
        vector=vector/np.linalg.norm(vector)
    authors_vectors[i][1:]=vector.copy()
print(s) # Checking how many authors have no abstracts in our database

# Saving obtained matrix
np.save("Global/authors_vectors.npy", authors_vectors)

## Concatenating vectors from the graph (embeddings_improved.npy) and from the abstracts (authors_vectors.npy)

embeddings_improved=np.load("DeepWalk/embeddings_improved.npy")
authors_vectors=np.load("Global/authors_vectors.npy")
G = nx.read_edgelist('data/coauthorship.edgelist', delimiter=' ', nodetype=int)

# Various useful dictionnaries

auth_vec_num_id_author=dict([(a,b) for a,b in enumerate(authors_vectors[:,0])])
id_author_auth_vec_num=dict([(b,a) for a,b in enumerate(authors_vectors[:,0])])

graph_num_id_author=dict([(a,b) for a,b in enumerate(G.nodes)])
id_author_graph_num=dict([(b,a) for a,b in enumerate(G.nodes)])

emb_num_id_author=dict([(a,int(b)) for a,b in enumerate(embeddings_improved[:,0])])
id_author_emb_num=dict([(int(b),a) for a,b in enumerate(embeddings_improved[:,0])])

# Filling in the matrix
n_nodes=G.number_of_nodes()
n_emb=embeddings_improved.shape[1]-1
n_abs=authors_vectors.shape[1]-1
n_dim_tot=1+n_emb+n_abs
full_matrix=np.zeros((n_nodes, n_dim_tot), dtype=np.float64)
for i in range(n_nodes):
    node=graph_num_id_author[i]
    full_matrix[i,0]=node
    full_matrix[i,1:1+n_emb]=embeddings_improved[id_author_emb_num[node],1:].copy()
    full_matrix[i,1+n_emb:]=authors_vectors[id_author_auth_vec_num[node],1:].copy()

# Saving to file
np.save("Global/full_embedding_matrix.npy", full_matrix)