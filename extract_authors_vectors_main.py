from Dataloader.training import *
import numpy as np



#pathData = "data/x.npy"
pathData = "/Users/maximebonnin/Notebooks/3A_Notebook/INF554/word2vec-pytorch/data/results.npy"
pathAuthor = "data/author_papers.txt"
pathLabels = "data/train.csv"

input, output = load_vectorized_text_numpy_data(pathData)
hash_vector = create_set(input,output)

author_paper = author_to_id_paper(pathAuthor)

hash_authors_vectors = authors_to_vectors(hash_vector,author_paper)

vectors = numpy_author_vector(hash_authors_vectors)

np.save("data/vectors_cbow_own_normalized.npy",vectors)
