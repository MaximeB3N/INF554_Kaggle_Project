import numpy as np
import pandas as pd
from tqdm import tqdm

def load_vectorized_text_numpy_data(path):
    """
    Loads a numpy file from the given path and extract the data and ids.

    Args:
        path: The path to the numpy file.

    Returns:
        The data and ids as numpy arrays.
    """
    X = np.load(path)
    ids = X[:, 0]
    X = X[:, 1:]
    return ids, X 

def create_set(input,output):
    """
    Creates a hash table for the given input and output.

    Args:
        input: The input data.
        output: The output data.

    Returns:
        The hash table.
    """
    hash_table = {}
    for i,input_i in enumerate(input):
        hash_table[input_i] = output[i]
    return hash_table


def author_to_id_paper(pathAuthors):
    """
    Finds the id for each author.

    Args:
        pathAuthors: The path to the authors.

    Returns:
        The id for each author.
    """
    with open(pathAuthors) as f:
        authors = f.readlines()

    author_to_papers = {}
    for line in authors:
        id_author = int(line.split(":")[0])
        id_papers = np.array(line.split(":")[1].split("-"), dtype=int)
        author_to_papers[id_author] = id_papers

    return author_to_papers

def authors_to_vectors(hashtable_vectors, hashtable_papers):
    """
    Finds the vectors for the given authors.

    Args:
        hashtable_vectors: The hashtable with the vectors.
        hashtable_papers: The hashtable with the papers.

    Returns:
        The vectors for the given authors.
    """


    hash_authors_vectors = {}
    for author in tqdm(hashtable_papers.keys()):

        shape = len(list(hashtable_vectors.values())[0])
        
        flag = True
        for id_paper in hashtable_papers[author]:
            if id_paper in hashtable_vectors:
                hash_authors_vectors[author] = hashtable_vectors[id_paper]
                flag = False
                break
            
        if flag:
            hash_authors_vectors[author] = np.zeros(shape)
        
        #break
    return hash_authors_vectors


def numpy_author_vector(hash_authors_vectors):

    author_vectors = np.zeros((len(hash_authors_vectors.keys()), 
                                len(list(hash_authors_vectors.values())[0]) + 1)
                                )

    for i, author in tqdm(enumerate(hash_authors_vectors.keys())):
        author_vectors[i, 0] = author
        author_vectors[i, 1:] = hash_authors_vectors[author]
    return author_vectors


def find_input_datas(pathLabels, X, ids):
    """
    Finds the labels for the given ids.

    Args:
        pathLabels: The path to the labels.
        ids: The ids to find the labels for.

    Returns:
        The labels for the given ids.
    """
    df = pd.read_csv(pathLabels).to_numpy()

    indices_Labels = {}
    for tab in df:
        indices_Labels[int(tab[0])] = tab[1] # tab[0] = id, tab[1] = h index
        
    indices_ids = {}
    for i,id in enumerate(ids):
        indices_ids[int(id)] = i # id = id, i = index in X
        

    #print(indices_Labels.keys())
    #print(indices_ids.keys())
    print(len(indices_Labels.keys()))
    indices = np.array([indices_ids[id] for id in indices_Labels.keys() if id in indices_ids.keys()])
    print(len(indices))
    labels = df[:,1]
    inputs = X[indices]
    return inputs, labels
    