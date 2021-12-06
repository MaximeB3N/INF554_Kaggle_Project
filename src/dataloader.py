import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import torch



def normalise_adjacency(A):
    """Normalise sparse adjacency matrix of a graph"""
    diag = np.asarray(sp.csr_matrix.sum(A, axis=1)).squeeze()
    D = sp.diags(diag, format='csr')
    I = sp.identity(D.shape[0], format='csr')
    fac = sp.linalg.inv(sp.csr_matrix.sqrt(D + I))
    A_normalised = fac @ (A + I) @ fac
    return A_normalised


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_partial_dataset(n_train, n_test, device):
    """Load the dataset"""

    # Load the graph
    G = nx.read_edgelist('data/coauthorship.edgelist', delimiter=' ', nodetype=int)
    nodes = np.array(list(G.nodes))

    # Read h indexes
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    # Select a random permutation
    indices = np.arange(110000)
    indices = np.random.permutation(indices)

    # Split into train and test sets
    y_train = df_train['hindex'][indices[:n_train]]
    authors_train = df_train['author'][indices[:n_train]]
    y_test = df_train['hindex'][indices[-n_test:]].to_numpy()
    authors_test = df_train['author'][indices[-n_test:]]
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Extract subgraph
    G = G.subgraph(np.array([nodes[nodes==auth] for auth in authors_train] + [nodes[nodes==auth] for auth in authors_test]).reshape(len(y_train)+len(y_test)))
    nodes = np.array(list(G.nodes))
    adjacency = nx.convert_matrix.to_scipy_sparse_matrix(G)

    # Load features
    N = np.load('data/vectors_normalized.npy')
    author_ids = N[:,0].astype(int)

    # Collect train and test indices
    idx_train = np.array([np.argwhere(author_ids==auth) for auth in authors_train])
    idx_test = np.array([np.argwhere(author_ids==auth) for auth in authors_test])
    idx_train = idx_train.reshape(len(authors_train))
    idx_test = idx_test.reshape(len(authors_test))
    idx = np.concatenate((idx_train, idx_test))

    # Add hindex to the features
    for i in idx_train:
        N[i,0] = df_train[df_train['author'] == author_ids[i]]['hindex']
    for i in idx_test:
        N[i,0] = -1.0

    features = N[idx]
    features = torch.tensor(features).to(device)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    adj = normalise_adjacency(sp.csr_matrix(adjacency)) 
    adj = sparse_to_torch_sparse(sp.csr_matrix(adjacency)).to(device)

    return adj, features, y_train, y_test, idx_train, idx_test