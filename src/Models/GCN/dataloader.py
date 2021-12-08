import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm


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


def correct_order(N, nodes):
    """Reorganize N to follow the same order than nodes"""
    M = np.zeros_like(N)
    count = 0
    for i, node in enumerate(tqdm(nodes)):
        idx = np.argwhere(N[:,0]==node)
        if idx != i:
            count += 1
        M[i,:] = N[idx,:]
    return M, count


def load_partial_dataset(n_train, n_test, path_features, path_edges, path_train_set, device):
    """Load a partial dataset"""

    # Load the graph
    G = nx.read_edgelist(path_edges, delimiter=' ', nodetype=int)
    nodes = np.array(list(G.nodes))

    # Read h indexes
    df_train = pd.read_csv(path_train_set)

    # Select a random permutation
    indices = np.arange(n_train + n_test)
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
    N = np.load(path_features)
    author_ids = N[:,0].astype(int)

    # Collect train and test indices
    idx_train = np.array([np.argwhere(author_ids==auth) for auth in authors_train])
    idx_test = np.array([np.argwhere(author_ids==auth) for auth in authors_test])
    idx_train = idx_train.reshape(n_train)
    idx_test = idx_test.reshape(n_test)
    idx = np.concatenate((idx_train, idx_test))

    # Add hindex to the features
    for i in idx_train:
        N[i,0] = df_train[df_train['author'] == author_ids[i]]['hindex']
    for i in idx_test:
        N[i,0] = -1.0

    features = N[idx]
    features = torch.tensor(features).to(device)

    adj = normalise_adjacency(sp.csr_matrix(adjacency)) 
    adj = sparse_to_torch_sparse(sp.csr_matrix(adj)).to(device)

    return adj, features, y_train, y_test


def compute_and_save_full_adjacency(path_edges, path_adjacency):
    """Compute and save adjacency matrix of the full graph"""
    # Load the graph
    G = nx.read_edgelist(path_edges, delimiter=' ', nodetype=int)
    adjacency = nx.convert_matrix.to_scipy_sparse_matrix(G)
    adj = normalise_adjacency(sp.csr_matrix(adjacency)) 
    adj = sparse_to_torch_sparse(sp.csr_matrix(adj))
    torch.save(adj, path_adjacency)
    return adj


def load_train_dataset(path_features, path_edges, path_train_set, path_adjacency, device, prop_train=0.8, compute_adjacency=False):
    """Load the train dataset"""

    # Load the graph
    G = nx.read_edgelist(path_edges, delimiter=' ', nodetype=int)
    nodes = np.array(list(G.nodes))

    # Read h indexes
    df_train = pd.read_csv(path_train_set)
    n = df_train.shape[0]
    n_train = int(np.floor(prop_train * n))
    n_test = n - n_train

    # Select a random permutation
    indices = np.arange(n)
    indices = np.random.permutation(indices)

    # Split into train and test sets
    y_train = df_train['hindex'][indices[:n_train]].to_numpy()
    authors_train = df_train['author'][indices[:n_train]]
    y_test = df_train['hindex'][indices[-n_test:]].to_numpy()
    authors_test = df_train['author'][indices[-n_test:]]
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Load features
    N = np.load(path_features)
    N, changes = correct_order(N, nodes)
    print(str(changes) + ' changes!')
    np.save('data/features.npy', N)
    #N = np.load('data/features.npy')
    author_ids = N[:,0].astype(int)

    # Collect train and test indices
    idx_train = np.array([np.argwhere(author_ids==auth) for auth in authors_train])
    idx_test = np.array([np.argwhere(author_ids==auth) for auth in authors_test])
    idx_train = idx_train.reshape(n_train)
    idx_test = idx_test.reshape(n_test)

    # Add hindex to the features
    for i in tqdm(idx_train):
        N[i,0] = df_train[df_train['author'] == author_ids[i]]['hindex']
    for i in tqdm(idx_test):
        N[i,0] = -1.0

    features = N
    s = features[:,1:].sum(axis=1)
    features = torch.tensor(features).to(device)
    print(np.sum(s != 0))

    # Load adjacency sparse matrix
    if compute_adjacency:
        adj = compute_and_save_full_adjacency(path_edges, path_adjacency).to(device)
    else:
        adj = torch.load(path_adjacency).to(device)

    return adj, features, y_train, y_test, idx_train, idx_test
