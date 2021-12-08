"""
Script implementing and training a MLP on the final vector matrix obtained previously
"""

import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from gensim.models import Word2Vec

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import sklearn.linear_model as LinearModels

import seaborn as sns
import matplotlib.pyplot as plt

import os
import pandas as pd
from tqdm import *
from scipy import stats

from pathlib import Path


class MLP(nn.Module):
    """Simple MLP model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, dropout):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        z0 = self.relu(self.fc1(x))
        z0 = self.dropout(z0)
        z1 = self.relu(self.fc2(z0))
        out = self.fc3(z1)
        return out

class MLP2(nn.Module):
    """More complex MLP model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2,n_hidden_3, dropout):
        super(MLP2, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4=nn.Linear(n_hidden_3,1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        z0 = self.relu(self.fc1(x))
        z0 = self.dropout(z0)
        z1 = self.relu(self.fc2(z0))
        z1 = self.dropout(z1)
        z2=self.relu(self.fc3(z1))
        out = self.fc4(z2)
        return out

## Defining work sets

G = nx.read_edgelist('data/coauthorship.edgelist', delimiter=' ', nodetype=int)
df_train = pd.read_csv('data/train.csv', dtype={'author': np.int64, 'hindex': np.float32})
df_test = pd.read_csv('data/test.csv', dtype={'author': np.int64})
full_embedding=np.load("Global/full_embedding_matrix.npy")

# Various useful directories
abs_nodeID_Train=dict(df_train["author"])
nodeID_abs_Train=dict([(b,a) for a,b in abs_nodeID_Train.items()])

abs_nodeID_Test=dict(df_test["author"])
nodeID_abs_Test=dict([(b,a) for a,b in abs_nodeID_Test.items()])

abs_hindex_Train=dict(df_train["hindex"])

abs_nodeID_Graph=dict(enumerate(G.nodes))
nodeID_abs_Graph=dict([(b,a) for a,b in enumerate(G.nodes)])

n=G.number_of_nodes()
n_train=abs_nodeID_Train.__len__()
n_test=abs_nodeID_Test.__len__()

## Training and Validation on Train set
# Defining partial set

#Careful, those indexes are related to the TRAIN set, not to the global graph indexing
idx=np.random.permutation(n_train)
idx_train=idx[:int(0.8*n_train)]
idx_val=idx[int(0.8*n_train):]

nodes_train=[abs_nodeID_Train[i] for i in idx_train]
nodes_val=[abs_nodeID_Train[i] for i in idx_val]

X_train_x = torch.tensor([full_embedding[nodeID_abs_Graph[node]][1:] for node in nodes_train], dtype=torch.float32)
X_val_x = torch.tensor([full_embedding[nodeID_abs_Graph[node]][1:] for node in nodes_val], dtype=torch.float32)

hindex_train_x=torch.tensor([abs_hindex_Train[i] for i in idx_train], dtype=torch.float32)
hindex_val_x=torch.tensor([abs_hindex_Train[i] for i in idx_val], dtype=torch.float32)

# Training
n_dim=X_train_x.shape[1]
model=MLP(n_dim,256,64,0.2)
#model=MLP2(n_dim,256,128,64,0.6) # Second MLP, more complex but more prone to overfitting

loss_vals=[]
loss_trains=[]

loss = nn.MSELoss()
lr=2e-2
for i in range(15):
    lr/=2 # The learning rate is reduced iteratively
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_x)
        loss_train = loss(output.reshape(-1), hindex_train_x)
        loss_trains.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

        model.eval()
        output= model(X_val_x)

        loss_val = loss(output.reshape(-1), hindex_val_x)
        loss_vals.append(loss_val.item())
        print('Epoch: {:03d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()))
        if (epoch>100 and loss_val.item()>loss_train.item()*1.1):
            break
plt.plot(loss_vals[20:])
plt.plot(loss_trains[20:])
plt.show()
torch.save(model.state_dict(), "Global/256_64_0.1_lr15.pt")

## Train on full set and generate submission
# Define sets
full_embedding_glob=np.load("Global/full_embedding_matrix.npy")
idx=range(n_train)
nodes_train=[abs_nodeID_Train[i] for i in idx]
nodes_test=[abs_nodeID_Test[i] for i in range(n_test)]
X_train_glob = torch.tensor([full_embedding_glob[nodeID_abs_Graph[node]][1:] for node in nodes_train], dtype=torch.float32)
hindex_train_glob=torch.tensor([abs_hindex_Train[i] for i in idx], dtype=torch.float32)

X_test_glob = torch.tensor([full_embedding_glob[nodeID_abs_Graph[node]][1:] for node in nodeID_abs_Test.keys()], dtype=torch.float32)

n_dim=X_train_x.shape[1]
model_glob=MLP(n_dim,256,64,0.1)

loss_trains_glob=[]
loss = nn.MSELoss()
lr=2e-2
for i in range(15):
    lr/=2
    optimizer = optim.Adam(model_glob.parameters(), lr=lr)
    for epoch in range(200):
        model_glob.train()
        optimizer.zero_grad()
        output = model_glob(X_train_glob)
        loss_train_glob = loss(output.reshape(-1), hindex_train_glob)
        loss_trains_glob.append(loss_train_glob.item())
        loss_train_glob.backward()
        optimizer.step()
        print('Epoch: {:03d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train_glob.item()))

plt.plot(loss_trains_glob[100:])

# Generate submission
_pred=model_glob(X_test_glob)
submission=dict([(nodes_test[i], _pred[i]) for i in range(len(X_test_glob))])
with open("submissions/deepwalk_MLP_full_emb_submission.csv", 'w') as f:
    f.write("author,hindex\n")
    for k,h in submission.items():
        f.write(str(k)+","+str(h.item())+"\n")
    f.close()