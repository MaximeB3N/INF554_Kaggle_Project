import numpy as np
import torch
import torch.nn as nn


class GCN(nn.Module):
    """Simple GCN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, dropout):
        super(GCN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        
        #Please insert your code for Task 5 here
        z0 = self.relu(self.fc1(adj.matmul(torch.tensor(x_in, dtype=torch.float32))))
        z0 = self.dropout(z0)
        z1 = self.relu(self.fc2(adj @ z0))
        x = self.fc3(z1)

        return x, z1

