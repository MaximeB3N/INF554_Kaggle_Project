import torch
from torch import nn
import numpy as np 

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,in_shape, hidden_params, out_shape=1):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_params = hidden_params
        
        self.flatten = nn.Sequential(nn.Flatten())

        self.lin1 = nn.Sequential(nn.Linear(in_shape, hidden_params[0]),
                                            nn.BatchNorm1d(hidden_params[0]),
                                            nn.ReLU())

        self.lin2 = nn.Sequential(nn.Linear(hidden_params[0],hidden_params[1]),
                                            nn.BatchNorm1d(hidden_params[1]),
                                            nn.ReLU())
        
        if len(self.hidden_params)==3:
            self.lin3 = nn.Sequential(nn.Linear(hidden_params[1],hidden_params[2]), 
                                            nn.BatchNorm1d(hidden_params[2]),
                                            nn.ReLU())

            self.linOut = nn.Sequential(nn.Linear(hidden_params[2],out_shape))
        
        
        else:
            self.linOut = nn.Sequential(nn.Linear(hidden_params[1],out_shape))
        
        self.pytorch_total_params = self.count_params()
        print(f"Le modèle contient {self.pytorch_total_params} paramètres")

    def count_params(self):
    
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
    def forward(self, x):
        '''Forward pass'''
        out = self.flatten(x)
        out = self.lin1(out)
        out = self.lin2(out)

        if len(self.hidden_params)==3:
            out = self.lin3(out)
            
        out = self.linOut(out)
        return out
