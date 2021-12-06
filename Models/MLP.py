import torch
from torch import nn
import numpy as np 
from tqdm import tqdm


class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,in_shape, hidden_params, out_shape=1, epochs=30, lr=0.001, verbose=True):
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

    def predict(self, x):
        '''Predict'''
        out = self.forward(x)
        return out.detach().numpy()

    def fit(self, X, y):
        '''Fit the model'''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.train()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y), batch_size=self.batch_size)

        for epoch, x, y in enumerate(tqdm(dataset)):
            self.optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f}")
