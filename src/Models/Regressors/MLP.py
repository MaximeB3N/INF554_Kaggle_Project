import torch
from torch import nn
import numpy as np 
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,in_shape, hidden_params, out_shape=1, 
                    epochs=30, lr=0.001, batch_size=32, verbose=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_params = hidden_params
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.flatten = nn.Sequential(nn.Flatten())

        self.lin1 = nn.Sequential(nn.Linear(in_shape, hidden_params[0]),
                                            #nn.Dropout(p=0.2),
                                            nn.BatchNorm1d(hidden_params[0]),
                                            nn.ReLU())

        self.lin2 = nn.Sequential(nn.Linear(hidden_params[0],hidden_params[1]),
                                            #nn.Dropout(p=0.2),
                                            nn.BatchNorm1d(hidden_params[1]),
                                            nn.ReLU())
        
        if len(self.hidden_params)==3:
            self.lin3 = nn.Sequential(nn.Linear(hidden_params[1],hidden_params[2]),
                                            #nn.Dropout(p=0.2),
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
        self.eval()
        x = torch.from_numpy(x).type(torch.float32).to(self.device)
        out = self.forward(x)
        return out.detach().cpu().numpy()

    def fit(self, X, y):
        '''Fit the model'''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.train()
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.90)

        dataset = TensorDataset(torch.from_numpy(X).type(torch.float32).to(self.device),
                                torch.unsqueeze(torch.from_numpy(y),-1).type(torch.float32).to(self.device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size)


        for epoch in range(self.epochs):
            
            mean_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                x, y = batch[0], batch[1]
                #print(x.dtype, y.dtype)
                #print(x.shape)
                y_pred = self.forward(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()
                mean_loss += loss.detach().cpu().numpy()            

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {mean_loss/len(dataloader):.4f}")


    def run(self, X_train, y_train, X_test, y_test):
        '''Fit the model'''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.train()
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.90)

        dataset_train = TensorDataset(torch.from_numpy(X_train).type(torch.float32).to(self.device),
                                torch.unsqueeze(torch.from_numpy(y_train),-1).type(torch.float32).to(self.device))
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size)


        dataset_val = TensorDataset(torch.from_numpy(X_test).type(torch.float32).to(self.device),
                                torch.unsqueeze(torch.from_numpy(y_test),-1).type(torch.float32).to(self.device))
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size)


        for epoch in range(self.epochs):
            
            mean_loss = 0
            for batch in dataloader_train:
                self.optimizer.zero_grad()
                x, y = batch[0], batch[1]
                #print(x.dtype, y.dtype)
                #print(x.shape)
                y_pred = self.forward(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()
                mean_loss += loss.detach().cpu().numpy()
                #break
            #break
            with torch.no_grad():
                mean_loss_val = 0
                for batch in dataloader_val:
                    x, y = batch[0], batch[1]
                    #print(x.dtype, y.dtype)
                    #print(x.shape)
                    y_pred = self.forward(x)
                    loss = self.loss_fn(y_pred, y)
                    #self.scheduler.step()
                    mean_loss_val += loss.detach().cpu().numpy()
            

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {mean_loss/len(dataloader_train):.4f} - Loss_val: {mean_loss_val/len(dataloader_val):.4f}")
