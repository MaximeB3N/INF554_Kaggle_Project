import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import yaml
import argparse

from src.Models.GCN.gcn import GCN
from src.Models.GCN.dataloader import load_train_dataset
from src.Models.GCN.utils import get_lr


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path to yaml config')
args = parser.parse_args()

with open(args.config, 'r') as stream:
    config = yaml.safe_load(stream)


# Hyperparameters
epochs = config['epochs']
n_hidden_1 = config['n_hidden_1']
n_hidden_2 = config['n_hidden_2']
n_hidden_3 = config['n_hidden_3']
learning_rate = config['learning_rate']
dropout_rate = config['dropout_rate']

# Learning rate scheduler function
lr_function = lambda epoch: 0.9995 ** epoch

# Options
load = config['load']
save = config['save']
load_path = config['load_path']
save_path = config['save_path']
losses_path = config['losses_path']
path_features = config['path_features']
path_edges = config['path_edges']
path_train_set = config['path_train_set']
path_test_set = config['path_test_set']
path_adjacency = config['path_adjacency']

device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the dataset
adj, features, y_train, y_test, idx_train, idx_test = load_train_dataset(path_features, path_edges, path_train_set, path_adjacency, device, 0.8, False)
n_train = len(idx_train)
n_test = len(idx_test)

# Create the model and specify the optimizer
model = GCN(features.shape[1], n_hidden_1, n_hidden_2, n_hidden_3, dropout_rate).to(device)
if load:
    model.load_state_dict(torch.load(load_path))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.MSELoss()
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_function)

losses = np.zeros((epochs, 2))




def train(epoch):
    """Perform one epoch of the training pipeline"""
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,_ = model(features, adj)
    loss_train = loss(output[idx_train].reshape(n_train), y_train)
    loss_train.backward()
    optimizer.step()
    
    model.eval()
    output,_ = model(features, adj)
    loss_val = loss(output[idx_test].reshape(n_test), y_test)
    print('Epoch: {:03d}'.format(epoch+1),
          'learning_rate: {:.4f}'.format(get_lr(optimizer)),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    losses[epoch, 0] = loss_train.item()
    losses[epoch, 1] = loss_val.item()
    
    scheduler.step()


def test():
    """Test the model"""
    model.eval()
    output, embeddings = model(features, adj)
    loss_test = loss(output[idx_test].reshape(n_test), y_test)
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))

    return embeddings



# Train model
print('Begin training...')
t_total = time.time()
for epoch in range(epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print()

# Save model
if save:
    torch.save(model.state_dict(), save_path)
    df_loss = pd.DataFrame(losses, columns=['loss_train', 'loss_val'])
    df_loss.to_csv(losses_path)

# Test model
GCN_embeddings = test()

print('Done')
