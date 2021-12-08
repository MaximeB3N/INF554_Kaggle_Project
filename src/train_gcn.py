import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from gcn import GCN
from dataloader import load_partial_dataset, load_train_dataset
from utils import get_lr

# Hyperparameters
epochs = 5000
n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 32
learning_rate = 1e-1
dropout_rate = 0.5
lr_function = lambda epoch: 0.9995 ** epoch
n_train = 100000
n_test = 10000

# Options
load = False
save = True
load_path = 'trained_models/model_2.pt'
save_path = 'trained_models/model_full.pt'
losses_path = 'trained_models/loss_full.csv'
path_features = 'data/authors_vectors.npy'

device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
#adj, features, y_train, y_test = load_partial_dataset(n_train, n_test, path_features, device)
adj, features, y_train, y_test, idx_train, idx_test = load_train_dataset(path_features, device, 0.8, False)
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
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,_ = model(features, adj)
    #loss_train = loss(output[:n_train].reshape(n_train), y_train)
    loss_train = loss(output[idx_train].reshape(n_train), y_train)
    loss_train.backward()
    optimizer.step()
    
    model.eval()
    output,_ = model(features, adj)
    #loss_val = loss(output[:n_test].reshape(n_test), y_test)
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
    model.eval()
    output, embeddings = model(features, adj)
    #loss_test = loss(output[:n_test].reshape(n_test), y_test)
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
