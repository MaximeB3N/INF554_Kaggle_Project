import torch
import torch.nn as nn
import torch.optim as optim
import time
from gcn import GCN
from dataloader import load_partial_dataset
from utils import get_lr

# Hyperparameters
epochs = 5000
n_hidden_1 = 256
n_hidden_2 = 32
learning_rate = 1e-2
dropout_rate = 0.5
lr_function = lambda epoch: 0.9995 ** epoch

# Options
load = False
save = True
load_path = 'trained_models/model_2.pt'
save_path = 'trained_models/model_vectorized_5000.pt'

#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
adj, features, y_train, y_test, idx_train, idx_test = load_partial_dataset(100000, 10000, device)

# Create the model and specify the optimizer
if load:
    model = GCN(features.shape[1], n_hidden_1, n_hidden_2, dropout_rate).to(device)
    model.load_state_dict(torch.load(load_path))
else:
    model = GCN(features.shape[1], n_hidden_1, n_hidden_2, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.MSELoss()
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_function)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,_ = model(features, adj)
    loss_train = loss(output[idx_train].reshape(len(idx_train)), y_train)
    loss_train.backward()
    optimizer.step()

    
    model.eval()
    output,_ = model(features, adj)

    loss_val = loss(output[idx_test].reshape(len(idx_test)), y_test)
    print('Epoch: {:03d}'.format(epoch+1),
          'learning_rate: {:.4f}'.format(get_lr(optimizer)),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    scheduler.step()


def test():
    model.eval()
    output, embeddings = model(features, adj)
    loss_test = loss(output[idx_test].reshape(len(idx_test)), y_test)
    
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

# Testing
GCN_embeddings = test()

if save:
    torch.save(model.state_dict(), save_path)

print('Done')
