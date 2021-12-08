from sklearn.model_selection import train_test_split
import torch

from Dataloader.dataloader import Dataset
from Models.Regressors import Regressors
from Models.MLP import MLP

# pathNLP = "data/authors_vectors.npy"
# pathGraph = "data/vectors_cbow_own_normalized.npy"
pathVectors = "data/full_embedding_matrix.npy"
pathLabels = "data/train.csv"

dataset = Dataset(pathVectors, pathLabels) #(pathNLP, pathGraph, pathLabels)

X = dataset.inputs
y = dataset.hindex

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=42)

IN_SHAPE = len(X[0])
HIDDEN_PARAMS = [256,64]
EPOCHS= 200
LR=0.01
BATCH_SIZE = 32
VERBOSE=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()

# regressors = Regressors(IN_SHAPE, HIDDEN_PARAMS,epochs=EPOCHS,lr=LR,
#                         batch_size=BATCH_SIZE,verbose=VERBOSE)

# regressors.fit(X_train,y_train)
# scores = regressors.score(X_test, y_test)

model = MLP(IN_SHAPE, HIDDEN_PARAMS,epochs=EPOCHS,lr=LR,
                         batch_size=BATCH_SIZE,verbose=VERBOSE).to(device)
model.run(X_train, y_train, X_test, y_test)

y_pred = torch.from_numpy(model.predict(X_test)).to(device)
y_test = torch.unsqueeze(torch.tensor(y_test).to(device),1)

torch.save(model,f"data/Models/modelMLP_{EPOCHS}_{LR}_{BATCH_SIZE}.pt")
score = loss_fn(y_pred, y_test)
print(f"Score : {score}")