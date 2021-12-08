import torch
from sklearn.model_selection import train_test_split

from src.Dataloader.dataloader import Dataset
from src.Models.Regressors import Regressors
from src.Models.Regressors.MLP import MLP


pathVectors = "data/full_embedding_matrix.npy"
pathLabels = "data/train.csv"

dataset = Dataset(pathVectors, pathLabels) 

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

regressors = Regressors(IN_SHAPE, HIDDEN_PARAMS,epochs=EPOCHS,lr=LR,
                        batch_size=BATCH_SIZE,verbose=VERBOSE)

regressors.fit(X_train,y_train)
scores = regressors.score(X_test, y_test)
