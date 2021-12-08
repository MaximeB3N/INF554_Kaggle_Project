import numpy as np
import torch  
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR

from Models.MLP import MLP


class Regressors:
    """ Regressor methods comparison"""

    def __init__(self, in_shape, hidden_params, epochs=30, lr=0.01, batch_size=32, verbose=True):
        """ Initialize the Regressors class."""
        self.in_shape = in_shape
        self.hidden_params = hidden_params
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {
            'MLP': MLP(in_shape, hidden_params, epochs=epochs, lr=lr, batch_size=batch_size, verbose=verbose).to(device)
            # 'LinearRegression': LinearRegression(),
            # 'RandomForestRegressor': RandomForestRegressor(),
            # 'GradientBoostingRegressor': GradientBoostingRegressor(),
            # 'AdaBoostRegressor': AdaBoostRegressor(),
            # 'BaggingRegressor': BaggingRegressor(),
            # 'SVR': SVR()
        }

    def fit(self, X, y):
        """ Fit the models."""
        for model in self.models:
            print("-"*20)
            print(f"Training {model}...")
            self.models[model].fit(X, y)

    def predict(self, X):
        """ Predict the output of the models."""
        predictions = {}
        for model in self.models:
            print("-"*20)
            print(f"Predicting {model}...")
            predictions[model] = self.models[model].predict(X)
        return predictions

    def MSE(self, y_true, y_pred):
        """ Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    def score(self, X, y):
        """ Score the models."""
        predictions = self.predict(X)
        scores = {}
        for model in self.models:
            print("-"*20)
            print(f"Scoring {model}...")
            scores[model] = self.MSE(predictions[model], y)

            if self.verbose:
                print(f'{model}: {scores[model]}')
        
        return scores
