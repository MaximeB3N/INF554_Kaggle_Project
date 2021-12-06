import numpy as np  
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR

from MLP import MLP


class Regressors:
    """ Regressor methods comparison"""

    def __init__(self, hidden_params, epochs=30, lr=0.001, verbose=True):
        """ Initialize the Regressors class."""
        self.hidden_params = hidden_params
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.models = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'AdaBoostRegressor': AdaBoostRegressor(),
            'BaggingRegressor': BaggingRegressor(),
            'SVR': SVR(),
            'MLP': MLP(hidden_params, epochs=epochs, lr=lr, verbose=verbose)
        }

    def fit(self, X, y):
        """ Fit the models."""
        for model in self.models:
            self.models[model].fit(X, y)

    def predict(self, X):
        """ Predict the output of the models."""
        predictions = {}
        for model in self.models:
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
            scores[model] = self.MSE(predictions[model], y)

            if self.verbose:
                print(f'{model}: {scores[model]}')
        
        return scores
