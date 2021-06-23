import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils


class Models:

    def __init__(self):
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }

        self.params = {
            'SVR': {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['auto', 'scale'],
                'C': [1, 5, 10]
            },
            'GRADIENT': {
                'loss': ['ls', 'lad'],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        }

    def grid_training(self, features, target):
        best_score = 999
        best_model = None

        for name, reg in self.reg.items():
            grid_regressor = GridSearchCV(reg, self.params[name], cv=3).fit(
                features, target.values.ravel())

            score = np.abs(grid_regressor.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_regressor.best_estimator_

        utils = Utils()
        utils.model_export(best_model, best_score)
