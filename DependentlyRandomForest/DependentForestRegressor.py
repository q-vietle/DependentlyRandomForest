from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from scipy import stats

class DependentForestRegressor:
    def __init__(self, n_estimators=10, alpha=1.0, max_features=None, **kwargs):
        #default n_estimator = 10, alpha = 1.0, max_feature = None
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.max_features = max_features
        self.tree_kwargs = kwargs
        self.forest = []

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame) else y

        alpha = [self.alpha] * len(X)

        for i in range(self.n_estimators):
            weights = np.random.dirichlet(alpha)
            weights /= np.sum(weights)

            train_id = np.random.choice(X.index, size=len(X), p=weights)
            x_train = X.loc[train_id]
            y_train = y.loc[train_id]

            rgs = DecisionTreeRegressor(max_features=self.max_features, **self.tree_kwargs)
            rgs.fit(x_train, y_train)
            self.forest.append(rgs)

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        pred = np.array([tree.predict(X) for tree in self.forest])
        y_hat = np.mean(pred)
        return y_hat
