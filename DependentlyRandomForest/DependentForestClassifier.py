from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from scipy import stats

class DependentForestClassifier:
    def __init__(self, n_estimators=10, alpha=1.0, max_features=None, **kwargs):
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

            clf = DecisionTreeClassifier(max_features=self.max_features, **self.tree_kwargs)
            clf.fit(x_train, y_train)
            self.forest.append(clf)

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        pred = np.array([tree.predict(X) for tree in self.forest])
        y_hat, _ = stats.mode(pred, axis=0)
        return y_hat
