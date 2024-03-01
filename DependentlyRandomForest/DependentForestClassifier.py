from sklearn import datasets #for data sample
from sklearn.tree import DecisionTreeClassifier #for build DTs
from sklearn.model_selection import train_test_split #for split data to train
from sklearn.metrics import accuracy_score #for calculating the error score

#the below libraries are for calculation
import pandas as pd
import numpy as np
import random
from scipy import stats

#the below libraries are for visualization
import seaborn as sns
import matplotlib.pyplot as plt

class DependentForestClassifier:
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from scipy import stats
    def __init__(self, n_estimators=10, alpha=1.0, max_features = None):
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.max_features = max_features
        self.forest = []

    def fit(self, X, y):
        self.xtrain = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.ytrain = pd.Series(y) if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame) else y

        alpha = [self.alpha] * len(self.xtrain)

        for i in range(self.n_estimators):
            weights = np.random.dirichlet(alpha)
            weights /= np.sum(weights)  # Normalize weights

            train_id = np.random.choice(self.xtrain.index, size=len(self.xtrain), p=weights)
            x_train = self.xtrain.loc[train_id]
            y_train = self.ytrain.loc[train_id]

            clf = DecisionTreeClassifier(max_features = self.max_features)
            clf.fit(x_train, y_train)
            self.forest.append(clf)

    def predict(self, X):
        
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        pred = np.array([tree.predict(X) for tree in self.forest])
        y_hat, _ = stats.mode(pred, axis=0)
        return y_hat

# Example usage:
# clf = DependentForestClassifier(n_estimators=10, alpha=1.0)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
