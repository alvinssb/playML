import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.mean(X[:, 1]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.std_ is not None, \
            "must fit before transform!"
        assert X.shape[1] == len(self.mean_), \
            "The feature number of X must be equal to mean_ and std_"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in (X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX
