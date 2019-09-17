import numpy as np
from collections import Counter

class knn():
    def __init__(self, k):
        self.k = k
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        num_train = self.Xtr.shape[0]
        for i in range(num_test):
            cloned_X = np.array([X[i]]*num_train)
            distances = np.linalg.norm(self.Xtr - cloned_X, axis = 1)
            closest_y = self.ytr[distances.argsort()[:self.k]]
            Ypred[i] = Counter(closest_y).most_common(1)[0][0]
        return Ypred
