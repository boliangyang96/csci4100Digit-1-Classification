import numpy as np

class rbf:
    def __init__(self, k):
        self.r = 2 / (k**0.5)
        self.k = k
        self.centers = []
        self.W = []

    def _phi(self, center, x):
        return np.exp(np.linalg.norm(center - x, axis = 1) ** 2 / self.r / (- 2))

    def _initialize_clusters(self, points):
        return points[np.random.randint(points.shape[0]-1, size = self.k)]

    def _get_distances(self, centroid, points):
        return np.linalg.norm(points - centroid, axis=1)

    def _k_means(self, X):
        centroids = self._initialize_clusters(X)
        ncentroids = self._initialize_clusters(X)
        classes = np.zeros(X.shape[0], dtype=np.float64)
        distances = np.zeros([X.shape[0], self.k], dtype=np.float64)
        update = True
        while(update):
            init = True
            for i, c in enumerate(centroids):
                distances[:, i] = self._get_distances(c, X)
            classes = np.argmin(distances, axis=1)
            for c in range(self.k):
                if (len(X[classes == c]) == 0):
                    init = False
                    centroids = self._initialize_clusters(X)
                else:
                    centroids[c] = np.mean(X[classes == c], axis=0)
            if np.array_equal(ncentroids, centroids) and init:
                update = False
            ncentroids = np.copy(centroids)
        return centroids


    def train(self, X, Y):
        self.centers = self._k_means(X)
        x_dim = np.shape(X)[0]
        z = np.zeros([X.shape[0], self.k], dtype=np.float64)
        for i, center in enumerate(self.centers):
            n_centers = np.tile(center, (np.shape(X)[0], 1))
            z[:,i] = self._phi(n_centers, X)
        z = np.insert(z, 0, 1, axis=1)
        #print(np.shape(z))
        self.W = np.linalg.pinv(z).dot(Y)

    def predict(self, X):
        #print(X)
        #print(self.centers)
        z = np.zeros([X.shape[0], self.k], dtype=np.float64)
        for i, center in enumerate(self.centers):
            n_centers = np.tile(center, (np.shape(X)[0], 1))
            z[:,i] = self._phi(n_centers, X)
        z = np.insert(z, 0, 1, axis=1)
        #print(z)
        return np.dot(z, self.W)
