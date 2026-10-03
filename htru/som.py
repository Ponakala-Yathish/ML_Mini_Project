"""Self-Organizing Map implemented from scratch with NumPy."""
import numpy as np


class SOM:
    def __init__(self, rows=5, cols=5, epochs=100, lr0=0.5, seed=0):
        self.rows, self.cols, self.epochs, self.lr0, self.seed = rows, cols, epochs, lr0, seed
        self.sigma0 = max(rows, cols) / 2
        self.grid = np.array([(i, j) for i in range(rows) for j in range(cols)], float)

    def bmu(self, x):
        return np.argmin(((self.W - x) ** 2).sum(1))

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.W = X[rng.choice(len(X), self.rows * self.cols, replace=False)].copy()
        for t in range(self.epochs):
            lr = self.lr0 * np.exp(-t / self.epochs)             # exponential decay per epoch
            sigma = self.sigma0 * np.exp(-t / self.epochs)
            for x in X[rng.permutation(len(X))]:
                b = self.bmu(x)
                d2 = ((self.grid - self.grid[b]) ** 2).sum(1)
                theta = np.exp(-d2 / (2 * sigma ** 2))           # neighbourhood function
                self.W += lr * theta[:, None] * (x - self.W)
        return self

    def predict(self, X):
        return np.array([self.bmu(x) for x in X])
