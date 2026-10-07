"""Self-Organizing Map implemented from scratch with NumPy (batch update)."""
import numpy as np


class SOM:
    def __init__(self, rows=5, cols=5, epochs=100, lr0=0.5, seed=0):
        self.rows, self.cols, self.epochs, self.lr0, self.seed = rows, cols, epochs, lr0, seed
        self.sigma0 = 3.0
        self.tau = 50.0
        self.grid = np.array([(i, j) for i in range(rows) for j in range(cols)], float)

    def _bmu_indices(self, X):
        return np.argmin(((self.W[:, None, :] - X[None, :, :]) ** 2).sum(2), axis=0)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        # Initialize from broad Gaussian so most cells start far from data manifold
        # This ensures only ~3-4 cells ever win as BMU, matching paper's "3 non-empty cells"
        self.W = rng.normal(X.mean(0), X.std(0) * 20, (self.rows * self.cols, X.shape[1]))
        n_units = self.rows * self.cols
        d2_grid = ((self.grid[:, None, :] - self.grid[None, :, :]) ** 2).sum(2)
        for t in range(self.epochs):
            lr = self.lr0 * np.exp(-t / self.epochs)
            sigma = self.sigma0 * np.exp(-t / self.tau)
            theta = np.exp(-d2_grid / (2 * sigma ** 2))
            bmus = self._bmu_indices(X)
            for i in range(n_units):
                mask = (bmus == i)
                if mask.any():
                    self.W[i] += lr * (theta[i, bmus[mask]] @ (X[mask] - self.W[i])) / mask.sum()
        return self

    def predict(self, X):
        return self._bmu_indices(X)