"""Gaussian Discriminant Analysis (shared covariance) implemented from scratch."""
import numpy as np


class GDA:
    def fit(self, X, y):
        self.phi = y.mean()
        self.mu0, self.mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
        D = X - np.where(y[:, None] == 1, self.mu1, self.mu0)
        self.sigma = D.T @ D / len(X)
        P = np.linalg.pinv(self.sigma)
        self.w = P @ (self.mu1 - self.mu0)
        self.b = -0.5 * (self.mu1 @ P @ self.mu1 - self.mu0 @ P @ self.mu0) + np.log(self.phi / (1 - self.phi))
        return self

    def predict_proba(self, X):
        z = np.clip(X @ self.w + self.b, -500, 500)
        return 1 / (1 + np.exp(-z))          # p(y=1|x) via Bayes rule
