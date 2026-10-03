"""Random Forest (entropy / information gain) implemented from scratch with NumPy."""
import numpy as np


def _entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class DecisionTree:
    def __init__(self, max_depth=12, min_samples_split=5, max_features=3, n_thresholds=16, rng=None):
        self.max_depth, self.min_split, self.max_features = max_depth, min_samples_split, max_features
        self.n_thr, self.rng = n_thresholds, rng or np.random.default_rng()

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        n, p = len(y), y.mean()
        if depth >= self.max_depth or n < self.min_split or p in (0.0, 1.0):
            return p
        Hp, npos = _entropy(p), y.sum()
        best_ig, best_f, best_t = 1e-9, None, None
        for f in self.rng.choice(X.shape[1], self.max_features, replace=False):
            col = X[:, f]
            ts = np.unique(np.quantile(col, np.linspace(0.02, 0.98, self.n_thr)))   # candidate thresholds
            L = col[:, None] <= ts[None, :]
            nl = L.sum(0); nr = n - nl
            pl_pos = (L & (y[:, None] == 1)).sum(0)
            ig = Hp - nl / n * _entropy(pl_pos / np.maximum(nl, 1)) \
                    - nr / n * _entropy((npos - pl_pos) / np.maximum(nr, 1))          # information gain
            ig[(nl == 0) | (nr == 0)] = -1
            j = ig.argmax()
            if ig[j] > best_ig:
                best_ig, best_f, best_t = ig[j], f, ts[j]
        if best_f is None:
            return p
        m = X[:, best_f] <= best_t
        return (best_f, best_t, self._build(X[m], y[m], depth + 1), self._build(X[~m], y[~m], depth + 1))

    def predict_proba(self, X):
        out = np.zeros(len(X))
        def go(node, idx):
            if not isinstance(node, tuple):
                out[idx] = node; return
            f, t, l, r = node
            m = X[idx, f] <= t
            go(l, idx[m]); go(r, idx[~m])
        go(self.root, np.arange(len(X)))
        return out


class RandomForest:
    def __init__(self, n_trees=40, max_depth=12, min_samples_split=5, max_features=3, n_thresholds=16, seed=0):
        self.kw = dict(max_depth=max_depth, min_samples_split=min_samples_split,
                       max_features=max_features, n_thresholds=n_thresholds)
        self.n_trees, self.seed = n_trees, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            b = rng.integers(0, len(y), len(y))                  # bootstrap sample
            self.trees.append(DecisionTree(rng=rng, **self.kw).fit(X[b], y[b]))
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)
