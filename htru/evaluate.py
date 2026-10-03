import numpy as np
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from .data import upsample


class SkRF:
    """Thin adapter so scikit-learn RF matches the fit / predict_proba(1-D) interface."""
    def __init__(self, **kw):
        from sklearn.ensemble import RandomForestClassifier
        self.m = RandomForestClassifier(random_state=0, n_jobs=-1, **kw)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict_proba(self, X): return self.m.predict_proba(X)[:, 1]


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(threshold=float(thr), accuracy=accuracy_score(y, yh), recall=recall_score(y, yh),
                precision=precision_score(y, yh, zero_division=0), f1=f1_score(y, yh),
                roc_auc=roc_auc_score(y, p), avg_precision=average_precision_score(y, p))


def cv_oof(factory, X, y, seed=0, k=5):
    """k-fold CV; upsampling is done INSIDE each fold on the training part only.
    Returns out-of-fold probabilities, per-fold metrics and mean (val - train) Brier gap."""
    rng = np.random.default_rng(seed)
    oof, folds, gaps = np.zeros(len(y)), [], []
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        Xu, yu = upsample(X[tr], y[tr], rng)
        m = factory().fit(Xu, yu)
        p = m.predict_proba(X[va]); oof[va] = p
        folds.append(metrics(y[va], p))
        gaps.append(brier_score_loss(y[va], p) - brier_score_loss(y[tr], m.predict_proba(X[tr])))
    return oof, folds, float(np.mean(gaps))


def mean_folds(folds):
    return {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}


def best_threshold(y, p):
    """Threshold maximising TPR - FPR (Youden J), chosen on out-of-fold predictions, NOT the test set."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])
