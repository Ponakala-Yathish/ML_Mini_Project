"""HTRU2 loading, splitting and class balancing."""
import io, os, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["IP mean", "IP std", "IP kurtosis", "IP skewness",
            "DM-SNR mean", "DM-SNR std", "DM-SNR kurtosis", "DM-SNR skewness"]
URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
DEFAULT_PATH = os.path.join("data", "HTRU_2.csv")


def _download(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading HTRU2 from {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = [n for n in z.namelist() if n.lower().endswith("htru_2.csv")][0]
    with open(dest, "wb") as f:
        f.write(z.read(name))


def synthetic(n=6000, pos_frac=0.09, seed=0):
    """Stand-in data with the HTRU2 schema (ONLY for smoke-testing the code)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_frac).astype(int)
    A = rng.normal(size=(8, 8)) * 0.4 + np.eye(8)
    X = rng.normal(size=(n, 8)) @ A
    X[y == 1] += np.array([-1.5, 0.8, 1.2, 1.5, 1.6, 1.4, -1.2, -1.0])
    return X * np.array([15, 6, 1, 8, 25, 15, 4, 60]) + np.array([110, 47, 0.5, 1, 12, 26, 8, 105]), y


def load(path=DEFAULT_PATH, use_synthetic=False):
    """Returns X (n,8) float array and y (n,) in {0,1}. In the UCI file the label is the LAST column."""
    if use_synthetic:
        return synthetic()
    if not os.path.exists(path):
        try:
            _download(path)
        except Exception as e:
            raise SystemExit(f"Could not download HTRU2 ({e}).\nDownload HTRU2.zip manually from "
                             f"https://archive.ics.uci.edu/dataset/372/htru2 and put HTRU_2.csv at {path}")
    df = pd.read_csv(path, header=None)
    assert df.shape[1] == 9, "expected 8 features + label"
    return df.iloc[:, :8].to_numpy(float), df.iloc[:, 8].to_numpy(int)


def split(X, y, seed=42, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def upsample(X, y, rng):
    """Random minority oversampling to a 50/50 class ratio. Apply to TRAINING data only."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    idx = rng.permutation(np.concatenate([neg, pos, extra]))
    return X[idx], y[idx]
