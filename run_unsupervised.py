"""Unsupervised part: cluster the true pulsars with K-means (baseline) and a from-scratch SOM; PCA for visualisation only."""
import argparse, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from htru import data
from htru.som import SOM

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(f"{a.out}/figures", exist_ok=True)
F = f"{a.out}/figures"

X, y = data.load(use_synthetic=a.synthetic)
P = X[y == 1]                                       # only the real pulsars
print("pulsar examples:", len(P))
Z = StandardScaler().fit_transform(P)
res = {"n_pulsars": int(len(P)), "synthetic": a.synthetic, "runs": {}}

# ---- PCA (visualisation & interpretation only; clustering uses all 8 features) ----
pca = PCA(2).fit(Z); emb = pca.transform(Z)
corr = np.array([[np.corrcoef(Z[:, j], emb[:, c])[0, 1] for c in range(2)] for j in range(8)])
res["pca"] = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
              "feature_pc_correlation": {f: corr[i].tolist() for i, f in enumerate(data.FEATURES)}}
plt.figure(figsize=(6, 3.4)); w = .38; i = np.arange(8)
plt.bar(i - w/2, corr[:, 0], w, label="PC1"); plt.bar(i + w/2, corr[:, 1], w, label="PC2")
plt.xticks(i, data.FEATURES, rotation=35, ha="right", fontsize=7); plt.ylabel("Correlation"); plt.legend()
plt.title("Feature / principal-component correlation"); plt.tight_layout(); plt.savefig(f"{F}/pca_corr.png", dpi=200); plt.close()

# ---- Clustering on raw features (as in the paper) and on standardized features ----
fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
for r, (tag, D) in enumerate([("raw", P), ("standardized", Z)]):
    km = KMeans(3, n_init=10, random_state=0).fit(D)
    som = SOM(5, 5, epochs=10 if a.quick else 100).fit(D)
    sl = som.predict(D); cells = np.unique(sl)
    som_lab = np.searchsorted(cells, sl)
    out = {"kmeans": {"k": 3, "silhouette": float(silhouette_score(D, km.labels_)),
                      "sizes": np.bincount(km.labels_).tolist()},
           "som": {"grid": "5x5", "n_clusters": int(len(cells)),
                   "silhouette": float(silhouette_score(D, som_lab)) if len(cells) > 1 else None,
                   "sizes": np.bincount(som_lab).tolist()}}
    out["kmeans_k_sweep_silhouette"] = {str(k): float(silhouette_score(D, KMeans(k, n_init=10, random_state=0).fit_predict(D)))
                                        for k in range(2, 7)}
    res["runs"][tag] = out
    print(tag, json.dumps(out))
    for c, (name, lab) in enumerate([("K-means", km.labels_), ("SOM", som_lab)]):
        axes[r, c].scatter(emb[:, 0], emb[:, 1], c=lab, s=4, cmap="viridis")
        axes[r, c].set_title(f"{name} ({tag} features)", fontsize=9)
        axes[r, c].set_xlabel("PC1"); axes[r, c].set_ylabel("PC2")
plt.tight_layout(); plt.savefig(f"{F}/clusters.png", dpi=200); plt.close()
json.dump(res, open(f"{a.out}/unsupervised.json", "w"), indent=2)
print("saved", f"{a.out}/unsupervised.json")
