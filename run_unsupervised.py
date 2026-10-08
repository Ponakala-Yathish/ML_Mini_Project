"""Unsupervised baseline: K-means (k=3) and SOM (5x5) on pulsar features; PCA for visualisation only.
Demo mode runs on raw features only (matching paper); full mode also runs on standardized.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from htru import data
from htru.som import SOM


def print_header(title):
    print("=" * 50)
    print(title)
    print("=" * 50)


def print_section(title):
    print(f"\n--- {title} ---")


def run_unsupervised(out_dir, demo=False, synthetic=False, quick=False):
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    F = f"{out_dir}/figures"

    X, y = data.load(use_synthetic=synthetic)
    P = X[y == 1]  # only the real pulsars

    print_header("UNSUPERVISED BASELINE")
    print_section("Dataset (pulsars only)")
    print(f"Pulsar examples: {len(P)}")
    print(f"Features: 8 (raw, as in paper)")

    res = {"n_pulsars": int(len(P)), "synthetic": synthetic, "demo_mode": demo, "runs": {}}

    # PCA on standardized data for visualisation only (clustering uses raw features)
    Z = StandardScaler().fit_transform(P)
    pca = PCA(2).fit(Z)
    emb = pca.transform(Z)
    corr = np.array([[np.corrcoef(Z[:, j], emb[:, c])[0, 1] for c in range(2)] for j in range(8)])

    res["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "feature_pc_correlation": {f: corr[i].tolist() for i, f in enumerate(data.FEATURES)},
    }

    print_section("PCA (for visualisation)")
    print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.4f}")

    # Save PCA correlation figure
    plt.figure(figsize=(6, 3.4))
    w = 0.38
    i = np.arange(8)
    plt.bar(i - w/2, corr[:, 0], w, label="PC1")
    plt.bar(i + w/2, corr[:, 1], w, label="PC2")
    plt.xticks(i, data.FEATURES, rotation=35, ha="right", fontsize=7)
    plt.ylabel("Correlation")
    plt.legend()
    plt.title("Feature / principal-component correlation")
    plt.tight_layout()
    plt.savefig(f"{F}/pca_corr.png", dpi=200)
    plt.close()
    print("  saved pca_corr.png")

    # Clustering: paper uses raw features only
    # Demo mode: only raw; full mode: raw + standardized
    tags = ["raw"] if demo else ["raw", "standardized"]

    fig, axes = plt.subplots(len(tags), 2, figsize=(9, 3.1 * len(tags)))
    if len(tags) == 1:
        axes = axes.reshape(1, 2)

    for r, tag in enumerate(tags):
        D = P if tag == "raw" else Z

        print_section(f"Clustering ({tag} features)")

        # K-means k=3
        km = KMeans(3, n_init=10, random_state=0).fit(D)
        km_sil = float(silhouette_score(D, km.labels_))
        km_sizes = np.bincount(km.labels_).tolist()

        # SOM 5x5
        som = SOM(5, 5, epochs=10 if quick else 100).fit(D)
        sl = som.predict(D)
        cells = np.unique(sl)
        som_lab = np.searchsorted(cells, sl)
        som_sil = float(silhouette_score(D, som_lab)) if len(cells) > 1 else None
        som_sizes = np.bincount(som_lab).tolist()
        n_active = int(len(cells))

        out = {
            "kmeans": {"k": 3, "silhouette": km_sil, "sizes": km_sizes},
            "som": {"grid": "5x5", "n_clusters": n_active, "silhouette": som_sil, "sizes": som_sizes},
        }
        res["runs"][tag] = out

        print(f"  K-means (k=3):")
        print(f"    Silhouette: {km_sil:.4f}")
        print(f"    Cluster sizes: {km_sizes}")
        print(f"  SOM (5x5, epochs={'10' if quick else '100'}):")
        print(f"    Active cells: {n_active}")
        print(f"    Silhouette: {som_sil:.4f}" if som_sil else "    Silhouette: n/a")
        print(f"    Cell sizes: {som_sizes}")

        # Plot
        for c, (name, lab) in enumerate([("K-means", km.labels_), ("SOM", som_lab)]):
            axes[r, c].scatter(emb[:, 0], emb[:, 1], c=lab, s=4, cmap="viridis")
            axes[r, c].set_title(f"{name} ({tag})", fontsize=9)
            axes[r, c].set_xlabel("PC1")
            axes[r, c].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(f"{F}/clusters.png", dpi=200)
    plt.close()
    print(f"\n  saved clusters.png")

    # K-sweep for K-means (full mode only)
    if not demo:
        print_section("K-means k-sweep (raw features)")
        for k in range(2, 7):
            sil = float(silhouette_score(P, KMeans(k, n_init=10, random_state=0).fit_predict(P)))
            print(f"  k={k}: Silhouette = {sil:.4f}")
        res["kmeans_k_sweep_silhouette"] = {
            str(k): float(silhouette_score(P, KMeans(k, n_init=10, random_state=0).fit_predict(P)))
            for k in range(2, 7)
        }

    print_section("Summary")
    for tag in tags:
        r = res["runs"][tag]
        print(f"  {tag}: K-means Sil={r['kmeans']['silhouette']:.4f}, "
              f"SOM Sil={r['som']['silhouette']:.4f} ({r['som']['n_clusters']} active cells)")

    json.dump(res, open(f"{out_dir}/unsupervised.json", "w"), indent=2)
    print(f"\nSaved results to {out_dir}/unsupervised.json")


def main():
    ap = argparse.ArgumentParser(description="Unsupervised pulsar clustering baseline")
    ap.add_argument("--demo", action="store_true",
                    help="Run in demo mode: raw features only (matching paper), skip k-sweep")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    ap.add_argument("--quick", action="store_true", help="Fewer SOM epochs for fast check")
    ap.add_argument("--out", default="results", help="Output directory")
    args = ap.parse_args()

    run_unsupervised(args.out, demo=args.demo, synthetic=args.synthetic, quick=args.quick)


if __name__ == "__main__":
    main()