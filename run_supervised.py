"""Supervised baseline: GDA vs Random Forest (scikit-learn) vs Random Forest (from scratch).
Demo mode uses fixed hyperparameters matching the paper; full mode does randomized search.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from htru import data
from htru.gda import GDA
from htru.forest import RandomForest
from htru.evaluate import SkRF, cv_oof, mean_folds, metrics, best_threshold

# Fixed hyperparameters from paper (n_estimators=200, min_samples_split=5, max_features=4, max_depth=110)
PAPER_RF_PARAMS = {
    "n_estimators": 200,
    "min_samples_split": 5,
    "max_features": 4,
    "max_depth": 110,
}


def print_header(title):
    print("=" * 50)
    print(title)
    print("=" * 50)


def print_section(title):
    print(f"\n--- {title} ---")


def run_supervised(out_dir, demo=False, synthetic=False, quick=False):
    os.makedirs(f"{out_dir}/figures", exist_ok=True)

    X, y = data.load(use_synthetic=synthetic)
    Xtr, Xte, ytr, yte = data.split(X, y)

    print_header("SUPERVISED BASELINE")
    print_section("Dataset")
    print(f"Samples: {len(y)}")
    print(f"Features: 8")
    print(f"Pulsars: {int(y.sum())}")
    print(f"Non-pulsars: {int((y == 0).sum())}")
    print(f"Train: {len(ytr)}")
    print(f"Test: {len(yte)}")

    rng = np.random.default_rng(0)
    res = {
        "n_samples": int(len(y)),
        "n_pulsars": int(y.sum()),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "synthetic": synthetic,
        "demo_mode": demo,
        "models": {},
    }

    if demo:
        best = PAPER_RF_PARAMS.copy()
        print_section("Random Forest (scikit-learn) — using paper hyperparameters")
        print(f"  n_estimators={best['n_estimators']}, min_samples_split={best['min_samples_split']}, "
              f"max_features={best['max_features']}, max_depth={best['max_depth']}")
    else:
        print_section("Random Forest (scikit-learn) — hyperparameter search")
        n_iter = 3 if quick else 12
        grid = dict(
            n_estimators=[20] if quick else [100, 200, 300],
            min_samples_split=[2, 5, 10],
            max_features=[2, 3, 4, "sqrt"],
            max_depth=[10, 30, 60, 110, None],
        )
        best, best_auc = None, -1
        for prm in ParameterSampler(grid, n_iter, random_state=0):
            _, folds, _ = cv_oof(lambda: SkRF(**prm), Xtr, ytr)
            auc = mean_folds(folds)["roc_auc"]
            print(f"  search {prm} AUC={auc:.4f}")
            if auc > best_auc:
                best, best_auc = prm, auc
        print(f"  best params: {best}")

    res["rf_best_params"] = {k: (v if v is None or isinstance(v, str) else int(v)) for k, v in best.items()}

    factories = {
        "GDA (baseline)": lambda: GDA(),
        "Random Forest (scikit-learn)": lambda: SkRF(**best),
        "Random Forest (from scratch)": lambda: RandomForest(
            n_trees=8 if quick else 40,
            max_depth=12,
            min_samples_split=5,
            max_features=3,
        ),
    }

    curves = {}
    for name, fac in factories.items():
        print_section(name)
        oof, folds, gap = cv_oof(fac, Xtr, ytr)
        thr = best_threshold(ytr, oof)
        cv = mean_folds(folds)
        cv_thr = metrics(ytr, oof, thr)
        Xu, yu = data.upsample(Xtr, ytr, rng)
        p = fac().fit(Xu, yu).predict_proba(Xte)

        res["models"][name] = {
            "cv_mean_at_0.5": cv,
            "cv_oof_at_tuned_threshold": cv_thr,
            "cv_val_minus_train_brier": gap,
            "tuned_threshold": thr,
            "test_at_0.5": metrics(yte, p, 0.5),
            "test_at_tuned_threshold": metrics(yte, p, thr),
        }
        curves[name] = p

        tm = res["models"][name]
        print(f"  CV recall @ 0.5: {cv['recall']:.4f}")
        print(f"  Test accuracy @ 0.5: {tm['test_at_0.5']['accuracy']:.4f}")
        print(f"  Test recall @ 0.5: {tm['test_at_0.5']['recall']:.4f}")
        print(f"  Tuned threshold: {thr:.4f}")
        print(f"  Test recall @ tuned: {tm['test_at_tuned_threshold']['recall']:.4f}")
        print(f"  Test precision @ tuned: {tm['test_at_tuned_threshold']['precision']:.4f}")
        print(f"  Test F1 @ tuned: {tm['test_at_tuned_threshold']['f1']:.4f}")
        print(f"  Test ROC-AUC: {tm['test_at_tuned_threshold']['roc_auc']:.4f}")

    print_section("Summary Table")
    print(f"{'Model':<30} {'Acc@0.5':>8} {'Rec@0.5':>8} {'Thresh':>7} {'Rec@thr':>8} {'Prec@thr':>8} {'AUC':>6}")
    print("-" * 85)
    for name, m in res["models"].items():
        t05 = m["test_at_0.5"]
        tthr = m["test_at_tuned_threshold"]
        print(f"{name:<30} {t05['accuracy']:.4f}  {t05['recall']:.4f}  {m['tuned_threshold']:.3f}  "
              f"{tthr['recall']:.4f}  {tthr['precision']:.4f}  {tthr['roc_auc']:.4f}")

    # ---- figures ----
    F = f"{out_dir}/figures"
    print_section("Generating figures")
    plt.figure(figsize=(5, 3.4))
    for n, p in curves.items():
        f, t, _ = roc_curve(yte, p)
        plt.plot(f, t, label=n)
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("Recall (TPR)")
    plt.title("ROC curves (test set)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{F}/roc.png", dpi=200)
    plt.close()
    print("  saved roc.png")

    plt.figure(figsize=(5, 3.4))
    for n, p in curves.items():
        pr, rc, _ = precision_recall_curve(yte, p)
        plt.plot(rc, pr, label=n)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (test set)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{F}/pr.png", dpi=200)
    plt.close()
    print("  saved pr.png")

    fig, ax = plt.subplots(1, len(curves), figsize=(10, 3), sharey=True)
    ths = np.linspace(0.01, 0.99, 99)
    for a_, (n, p) in zip(ax, curves.items()):
        tpr = [((p >= t) & (yte == 1)).sum() / (yte == 1).sum() for t in ths]
        fpr = [((p >= t) & (yte == 0)).sum() / (yte == 0).sum() for t in ths]
        a_.plot(ths, tpr, label="TPR")
        a_.plot(ths, fpr, label="FPR")
        a_.plot(ths, np.array(tpr) - np.array(fpr), label="TPR-FPR")
        a_.set_title(n, fontsize=8)
        a_.set_xlabel("Threshold")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{F}/threshold.png", dpi=200)
    plt.close()
    print("  saved threshold.png")

    plt.figure(figsize=(4, 3.4))
    fr, mp = calibration_curve(yte, curves["Random Forest (scikit-learn)"], n_bins=10)
    plt.plot(mp, fr, "o-")
    plt.plot([0, 1], [0, 1], "k:")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of pulsars")
    plt.title("RF calibration")
    plt.tight_layout()
    plt.savefig(f"{F}/calibration.png", dpi=200)
    plt.close()
    print("  saved calibration.png")

    json.dump(res, open(f"{out_dir}/supervised.json", "w"), indent=2)
    print(f"\nSaved results to {out_dir}/supervised.json")


def main():
    ap = argparse.ArgumentParser(description="Supervised pulsar classification baseline")
    ap.add_argument("--demo", action="store_true",
                    help="Run in demo mode: skip hyperparameter search, use paper params")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    ap.add_argument("--quick", action="store_true", help="Tiny search / few trees for fast check")
    ap.add_argument("--out", default="results", help="Output directory")
    args = ap.parse_args()

    run_supervised(args.out, demo=args.demo, synthetic=args.synthetic, quick=args.quick)


if __name__ == "__main__":
    main()