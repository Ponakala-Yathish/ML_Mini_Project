"""Supervised baseline: GDA vs Random Forest (scikit-learn) vs Random Forest (from scratch).
Demo mode uses fixed hyperparameters matching the paper; full mode does randomized search.
Interactive mode allows manual feature input for live prediction.
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

# Feature metadata for interactive mode
FEATURES = [
    ("IP mean", "Integrated profile mean", 0, 200),
    ("IP std", "Integrated profile std", 20, 100),
    ("IP kurtosis", "Integrated profile excess kurtosis", -2, 9),
    ("IP skewness", "Integrated profile skewness", -2, 70),
    ("DM-SNR mean", "DM-SNR curve mean", 0, 230),
    ("DM-SNR std", "DM-SNR curve std", 7, 115),
    ("DM-SNR kurtosis", "DM-SNR curve excess kurtosis", -3, 35),
    ("DM-SNR skewness", "DM-SNR curve skewness", -2, 1200),
]

FEATURE_NAMES = [f[0] for f in FEATURES]


def print_header(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_section(title):
    print(f"\n--- {title} ---")


def get_feature_input():
    """Prompt user for 8 feature values with validation."""
    print("\nEnter the 8 feature values for a pulsar candidate:")
    print("(Press Enter to use typical mean values, or type 'q' to quit)\n")
    values = []
    for i, (name, desc, vmin, vmax) in enumerate(FEATURES):
        while True:
            prompt = f"  {i+1}. {name} ({desc})\n      Range: [{vmin:.1f}, {vmax:.1f}] → "
            try:
                s = input(prompt).strip()
                if s.lower() == 'q':
                    return None
                if s == '':
                    # Use approximate mean as default
                    defaults = [111, 47, 0.5, 1.8, 13, 26, 8, 105]
                    val = defaults[i]
                    print(f"      Using default: {val}")
                else:
                    val = float(s)
                    if val < vmin or val > vmax:
                        print(f"      WARNING: Value outside typical range [{vmin}, {vmax}]")
                values.append(val)
                break
            except ValueError:
                print("      Invalid number. Try again.")
    return np.array(values).reshape(1, -1)


def train_models(Xtr, ytr, demo=False, quick=False):
    """Train all three models and return fitted models with thresholds."""
    rng = np.random.default_rng(0)

    if demo:
        best = PAPER_RF_PARAMS.copy()
        print("  Using paper hyperparameters for RF (sklearn)")
    else:
        print("  Hyperparameter search for RF (sklearn)...")
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
            if auc > best_auc:
                best, best_auc = prm, auc
        print(f"  Best params: {best}")

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

    models = {}
    for name, fac in factories.items():
        print(f"  Training {name} (5-fold CV)...")
        oof, folds, _ = cv_oof(fac, Xtr, ytr)
        thr = best_threshold(ytr, oof)
        Xu, yu = data.upsample(Xtr, ytr, rng)
        model = fac().fit(Xu, yu)
        models[name] = {"model": model, "threshold": thr}
        print(f"    done — threshold={thr:.4f}")

    return models


def predict_interactive(models, X_input):
    """Make predictions with all models on user input."""
    print(f"\n{'Model':<35} {'Probability':>12} {'Threshold':>10} {'Prediction':>12}")
    print("-" * 70)
    for name, m in models.items():
        prob = m["model"].predict_proba(X_input)[0]
        thr = m["threshold"]
        pred = "PULSAR" if prob >= thr else "NON-PULSAR"
        print(f"{name:<35} {prob:>12.4f} {thr:>10.4f} {pred:>12}")


def run_supervised(out_dir, demo=False, synthetic=False, quick=False, interactive=False):
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

    # Train models
    print_section("Training models")
    models = train_models(Xtr, ytr, demo=demo, quick=quick)

    # Evaluate on test set
    print_section("Test Set Evaluation")
    curves = {}
    res = {
        "n_samples": int(len(y)),
        "n_pulsars": int(y.sum()),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "synthetic": synthetic,
        "demo_mode": demo,
        "models": {},
    }

    for name, m in models.items():
        p = m["model"].predict_proba(Xte)
        curves[name] = p
        tm = metrics(yte, p, 0.5)
        tm_thr = metrics(yte, p, m["threshold"])

        res["models"][name] = {
            "tuned_threshold": m["threshold"],
            "test_at_0.5": tm,
            "test_at_tuned_threshold": tm_thr,
        }

        print(f"{name:<35} Acc@0.5={tm['accuracy']:.4f}  Rec@0.5={tm['recall']:.4f}  "
              f"Thr={m['threshold']:.4f}  Rec@thr={tm_thr['recall']:.4f}  "
              f"Prec@thr={tm_thr['precision']:.4f}  AUC={tm_thr['roc_auc']:.4f}")

    # Summary table
    print_section("Summary Table")
    print(f"{'Model':<30} {'Acc@0.5':>8} {'Rec@0.5':>8} {'Thresh':>7} {'Rec@thr':>8} {'Prec@thr':>8} {'AUC':>6}")
    print("-" * 85)
    for name, m in res["models"].items():
        t05 = m["test_at_0.5"]
        tthr = m["test_at_tuned_threshold"]
        print(f"{name:<30} {t05['accuracy']:.4f}  {t05['recall']:.4f}  "
              f"{res['models'][name]['tuned_threshold']:.3f}  "
              f"{tthr['recall']:.4f}  {tthr['precision']:.4f}  {tthr['roc_auc']:.4f}")

    # Generate figures
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

    # Interactive prediction loop
    if interactive:
        print_header("INTERACTIVE PREDICTION MODE")
        print("Enter feature values for a candidate. Type 'q' at any prompt to exit.\n")
        while True:
            X_input = get_feature_input()
            if X_input is None:
                print("\nExiting interactive mode.")
                break
            predict_interactive(models, X_input)


def main():
    ap = argparse.ArgumentParser(description="Supervised pulsar classification baseline")
    ap.add_argument("--demo", action="store_true",
                    help="Run in demo mode: skip hyperparameter search, use paper params")
    ap.add_argument("--interactive", action="store_true",
                    help="After training, enter interactive prediction mode")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    ap.add_argument("--quick", action="store_true", help="Tiny search / few trees for fast check")
    ap.add_argument("--out", default="results", help="Output directory")
    args = ap.parse_args()

    run_supervised(args.out, demo=args.demo, synthetic=args.synthetic,
                   quick=args.quick, interactive=args.interactive)


if __name__ == "__main__":
    main()