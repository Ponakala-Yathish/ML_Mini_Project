"""Supervised part: GDA baseline vs Random Forest (scikit-learn, tuned) vs Random Forest (from scratch)."""
import argparse, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from htru import data
from htru.gda import GDA
from htru.forest import RandomForest
from htru.evaluate import SkRF, cv_oof, mean_folds, metrics, best_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true", help="smoke-test on fake data (no download)")
ap.add_argument("--quick", action="store_true", help="tiny search / few trees for a fast check")
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(f"{a.out}/figures", exist_ok=True)

X, y = data.load(use_synthetic=a.synthetic)
Xtr, Xte, ytr, yte = data.split(X, y)
print(f"{len(y)} samples, {y.sum()} pulsars ({y.mean():.1%}); train {len(ytr)}, test {len(yte)}")
rng = np.random.default_rng(0)
res = {"n_samples": int(len(y)), "n_pulsars": int(y.sum()), "n_train": int(len(ytr)), "n_test": int(len(yte)),
       "synthetic": a.synthetic, "models": {}}

# ---- 1) hyper-parameter search for scikit-learn RF (randomized search, CV with in-fold upsampling) ----
n_iter = 3 if a.quick else 12
grid = dict(n_estimators=[20] if a.quick else [100, 200, 300], min_samples_split=[2, 5, 10],
            max_features=[2, 3, 4, "sqrt"], max_depth=[10, 30, 60, 110, None])
best, best_auc = None, -1
for prm in ParameterSampler(grid, n_iter, random_state=0):
    _, folds, _ = cv_oof(lambda: SkRF(**prm), Xtr, ytr)
    auc = mean_folds(folds)["roc_auc"]
    print("  search", prm, f"AUC={auc:.4f}")
    if auc > best_auc: best, best_auc = prm, auc
print("best RF params:", best)

factories = {
    "GDA (baseline)": lambda: GDA(),
    "Random Forest (scikit-learn)": lambda: SkRF(**best),
    "Random Forest (from scratch)": lambda: RandomForest(n_trees=8 if a.quick else 40, max_depth=12,
                                                         min_samples_split=5, max_features=3),
}
res["rf_best_params"] = {k: (v if v is None or isinstance(v, str) else int(v)) for k, v in best.items()}
curves = {}
for name, fac in factories.items():
    print("evaluating", name)
    oof, folds, gap = cv_oof(fac, Xtr, ytr)
    thr = best_threshold(ytr, oof)                       # threshold picked from out-of-fold predictions
    cv = mean_folds(folds)
    cv_thr = metrics(ytr, oof, thr)
    Xu, yu = data.upsample(Xtr, ytr, rng)
    p = fac().fit(Xu, yu).predict_proba(Xte)             # final model: all training data (upsampled)
    res["models"][name] = {"cv_mean_at_0.5": cv, "cv_oof_at_tuned_threshold": cv_thr,
                           "cv_val_minus_train_brier": gap, "tuned_threshold": thr,
                           "test_at_0.5": metrics(yte, p, 0.5), "test_at_tuned_threshold": metrics(yte, p, thr)}
    curves[name] = p
    tm = res["models"][name]
    print(f"   CV recall {cv['recall']:.3f} | test recall@0.5 {tm['test_at_0.5']['recall']:.3f}"
          f" | thr {thr:.2f} -> test recall {tm['test_at_tuned_threshold']['recall']:.3f}")

# ---- figures ----
F = f"{a.out}/figures"
plt.figure(figsize=(5, 3.4))
for n, p in curves.items():
    f, t, _ = roc_curve(yte, p); plt.plot(f, t, label=n)
plt.plot([0, 1], [0, 1], "k:", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("Recall (TPR)")
plt.title("ROC curves (test set)"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{F}/roc.png", dpi=200); plt.close()

plt.figure(figsize=(5, 3.4))
for n, p in curves.items():
    pr, rc, _ = precision_recall_curve(yte, p); plt.plot(rc, pr, label=n)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test set)")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{F}/pr.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, len(curves), figsize=(10, 3), sharey=True)
ths = np.linspace(0.01, 0.99, 99)
for a_, (n, p) in zip(ax, curves.items()):
    tpr = [((p >= t) & (yte == 1)).sum() / (yte == 1).sum() for t in ths]
    fpr = [((p >= t) & (yte == 0)).sum() / (yte == 0).sum() for t in ths]
    a_.plot(ths, tpr, label="TPR"); a_.plot(ths, fpr, label="FPR"); a_.plot(ths, np.array(tpr) - np.array(fpr), label="TPR-FPR")
    a_.set_title(n, fontsize=8); a_.set_xlabel("Threshold")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{F}/threshold.png", dpi=200); plt.close()

plt.figure(figsize=(4, 3.4))
fr, mp = calibration_curve(yte, curves["Random Forest (scikit-learn)"], n_bins=10)
plt.plot(mp, fr, "o-"); plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("Predicted probability"); plt.ylabel("Fraction of pulsars")
plt.title("RF calibration"); plt.tight_layout(); plt.savefig(f"{F}/calibration.png", dpi=200); plt.close()

json.dump(res, open(f"{a.out}/supervised.json", "w"), indent=2)
print("saved", f"{a.out}/supervised.json")
