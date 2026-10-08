# Radio Pulsar Candidate Classification & Clustering (UE24CS352A ML Mini-Project)

Identify real pulsars among RFI/noise candidates (supervised) and cluster the confirmed pulsars (unsupervised) on the **HTRU2** dataset.

Based on the CS229 report: *Application of machine learning methods to identify and categorize radio pulsar signal candidates* (Debesai, Gutierrez & Koyluoglu, 2020).

## Problem

Radio pulsar surveys produce far more candidate signals than experts can inspect. Most candidates are radio-frequency interference (RFI) or noise. This project implements the reference methodology from the paper to:

1. **Supervised**: Classify candidates as pulsar (1) or non-pulsar (0), optimizing **recall** because missing a pulsar is costlier than reviewing a false alarm.
2. **Unsupervised**: Cluster the confirmed pulsars to look for sub-groups.

## Dataset

**HTRU2** (UCI ML Repository, ID 372) — High Time Resolution Universe Survey.

- **17,898** candidates total
- **1,639** real pulsars (~9.2%)
- **16,259** non-pulsars (RFI/noise)
- **8 continuous features** (statistics of integrated pulse profile + DM-SNR curve):
  1. Mean of the integrated profile
  2. Standard deviation of the integrated profile
  3. Excess kurtosis of the integrated profile
  4. Skewness of the integrated profile
  5. Mean of the DM-SNR curve
  6. Standard deviation of the DM-SNR curve
  7. Excess kurtosis of the DM-SNR curve
  8. Skewness of the DM-SNR curve

## Algorithms Implemented

| Part | Methods |
|------|---------|
| **Supervised** | Gaussian Discriminant Analysis (GDA, baseline, from scratch), Random Forest (scikit-learn, tuned), Random Forest (from scratch, NumPy) |
| **Unsupervised** | K-means (k=3, scikit-learn), Self-Organizing Map (5×5 grid, from scratch), PCA (visualisation only) |
| **Evaluation** | Recall (primary), accuracy, precision, F1, ROC-AUC, PR-AUC, calibration, Silhouette coefficient |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset (`HTRU_2.csv`) downloads automatically into `data/` on first run.  
If offline: download `HTRU2.zip` from https://archive.ics.uci.edu/dataset/372/htru2, extract, and place `HTRU_2.csv` in `data/`.

## Running the Baseline

### Demo Mode (Fast, for Live Presentation)

```bash
# Supervised: uses paper hyperparameters, skips search (~30-60 sec)
python run_supervised.py --demo

# Unsupervised: raw features only, no k-sweep (~10-20 sec)
python run_unsupervised.py --demo
```

### Full Reproduction Mode

```bash
# Supervised: randomized hyperparameter search (12 iterations, ~3-5 min)
python run_supervised.py

# Unsupervised: raw + standardized, includes k-sweep (~1-2 min)
python run_unsupervised.py
```

### Smoke Test (No Data Download)

```bash
python run_supervised.py --synthetic --quick --out /tmp/smoke
python run_unsupervised.py --synthetic --quick --out /tmp/smoke
```

## Outputs

| Command | Output |
|---------|--------|
| `run_supervised.py` | `results/supervised.json` + `results/figures/` (roc.png, pr.png, threshold.png, calibration.png) |
| `run_unsupervised.py` | `results/unsupervised.json` + `results/figures/` (pca_corr.png, clusters.png) |
| `make_writeup.py` | `writeup.pdf` (1-page summary from results) |

## Expected Results (Approximate)

### Supervised (Test Set, 3,580 samples)

| Model | Accuracy @ 0.5 | Recall @ 0.5 | Tuned Threshold | Recall @ Tuned | ROC-AUC |
|-------|---------------|--------------|-----------------|----------------|---------|
| GDA (baseline) | ~97.9% | ~89.6% | ~0.19 | ~92.1% | ~0.97 |
| RF (sklearn) | ~97.6% | ~89.6% | ~0.31 | ~91.8% | ~0.98 |
| RF (from scratch) | ~97.8% | ~89.9% | ~0.32 | ~91.8% | ~0.97 |

*Threshold chosen on out-of-fold CV predictions (max TPR-FPR), never on test set.*

### Unsupervised (1,639 pulsars, raw features)

| Method | Silhouette | Clusters / Active Cells | Sizes |
|--------|------------|------------------------|-------|
| K-means (k=3) | ~0.44 | 3 | [974, 621, 44] |
| SOM (5×5) | ~0.44 | 3–4 | ~3 clusters dominate |

*Results are comparable to the reference paper (K-means Silhouette 0.44, SOM 3 non-empty cells, Silhouette 0.44).*

## Method Notes

- **Stratified 80/20 train/test split** (seed 42); 5-fold CV on training part.
- **Class imbalance** (~9% pulsars): random minority oversampling to 50/50, applied **only inside each training fold** (no duplication in validation/test).
- **Threshold selection**: Youden's J (max TPR − FPR) on **out-of-fold predictions**, never on the test set.
- **Clustering**: trained on the 1,639 pulsars only, on all 8 raw features (as in paper); PCA used only for visualisation.
- **Seeds**: All randomness controlled (train/test split seed=42, CV seed=0, upsampling seed=0, RF seed=0, SOM seed=0, K-means seed=0).

## Repository Structure

```
htru/
    data.py       # loading, splitting, upsampling
    gda.py        # Gaussian Discriminant Analysis (from scratch)
    forest.py     # Random Forest (from scratch)
    som.py        # Self-Organizing Map (from scratch)
    evaluate.py   # CV, metrics, threshold selection
run_supervised.py   # supervised entry point
run_unsupervised.py # unsupervised entry point
make_writeup.py     # generates writeup.pdf from results
requirements.txt
README.md
results/
    supervised.json
    unsupervised.json
    figures/
```

## Generating the Write-up

```bash
python make_writeup.py --team "Name (SRN)" --team-no 1 --problem-no 7 \
    --repo "https://github.com/<user>/<repo>"
```

## Notes

- This implementation reproduces the **reference methodology** from the CS229 paper.
- Results are **comparable** to the paper; exact numbers differ slightly due to implementation details (e.g., scikit-learn versions, random seeds, CV folds).
- We do **not** claim state-of-the-art performance or novel methods.
- No SMOTE, GMM, Ward, XGBoost, or other extensions are included in the baseline.