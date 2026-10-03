# Radio Pulsar Candidate Classification & Clustering (UE24CS352A ML Mini-Project)

Identify real pulsars among RFI/noise candidates (supervised) and cluster the confirmed pulsars (unsupervised) on the
**HTRU2** dataset. Based on the CS229 report *Application of machine learning methods to identify and categorize radio pulsar signal candidates*.

| Part | Methods (all core algorithms written from scratch unless noted) |
|---|---|
| Supervised | Gaussian Discriminant Analysis (baseline), Random Forest from scratch (`htru/forest.py`), scikit-learn Random Forest (randomized hyper-parameter search) |
| Unsupervised | K-means (scikit-learn, baseline), Self-Organizing Map (`htru/som.py`), PCA for visualisation only |
| Evaluation | Recall (primary), accuracy, precision, F1, ROC-AUC, PR curve, calibration, Silhouette coefficient |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
The dataset (`HTRU_2.csv`, UCI id 372) is downloaded automatically into `data/` on first run.
If you are offline: download `HTRU2.zip` from https://archive.ics.uci.edu/dataset/372/htru2, extract, and place `HTRU_2.csv` in `data/`.

## Run
```bash
python run_supervised.py      # -> results/supervised.json + figures (ROC, PR, threshold, calibration)
python run_unsupervised.py    # -> results/unsupervised.json + figures (PCA correlations, clusters)
python make_writeup.py --team "Name1 (SRN1), Name2 (SRN2)" --team-no 12 --problem-no 7 --repo "https://github.com/<user>/<repo>"
                              # -> writeup.pdf (built from the saved results, fits on one A4 page)
```
Quick sanity check without the dataset / with tiny settings: `python run_supervised.py --synthetic --quick --out /tmp/smoke`
(synthetic data is for smoke-testing only – never report its numbers).
Full supervised run takes a few minutes (the from-scratch forest and the SOM are pure NumPy).

## Method notes
* Stratified 80/20 train/test split (seed 42); 5-fold CV on the training part.
* Class imbalance (~9% pulsars): random minority oversampling to 50/50, applied **only to training data inside each fold** (no duplicated rows in validation/test).
* Decision threshold is chosen on out-of-fold predictions (max TPR-FPR), never on the test set.
* Clustering is trained on the 1,639 pulsars only, on all 8 features; results reported for raw and standardized features.

## Structure
```
htru/data.py  gda.py  forest.py  som.py  evaluate.py     # library code
run_supervised.py  run_unsupervised.py  make_writeup.py  # entry points
results/                                                 # generated json + figures
```
