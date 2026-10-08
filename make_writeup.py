"""Builds the 2-page PDF write-up from results/*.json (so every number comes from an actual run)."""
import argparse, html, json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from pypdf import PdfReader

ap = argparse.ArgumentParser()
ap.add_argument("--team", default="Member 1 (SRN), Member 2 (SRN)")
ap.add_argument("--team-no", default="<team no.>")
ap.add_argument("--problem-no", default="<problem no.>")
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="writeup.pdf")
ap.add_argument("--repo", default="<GitHub repo URL>")
a = ap.parse_args()
for k in ('team','team_no','problem_no','repo'): setattr(a, k, html.escape(getattr(a, k)))
S = json.load(open(f"{a.results}/supervised.json")); U = json.load(open(f"{a.results}/unsupervised.json"))

B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.3, leading=10.2, spaceAfter=2.5)
H = ParagraphStyle("h", parent=B, fontName="Helvetica-Bold", fontSize=9.6, leading=11.5, spaceBefore=4, spaceAfter=1.5,
                   textColor=colors.HexColor("#1F3A5F"))
T = ParagraphStyle("t", parent=B, fontName="Helvetica-Bold", fontSize=13, leading=15, spaceAfter=1)
C = ParagraphStyle("c", parent=B, fontSize=7.4, leading=9, textColor=colors.HexColor("#444444"))
pc = lambda v: f"{100*v:.1f}%"

def table(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.4), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.4),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF4")),
                           ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t

M = S["models"]
rows = [["Model", "CV recall", "Test acc.", "Test recall @0.5", "Threshold*", "Test recall @thr", "Test prec. @thr", "Test AUC"]]
for n, m in M.items():
    tt = m["test_at_tuned_threshold"]
    rows.append([n, pc(m["cv_mean_at_0.5"]["recall"]), pc(m["test_at_0.5"]["accuracy"]), pc(m["test_at_0.5"]["recall"]),
                 f"{m['tuned_threshold']:.2f}", pc(tt["recall"]), pc(tt["precision"]), f"{tt['roc_auc']:.3f}"])
sup_tab = table(rows, [46*mm, 15*mm, 16*mm, 24*mm, 17*mm, 24*mm, 22*mm, 15*mm])

rows = [["Clustering (pulsars only)", "Features", "#clusters", "Silhouette"]]
for tag in ("raw", "standardized"):
    if tag in U["runs"]:
        r = U["runs"][tag]
        rows.append(["K-means (k=3)", tag, "3", f"{r['kmeans']['silhouette']:.2f}"])
        s = r["som"]["silhouette"]
        rows.append(["SOM (5x5 map)", tag, str(r["som"]["n_clusters"]), "n/a" if s is None else f"{s:.2f}"])
uns_tab = table(rows, [48*mm, 30*mm, 22*mm, 22*mm])

best = max(M, key=lambda n: M[n]["test_at_tuned_threshold"]["recall"])
sk, sc, g = M["Random Forest (scikit-learn)"], M["Random Forest (from scratch)"], M["GDA (baseline)"]
kbest = max((U["runs"][t]["kmeans"]["silhouette"], t) for t in U["runs"] if "kmeans" in U["runs"][t])
conc = (f"<b>Supervised.</b> On the held-out test set, {best} achieved the highest recall ({pc(M[best]['test_at_tuned_threshold']['recall'])}) "
        f"at its cross-validated threshold. The tuned scikit-learn Random Forest reached {pc(sk['test_at_tuned_threshold']['recall'])} recall / "
        f"{sk['test_at_tuned_threshold']['roc_auc']:.3f} AUC versus {pc(g['test_at_tuned_threshold']['recall'])} / {g['test_at_tuned_threshold']['roc_auc']:.3f} "
        f"for the GDA baseline, and our from-scratch forest reached {pc(sc['test_at_tuned_threshold']['recall'])} / {sc['test_at_tuned_threshold']['roc_auc']:.3f}. "
        f"For the scikit-learn forest, moving the threshold from 0.5 to {sk['tuned_threshold']:.2f} changed test recall from {pc(sk['test_at_0.5']['recall'])} to {pc(sk['test_at_tuned_threshold']['recall'])} "
        f"(precision {pc(sk['test_at_0.5']['precision'])} to {pc(sk['test_at_tuned_threshold']['precision'])}); this trade-off matters because a missed pulsar is costlier than an extra candidate for human review. "
        f"<b>Unsupervised.</b> The best K-means silhouette was {kbest[0]:.2f} ({kbest[1]} features); values well below 1 indicate that the pulsars do not form clearly separated groups in these 8 summary statistics. "
        f"This is expected: the features are integrated over frequency/time, so information used to define pulsar sub-classes (e.g. spin period) is not present. "
        f"<b>Limitations / future work.</b> Results come from a single stratified train/test split; the from-scratch forest uses quantile-based candidate thresholds and a fixed configuration; "
        f"applying the clustering to unprocessed pulse data would be more informative.")

st = [Paragraph("Identifying and Categorising Radio Pulsar Candidates with Machine Learning", T),
      Paragraph(f"UE24CS352A Machine Learning – Mini-Project &nbsp;|&nbsp; Team {a.team_no}, Problem {a.problem_no} &nbsp;|&nbsp; {a.team} &nbsp;|&nbsp; Code: {a.repo}", C)]
if S.get("synthetic"): st.append(Paragraph("<font color='red'><b>WARNING: built from SYNTHETIC smoke-test results – re-run on real data before submitting.</b></font>", B))
st += [Paragraph("1. Problem statement", H),
       Paragraph("Radio pulsar surveys produce far more candidate signals than experts can inspect, and most are radio-frequency interference (RFI) or noise. "
                 "We (i) build a classifier that flags true pulsars, optimising <b>recall</b> because missing a pulsar is more costly than reviewing a false alarm, and "
                 "(ii) cluster the confirmed pulsars to look for sub-groups. The task follows the CS229 report <i>Application of machine learning methods to identify and categorize radio pulsar signal candidates</i>.", B),
       Paragraph("2. Dataset", H),
       Paragraph(f"HTRU2 (UCI ML repository; Lyon et al., 2016; HTRU survey): {S['n_samples']:,} candidates, of which {S['n_pulsars']:,} ({100*S['n_pulsars']/S['n_samples']:.1f}%) are real pulsars – a severe class imbalance. "
                 "Each candidate has 8 continuous features: mean, standard deviation, excess kurtosis and skewness of (a) the integrated pulse profile and (b) the DM-SNR curve; label 1 = pulsar. "
                 f"We hold out a stratified 20% test set ({S['n_test']:,} rows) and use the remaining {S['n_train']:,} for 5-fold cross-validation.", B),
       Paragraph("3. Approach", H),
       Paragraph("<b>Supervised:</b> Gaussian Discriminant Analysis (baseline; shared covariance, implemented from scratch) versus Random Forest. The Random Forest is (a) scikit-learn's, tuned with a randomized search over "
                 f"n_estimators, max_depth, max_features and min_samples_split (best: {json.dumps(S['rf_best_params'])}), and (b) our own NumPy implementation (bootstrap trees, random feature subsets, entropy/information-gain splits). "
                 "To handle imbalance we randomly oversample the minority class to 50/50 <i>inside each training fold only</i>, so validation/test data are never duplicated. "
                 "The decision threshold is chosen on out-of-fold predictions by maximising TPR-FPR, not on the test set. "
                 "<b>Unsupervised:</b> trained on the pulsars only: K-means (baseline, k=3) and a from-scratch Self-Organizing Map (5x5 grid, 100 epochs, exponentially decaying learning rate and neighbourhood radius). "
                 "Clusters are evaluated with the Silhouette coefficient on all 8 features (raw, as in the paper, and standardized); PCA is used only to visualise clusters and interpret features.", B),
       Paragraph("4. Implementation overview", H),
       Paragraph("Python 3, NumPy, pandas, scikit-learn, matplotlib. Package <font face='Courier'>htru/</font> holds data handling (<font face='Courier'>data.py</font>), "
                 "<font face='Courier'>gda.py</font>, <font face='Courier'>forest.py</font>, <font face='Courier'>som.py</font> and evaluation utilities (<font face='Courier'>evaluate.py</font>); "
                 "<font face='Courier'>run_supervised.py</font> and <font face='Courier'>run_unsupervised.py</font> reproduce all results and figures, and <font face='Courier'>make_writeup.py</font> generates this document from the saved results.", B),
       Paragraph("5. Results", H), sup_tab,
       Paragraph("*Threshold chosen on out-of-fold training predictions (maximising TPR-FPR). CV recall = mean over 5 folds at threshold 0.5.", C),
       Spacer(1, 2)]
F = f"{a.results}/figures"
st.append(Table([[Image(f"{F}/roc.png", 57*mm, 39*mm), Image(f"{F}/pr.png", 57*mm, 39*mm), Image(f"{F}/clusters.png", 60*mm, 41*mm)]],
                colWidths=[60*mm, 60*mm, 62*mm]))
st += [Paragraph("Figure: ROC and precision-recall curves on the test set; K-means/SOM clusters (raw and standardized features) in PCA space.", C), uns_tab,
       Paragraph("6. Conclusions", H), Paragraph(conc, B)]

doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=14*mm, rightMargin=14*mm, topMargin=11*mm, bottomMargin=10*mm,
                        title="Pulsar candidate classification - ML mini-project")
doc.build(st)
n = len(PdfReader(a.out).pages)
print(f"wrote {a.out} ({n} page{'s' if n != 1 else ''})")
if n > 2: print("WARNING: more than 2 pages - shorten text or figures")
