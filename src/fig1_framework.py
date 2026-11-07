"""Figure 1: overview of the framework."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 110); ax.set_ylim(0, 56); ax.axis("off")
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=7.2, linespacing=1.35)
def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333"))
C1, C2, C3, C4 = "#eaf2fb", "#fdf2e9", "#eafaf1", "#f5eef8"
box(1, 37, 20, 17, "Flow data", "CICAndMal2017\n14,997 flows, 68 features\nCIC-AAGM2017\n20,000 flows, 74 features\nclass-balanced, no duplicates", C1)
box(25, 37, 25, 17, "Outer stratified 5-fold CV", "per fold:\nFIT 60%  |  CAL 20%  |  TEST 20%\nFIT: training only\nCAL: calibration / thresholds only\nTEST: reporting only", C1)
box(54, 37, 25, 17, "PSO search (FIT only)", "8 particles × 10 iterations\nw = 0.7, c1 = c2 = 1.5\nfitness = 1 − mean 3-fold CV F1\nlearning rate, max depth,\nnumber of trees, subsample", C2)
box(83, 37, 26, 17, "PSO-XGBoost", "XGBoost refitted on all FIT flows\nwith the best particle\n→ raw malware probability p(x)\n(baselines: XGBoost, LightGBM,\nAdaBoost, GBM, fixed settings)", C2)
box(54, 15, 25, 17, "Calibration (CAL)", "Platt scaling:\nq = σ(a·logit p + b)\n(isotonic regression compared)\nreliability diagram, ECE, Brier", C3)
box(25, 15, 25, 17, "Reject option (CAL)", "Chow two-threshold rule:\nwindow (τL, τH) holding a target\nshare r of CAL flows, placed to\ncontain the most CAL errors\nr = 5, 10, 15, 20, 30%", C3)
box(83, 15, 26, 17, "Mondrian conformal (CAL)", "class-wise p-values from CAL\nnonconformity 1 − p_c(x)\nprediction set {c : p-value > α}\nper-class coverage ≥ 1 − α\n(singleton = automatic decision)", C3)
box(1, 15, 20, 17, "Decision on TEST", "q ≥ τH → malware\nq ≤ τL → benign\notherwise →\nanalyst queue", C4)
box(1, 1, 108, 10, "", "", "#fbfcfc")
ax.text(55, 8.3, "Evaluation on TEST", ha="center", fontsize=9.2, weight="bold")
ax.text(55, 4.6, "accuracy, F1, MCC, ROC-AUC  •  ECE, Brier  •  accuracy, FPR and residual errors on automatic decisions vs analyst workload  •  conformal coverage\n"
        "corrected resampled t-test with Holm adjustment  •  exact TreeSHAP of PSO-XGBoost, including deferred vs automatic flows", ha="center", va="center", fontsize=7.8, linespacing=1.5)
arr(21, 45.5, 25, 45.5); arr(50, 45.5, 54, 45.5); arr(79, 45.5, 83, 45.5)
arr(96, 37, 96, 32); ax.text(97, 34.5, "raw p", fontsize=7, style="italic")
arr(88, 37, 72, 32); ax.text(76.5, 35.2, "raw p", fontsize=7, style="italic")
arr(54, 23.5, 50, 23.5); arr(25, 23.5, 21, 23.5); arr(11, 15, 11, 11); arr(96, 15, 96, 11)
from common import FIG
plt.savefig(FIG / "fig1_framework.png", dpi=300, bbox_inches="tight"); print("ok")
