"""Figures for the revised manuscript (all from the saved per-fold predictions / result CSVs)."""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, glob, re, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
sys.path.insert(0, ROOT + "/src")
from analysis_lib import platt, f1_thr, window_error, window_sym, selective
R = ROOT + "/results/"; FIG = ROOT + "/figures/"; PRED = ROOT + "/predictions/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": .6})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
AP = pd.read_parquet(ROOT + "/data/cicandmal_app_K20.parquet", columns=["app", "y"]); APP = AP.app.to_numpy(); YA = AP.y.to_numpy()

def agg(idx, p):
    g = pd.DataFrame({"app": APP[idx], "y": YA[idx], "p": p}).groupby("app").agg(y=("y", "first"), s=("p", "mean"))
    return g.y.to_numpy(), g.s.to_numpy()

def folds(ds):
    pat = "andmal_app_main_r*_f*_full.npz" if ds == "andmal" else "aagm_flow_main_r0_f*_full.npz"
    for fn in sorted(glob.glob(PRED + pat)):
        z = np.load(fn, allow_pickle=True); m = "PSO-XGBoost"
        if ds == "andmal":
            yt, st = agg(z["idx_te"], z[f"te__{m}"]); yc, sc = agg(z["idx_cal"], z[f"cal__{m}"])
        else:
            yt, st, yc, sc = z["y_te"], z[f"te__{m}"], z["y_cal"], z[f"cal__{m}"]
        yield fn, z, yt, st, yc, sc

# ---------------------------------------------------------------- Figure 1: framework
def fig1():
    fig, ax = plt.subplots(figsize=(12, 6.6)); ax.set_xlim(0, 120); ax.set_ylim(0, 66); ax.axis("off"); ax.grid(False)
    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333333", lw=1))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=8.2, linespacing=1.45)
    def arr(x1, y1, x2, y2, lab=None, dx=.8, dy=0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#333333"))
        if lab: ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, lab, fontsize=8, style="italic")
    C1, C2, C3, C4 = "#eaf2fb", "#fdf2e9", "#eafaf1", "#f5eef8"
    W, G = 26, 4.5; X = [0.5 + i * (W + G) for i in range(4)]
    box(X[0], 44, W, 20, "Flow data", "CICAndMal2017: 2112 apps,\nup to 20 flows per app\n(41,900 flows, 18.9% malware)\nCIC-AAGM2017: 20,000 flows\n(34.8% malware, natural share)", C1)
    box(X[1], 44, W, 20, "Outer cross-validation", "CICAndMal2017: 5 folds grouped\nby app, repeated 3 times\ntraining apps → FIT 75%, CAL 25%\nTEST apps unseen in tuning,\nfitting and calibration", C1)
    box(X[2], 44, W, 20, "Tuning on FIT only", "equal budget: 48 configurations\nPSO (8 particles, 1 + 5 rounds)\nrandom search, TPE (Optuna)\nfitness: mean ROC-AUC of\n3-fold app-grouped inner CV", C2)
    box(X[3], 44, W, 20, "Flow model → app score", "PSO-tuned XGBoost, LightGBM,\nRF, ExtraTrees + 4 defaults\nflow probability p(x)\napp score s(a) = mean p(x)\nover the app's observed flows", C2)
    box(X[3], 19, W, 20, "Calibration (CAL)", "Platt: q = σ(α·logit s + β)\n(isotonic compared)\nECE, Brier, reliability\nprior shift to π' = 10, 5, 1%", C3)
    box(X[2], 19, W, 20, "Deferral rule (CAL)", "analyst workload r\n(share of items deferred)\ncost-weighted window\nvs symmetric (Chow-type) band\nvs unweighted error window", C3)
    box(X[1], 19, W, 20, "Conformal sets (CAL)", "Mondrian p-values per class\nset = {c : p_c > ε}\ncoverage ≥ 1 − ε per class\nunder exchangeability\nsingleton → automatic", C3)
    box(X[0], 19, W, 20, "Decision on TEST", "q in [τ_L, τ_H] → analyst\notherwise:\nq ≥ t → malware\nq < t → benign", C4)
    box(0.5, 1, 117, 13.5, "", "", "#fbfcfc")
    ax.text(59, 12.2, "Evaluation on TEST (CIC-AAGM2017: stratified 5-fold by flow, no app identifiers available)", ha="center", fontsize=9.5, weight="bold")
    ax.text(59, 6.2, "ROC-AUC, PR-AUC, F1, MCC at flow and app level  •  evidence curve over k flows  •  ECE, Brier  •  error, FNR, FPR and cost vs workload\n"
            "per-class coverage with Clopper–Pearson CIs  •  precision at 10/5/1% prevalence  •  corrected repeated k-fold t-tests, Holm  •  TreeSHAP",
            ha="center", va="center", fontsize=8.2, linespacing=1.6)
    for i in range(3): arr(X[i] + W + .5, 54, X[i + 1] - .5, 54)
    arr(X[3] + W / 2, 44, X[3] + W / 2, 39.5, "s(a)")
    arr(X[3] - .5, 29, X[2] + W + .5, 29, "q", dx=-.6, dy=.8); arr(X[2] - .5, 29, X[1] + W + .5, 29); arr(X[1] - .5, 29, X[0] + W + .5, 29)
    arr(X[0] + W / 2, 19, X[0] + W / 2, 14.5)
    plt.savefig(FIG + "fig1_framework.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 2: evidence curve
def fig2():
    K = pd.read_csv(R + "K.csv"); K = K[(K.features == "main") & (K.modelset == "full")]
    models = ["PSO-XGBoost", "XGBoost", "PSO-LightGBM", "PSO-RF", "GBM", "AdaBoost"]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for i, m in enumerate(models):
        g = K[K.model == m].groupby("k").roc_auc; mu, sd, n = g.mean(), g.std(), g.size()
        ax.plot(mu.index, mu.values, color=CAT[i], marker=MK[i], ms=6, lw=2, label=m)
        if m == "PSO-XGBoost":
            h = 1.96 * sd / np.sqrt(n); ax.fill_between(mu.index, mu - h, mu + h, color=CAT[i], alpha=.15, lw=0)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20]); ax.set_xticklabels(["1", "2", "3", "5", "10", "20"])
    ax.set_xlabel("Flows observed per test app (k)"); ax.set_ylabel("App-level ROC-AUC (unseen apps)")
    ax.set_ylim(.5, .95); ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout(); plt.savefig(FIG + "fig2_evidence.png", dpi=300); plt.close()

# ---------------------------------------------------------------- Figure 3: reliability
def rel_points(y, q, nb=15, minn=10):
    b = np.minimum((q * nb).astype(int), nb - 1); xs, ys, ns = [], [], []
    for k in range(nb):
        s = b == k
        if s.sum() >= minn: xs.append(q[s].mean()); ys.append(y[s].mean()); ns.append(s.sum())
    return np.array(xs), np.array(ys)
def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, ds, ttl, minn in ((axs[0], "andmal", "(a) CICAndMal2017, app level", 10), (axs[1], "aagm", "(b) CIC-AAGM2017, flow level", 20)):
        Y, RAW, PL = [], [], []
        for fn, z, yt, st, yc, sc in folds(ds):
            Y.append(yt); RAW.append(st); PL.append(platt(sc, yc)(st))
        Y, RAW, PL = map(np.concatenate, (Y, RAW, PL))
        ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
        for (q, lab, c, mk) in ((RAW, "uncalibrated", CAT[1], "s"), (PL, "Platt (CAL apps)" if ds == "andmal" else "Platt (CAL)", CAT[0], "o")):
            x, y = rel_points(Y, q, minn=minn); ax.plot(x, y, marker=mk, color=c, lw=2, ms=6, label=lab)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean predicted malware probability")
        ax.set_ylabel("Observed malware fraction"); ax.set_title(ttl, fontsize=9.5, loc="left"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG + "fig3_reliability.png", dpi=300); plt.close()

# ---------------------------------------------------------------- Figure 4: risk / cost vs deferral
def curves(ds, rates=np.round(np.arange(0, .501, .025), 3)):
    rows = []
    for fn, z, yt, st, yc, sc in folds(ds):
        f = platt(sc, yc); qc, qt = f(sc), f(st); thr = f1_thr(qc, yc)
        for rule in ("cost_window", "symmetric", "error_window"):
            for r in rates:
                win = None if r == 0 else (window_error(qc, yc, thr, r, cfn=10) if rule == "cost_window" else
                                           window_error(qc, yc, thr, r) if rule == "error_window" else window_sym(qc, thr, r))
                rows.append(dict(fold=fn, rule=rule, target=r, **selective(qt, yt, thr, win)))
    return pd.DataFrame(rows)
def fig4():
    lab = {"cost_window": r"cost-weighted window ($c_{FN}=10$)", "symmetric": "symmetric band (Chow-type)", "error_window": "unweighted error window"}
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.6))
    for j, (ds, ttl) in enumerate((("andmal", "CICAndMal2017 (apps)"), ("aagm", "CIC-AAGM2017 (flows)"))):
        d = curves(ds); d.to_csv(R + f"curves_{ds}.csv", index=False); g = d.groupby(["rule", "target"]).mean(numeric_only=True)
        for i, (col, yl) in enumerate((("fnr_auto", "Missed-malware rate (FNR) of\nautomatic decisions"), ("cost_per10k", "Cost per 10,000 items\n" + r"($c_{FN}=10,\ c_{FP}=1,\ c_R=0.5$)"))):
            ax = axs[i, j]
            for k, rule in enumerate(("cost_window", "symmetric", "error_window")):
                x = g.loc[rule]; ax.plot(100 * x.defer, x[col] if col == "cost_per10k" else 100 * x[col], color=CAT[k], marker=MK[k], ms=4, lw=2, label=lab[rule])
            ax.set_xlabel("Items deferred to the analyst (%)"); ax.set_ylabel(yl + (" (%)" if col == "fnr_auto" else ""))
            ax.set_title(f"({'abcd'[2 * i + j]}) {ttl}", fontsize=9.5, loc="left")
    axs[0, 0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(FIG + "fig4_deferral.png", dpi=300); plt.close()

# ---------------------------------------------------------------- Figure 5: tuner convergence
def fig5():
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, pat, ttl in ((axs[0], "andmal_app_main_r0_f*_full.npz", "(a) CICAndMal2017 (app-grouped inner CV)"),
                         (axs[1], "aagm_flow_main_r0_f*_full.npz", "(b) CIC-AAGM2017")):
        H = {m: [] for m in ("PSO-XGBoost", "RS-XGBoost", "TPE-XGBoost")}
        for fn in sorted(glob.glob(PRED + pat)):
            info = json.loads(str(np.load(fn, allow_pickle=True)["info"]))
            for m in H: H[m].append(info[m]["convergence_auc"])
        for k, (m, lab) in enumerate((("PSO-XGBoost", "PSO"), ("RS-XGBoost", "random search"), ("TPE-XGBoost", "TPE (Optuna)"))):
            a = np.array(H[m]); x = 8 * np.arange(1, a.shape[1] + 1)
            ax.plot(x, a.mean(0), color=CAT[k], marker=MK[k], ms=6, lw=2, label=lab)
        ax.set_xlabel("Configurations evaluated"); ax.set_ylabel("Best inner-CV ROC-AUC (mean of 5 folds)")
        ax.set_xticks(x); ax.set_title(ttl, fontsize=9.5, loc="left"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG + "fig5_convergence.png", dpi=300); plt.close()

# ---------------------------------------------------------------- Figure 6: SHAP (two panels side by side)
def fig6():
    from PIL import ImageDraw, ImageFont
    ims = [Image.open(FIG + f).convert("RGB") for f in ("shap_andmal_app_main.png", "shap_aagm_flow_main.png")]
    ims = [i.crop((0, int(i.height * .045), i.width, i.height)) for i in ims]          # remove the internal title
    h = max(i.height for i in ims); ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    top = 110; out = Image.new("RGB", (sum(i.width for i in ims) + 80, h + top), "white"); x = 0
    d = ImageDraw.Draw(out); f = ImageFont.truetype("DejaVuSans.ttf", 64)
    for i, lab in zip(ims, ("(a) CICAndMal2017 (unseen apps, 5 folds of repeat 1)", "(b) CIC-AAGM2017 (5 folds)")):
        out.paste(i, (x, top)); d.text((x + 40, 20), lab, fill="black", font=f); x += i.width + 80
    out.save(FIG + "fig6_shap.png", dpi=(300, 300))

if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]: globals()[f](); print(f, "ok", flush=True)
