"""Final analysis for the manuscript (PSO-XGBoost with calibrated triage).
Scope: CICAndMal2017 and CIC-AAGM2017 (stratified 5-fold, random flow-level splits);
models: PSO-XGBoost (proposed) vs XGBoost, LightGBM, AdaBoost, GBM (fixed settings).
Input: the per-fold probabilities saved by calib_train.py on the CAL split and the TEST fold.
All calibrators, thresholds and conformal scores are fitted on CAL only; TEST is used for reporting only.
Statistics: corrected resampled t-test (Nadeau & Bengio 2003) with rho = n_test / n_fit (model trained on FIT only),
Holm adjustment within dataset and metric family; wins/losses over the 5 folds are reported.
Input: results/fold_predictions/*.npz (from train.py). Outputs: results/*.csv, figures/fig2_*.png, figures/fig3_*.png
Usage: python src/analysis.py
"""
import json, numpy as np, pandas as pd, matplotlib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, matthews_corrcoef, accuracy_score
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from common import PRED as R, RES as OUT, FIG, DATA
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
DS = ["CICAndMal2017", "CIC-AAGM2017"]; MODELS = ["PSO-XGBoost", "XGBoost", "LightGBM", "AdaBoost", "GBM"]
PROP = "PSO-XGBoost"; RATES = [0, .05, .10, .15, .20, .30]; ALPHAS = [.01, .05, .10]; EPS = 1e-6
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

def ece(y, p, nb=15):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any())
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def platt(pc, yc):
    m = LogisticRegression(C=1e6).fit(logit(pc)[:, None], yc); return (lambda p: m.predict_proba(logit(p)[:, None])[:, 1]), m
def isotonic(pc, yc):
    m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pc, yc); return m.predict

def chow(pc, yc, pt, yt, r):
    n = len(pc); w = int(round(r * n)); yh = (pt >= .5).astype(int)
    if w == 0: keep = np.ones(len(pt), bool); lo = hi = np.nan
    else:
        o = np.argsort(pc); ps, err = pc[o], ((pc[o] >= .5).astype(int) != yc[o]).astype(int)
        cs = np.concatenate([[0], np.cumsum(err)]); inside = cs[w:] - cs[:-w]
        mid = np.abs((ps[:n - w + 1] + ps[w - 1:]) / 2 - .5)
        s = np.lexsort((mid, -inside))[0]; lo, hi = ps[s], ps[s + w - 1]
        keep = ~((pt >= lo) & (pt <= hi))
    yk, pk = yt[keep], yh[keep]
    tp = ((pk == 1) & (yk == 1)).sum(); fp = ((pk == 1) & (yk == 0)).sum(); fn = ((pk == 0) & (yk == 1)).sum(); tn = ((pk == 0) & (yk == 0)).sum()
    e = yh != yt
    return {"target_defer": r, "defer_rate": 1 - keep.mean(), "acc_auto": (yk == pk).mean(),
            "precision_auto": tp / max(tp + fp, 1), "recall_auto": tp / max(tp + fn, 1),
            "f1_auto": 2 * tp / max(2 * tp + fp + fn, 1), "fpr_auto": fp / max(fp + tn, 1), "fnr_auto": fn / max(fn + tp, 1),
            "errors_caught_pct": 100 * e[~keep].sum() / max(e.sum(), 1), "errors_auto_per_10k": 1e4 * (yk != pk).sum() / len(yt),
            "tau_low": lo, "tau_high": hi}, keep

def mondrian(pc, yc, pt, yt, a):
    sets = np.zeros((len(pt), 2), bool)
    for c in (0, 1):
        sc = 1 - (pc if c == 1 else 1 - pc)[yc == c]; st = 1 - (pt if c == 1 else 1 - pt)
        pv = (len(sc) - np.searchsorted(np.sort(sc), st, side="left") + 1) / (len(sc) + 1)
        sets[:, c] = pv > a
    single = sets.sum(1) == 1; pred = sets[:, 1].astype(int)
    return {"alpha": a, "coverage_malware": sets[yt == 1, 1].mean(), "coverage_benign": sets[yt == 0, 0].mean(),
            "singleton_rate": single.mean(), "both_rate": (sets.sum(1) == 2).mean(), "empty_rate": (sets.sum(1) == 0).mean(),
            "acc_singleton": (pred[single] == yt[single]).mean() if single.any() else np.nan}

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj

D, C, S, M, P, pool, sizes = [], [], [], [], [], {}, {}
for ds in DS:
    for fn in sorted(R.glob(f"{ds}_fold*.npz")):
        z = np.load(fn, allow_pickle=True); fold = int(fn.stem[-1]); yc, yt = z["y_cal"], z["y_te"]
        prm = json.loads(str(z["params"]))[PROP]; P.append({"dataset": ds, "fold": fold, **prm})
        sizes.setdefault(ds, []).append((len(yc), len(yt)))
        for m in MODELS:
            pc, pt = z[f"cal__{m}"], z[f"te__{m}"]; fP, _ = platt(pc, yc); fI = isotonic(pc, yc); yh = (pt >= .5).astype(int)
            D.append({"dataset": ds, "fold": fold, "model": m, "accuracy": accuracy_score(yt, yh), "f1": f1_score(yt, yh),
                      "mcc": matthews_corrcoef(yt, yh), "roc_auc": roc_auc_score(yt, pt)})
            for meth, q in (("raw", pt), ("platt", fP(pt)), ("isotonic", fI(pt))):
                C.append({"dataset": ds, "fold": fold, "model": m, "calibration": meth, "ECE": ece(yt, q),
                          "Brier": brier_score_loss(yt, q)})
                pool.setdefault((ds, m, meth), []).append((yt, q))
            qc, qt = fP(pc), fP(pt)
            for r in RATES:
                row, keep = chow(qc, yc, qt, yt, r); S.append({"dataset": ds, "fold": fold, "model": m, **row})
                if m == PROP and r == .20:
                    np.savez(R / f"keep20_{ds}_fold{fold}.npz", keep=keep, q=qt, y=yt)
            for a in ALPHAS: M.append({"dataset": ds, "fold": fold, "model": m, **mondrian(pc, yc, pt, yt, a)})
D, C, S, M, P = map(pd.DataFrame, (D, C, S, M, P))
for n, d in (("discrimination", D), ("calibration", C), ("triage", S), ("conformal", M), ("pso_params", P)):
    d.to_csv(OUT / f"{n}_perfold.csv", index=False)

# --- statistics -------------------------------------------------------------------------------------------------
fit_n = {"CICAndMal2017": None, "CIC-AAGM2017": None}
z0 = {ds: np.load(sorted(R.glob(f"{ds}_fold0.npz"))[0], allow_pickle=True) for ds in DS}
T = []
def ttests(ds, fam, metric, piv, rho, better_low=False):
    rows = []
    for b in MODELS[1:]:
        d = (piv[PROP] - piv[b]).to_numpy(); d = -d if better_low else d; n = len(d)
        se = np.sqrt((1 / n + rho) * d.var(ddof=1)); t = d.mean() / se if se > 0 else np.inf
        rows.append({"dataset": ds, "family": fam, "metric": metric, "baseline": b,
                     "diff": (piv[PROP] - piv[b]).mean(), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "t": t, "p": 2 * stats.t.sf(abs(t), n - 1)})
    return rows
for ds in DS:
    n_cal, n_te = np.mean(sizes[ds], 0); n_all = len(pd.read_parquet(DATA / f"{ds}.parquet"))
    n_fit = n_all - n_cal - n_te; rho = n_te / n_fit
    print(f"{ds}: n={n_all} fit~{n_fit:.0f} cal~{n_cal:.0f} test~{n_te:.0f} rho={rho:.3f}")
    fams = {"discrimination": [], "triage": [], "probability": []}
    for m in ("accuracy", "f1", "mcc", "roc_auc"):
        fams["discrimination"] += ttests(ds, "discrimination", m, D[D.dataset == ds].pivot(index="fold", columns="model", values=m), rho)
    for r in (.10, .20):
        s = S[(S.dataset == ds) & (S.target_defer == r)]
        fams["triage"] += ttests(ds, "triage", f"acc_auto@{int(r*100)}%", s.pivot(index="fold", columns="model", values="acc_auto"), rho)
    c = C[(C.dataset == ds) & (C.calibration == "platt")]
    for m in ("ECE", "Brier"):
        fams["probability"] += ttests(ds, "probability", f"{m} (Platt)", c.pivot(index="fold", columns="model", values=m), rho, True)
    # within-model effect of deferral for PSO-XGBoost (r vs no deferral), Holm over the 4 tests
    w = []
    sp = S[(S.dataset == ds) & (S.model == PROP)].pivot(index="fold", columns="target_defer")
    for r in (.10, .20):
        for key, low in (("acc_auto", False), ("fpr_auto", True)):
            d = (sp[(key, r)] - sp[(key, 0.0)]).to_numpy(); dd = -d if low else d; n = len(d)
            t = dd.mean() / np.sqrt((1 / n + rho) * dd.var(ddof=1))
            w.append({"dataset": ds, "family": "within PSO-XGBoost", "metric": f"{key}@{int(r*100)}% vs 0%", "baseline": "-",
                      "diff": d.mean(), "wins": int((dd > 0).sum()), "losses": int((dd < 0).sum()), "t": t, "p": 2 * stats.t.sf(abs(t), n - 1)})
    fams["within"] = w
    for f, rows in fams.items():
        r_ = pd.DataFrame(rows); r_["p_holm"] = holm(r_.p); T.append(r_)
T = pd.concat(T); T.to_csv(OUT / "stats.csv", index=False)

# --- printed summaries -------------------------------------------------------------------------------------------
ms = lambda v: f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"
for ds in DS:
    print(f"\n################ {ds}")
    print(D[D.dataset == ds].groupby("model")[["accuracy", "f1", "mcc", "roc_auc"]].agg(ms).reindex(MODELS).to_string())
    c = C[C.dataset == ds].groupby(["model", "calibration"])[["ECE", "Brier"]].agg(ms).unstack().reindex(MODELS)
    print(c.to_string())
    s = S[S.dataset == ds]
    a = s.groupby(["model", "target_defer"]).acc_auto.mean().unstack().reindex(MODELS); print("acc_auto\n", a.round(4).to_string())
    print("defer\n", s.groupby(["model", "target_defer"]).defer_rate.mean().unstack().reindex(MODELS).round(4).to_string())
    print("fpr_auto\n", s.groupby(["model", "target_defer"]).fpr_auto.mean().unstack().reindex(MODELS).round(4).to_string())
    print("PSO detail\n", s[s.model == PROP].groupby("target_defer")[["defer_rate", "acc_auto", "precision_auto", "recall_auto",
          "f1_auto", "fpr_auto", "fnr_auto", "errors_caught_pct", "errors_auto_per_10k"]].agg(["mean", "std"]).round(4).to_string())
    print("conformal\n", M[M.dataset == ds].groupby(["model", "alpha"])[["coverage_malware", "coverage_benign", "singleton_rate",
          "both_rate", "empty_rate", "acc_singleton"]].mean().reindex(MODELS, level=0).round(4).to_string())
    print(P[P.dataset == ds].round(4).to_string(index=False))
pd.set_option("display.width", 250)
print("\n", T.assign(diff=lambda d: d["diff"].round(5), t=lambda d: d.t.round(3), p=lambda d: d.p.map("{:.2e}".format),
                     p_holm=lambda d: d.p_holm.map("{:.2e}".format)).to_string(index=False))

# --- figures -----------------------------------------------------------------------------------------------------
COL = {"PSO-XGBoost": "#c0392b", "XGBoost": "#2471a3", "LightGBM": "#76b7e5", "AdaBoost": "#a6a6a6", "GBM": "#4d4d4d"}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.9))
for i, ds in enumerate(DS):
    for meth, col, lab in (("raw", "#a6a6a6", "Uncalibrated"), ("platt", "#2471a3", "Platt scaling"), ("isotonic", "#e67e22", "Isotonic regression")):
        y = np.concatenate([a for a, _ in pool[(ds, PROP, meth)]]); q = np.concatenate([b for _, b in pool[(ds, PROP, meth)]])
        k = np.minimum((q * 15).astype(int), 14); ok = [j for j in range(15) if (k == j).sum() >= 20]
        e = C[(C.dataset == ds) & (C.model == PROP) & (C.calibration == meth)].ECE.mean()
        ax[i].plot([q[k == j].mean() for j in ok], [y[k == j].mean() for j in ok], "o-", ms=3, color=col, label=f"{lab} (ECE = {e:.3f})")
    ax[i].plot([0, 1], [0, 1], "k--", lw=.8); ax[i].set_xlim(0, 1); ax[i].set_ylim(0, 1)
    ax[i].set_xlabel("Predicted malware probability"); ax[i].set_ylabel("Observed fraction of malware flows")
    ax[i].set_title(f"({'ab'[i]}) {ds}", fontsize=9); ax[i].legend(fontsize=7, loc="upper left"); ax[i].grid(alpha=.25)
plt.tight_layout(); plt.savefig(FIG / "fig2_reliability.png", dpi=300); plt.close("all")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.9))
for i, ds in enumerate(DS):
    s = S[S.dataset == ds]
    for m in MODELS:
        g = s[s.model == m].groupby("target_defer")[["defer_rate", "acc_auto"]].mean()
        ax[i].plot(100 * (1 - g.defer_rate), 100 * g.acc_auto, "o-", ms=3.5, color=COL[m], lw=2.2 if m == PROP else 1.1, label=m)
    ax[i].invert_xaxis(); ax[i].grid(alpha=.25); ax[i].set_xlabel("Flows decided automatically (%)")
    ax[i].set_ylabel("Accuracy on automatic decisions (%)"); ax[i].set_title(f"({'ab'[i]}) {ds}", fontsize=9); ax[i].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "fig3_accuracy_coverage.png", dpi=300); plt.close("all")
print("figures saved")
