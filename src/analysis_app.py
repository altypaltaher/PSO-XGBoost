"""App-level triage analysis (CICAndMal2017, app protocol). Flow probabilities from train_v2.py are aggregated per app;
calibration, deferral thresholds and conformal scores are fitted on the held-out CAL apps (whole apps never used for
training), and everything is reported on the unseen TEST apps.

App score s(app) = mean of the flow malware probabilities of the app's observed flows (k flows, k <= 20).
Blocks
  A   app-level discrimination: ROC-AUC, PR-AUC, F1/MCC at the CAL-chosen F1-optimal threshold, for all models
  K   evidence curve: app ROC-AUC when only k randomly chosen flows per app are observed, k in {1,2,3,5,10,20}
  AC  calibration of the app score (Platt, isotonic fitted on CAL apps): ECE, Brier on TEST apps
  AS  reject option at app level (share of apps sent to an analyst): error-targeted window vs symmetric band;
      error of automatic decisions, FPR, FNR, errors intercepted, end-to-end error with analyst accuracy a, AURC
  AP  prevalence: TEST apps reweighted to 10/5/1% malicious apps with prior-shift-adjusted probabilities
  AM  Mondrian conformal at app level: per-class coverage (Clopper-Pearson CI on pooled counts), singleton rate
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import re, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, matthews_corrcoef
from analysis_lib import platt, iso, ece, f1_thr, window_error, window_sym, selective, aurc, mondrian, prior_shift
PRED = Path(ROOT + "/predictions"); OUT = Path(ROOT + "/results"); OUT.mkdir(exist_ok=True)
D = pd.read_parquet(ROOT + "/data/cicandmal_app_K20.parquet", columns=["app", "family", "category", "y"])
APP, FAM, CAT = D.app.to_numpy(), D.family.to_numpy(), D.category.to_numpy()
RATES = [0, .05, .10, .20, .30]; KS = [1, 2, 3, 5, 10, 20]

def agg(idx, p, k=None, seed=0):
    df = pd.DataFrame({"app": APP[idx], "y": D.y.to_numpy()[idx], "p": p, "fam": FAM[idx], "cat": CAT[idx]})
    if k is not None:
        df = df.sample(frac=1, random_state=seed); df = df[df.groupby("app").cumcount() < k]
    g = df.groupby("app").agg(y=("y", "first"), s=("p", "mean"), n=("p", "size"), fam=("fam", "first"), cat=("cat", "first"))
    return g

rows = {k: [] for k in ("A", "K", "AC", "AS", "AP", "AM", "AF", "AV")}
for fn in sorted(PRED.glob("andmal_app_*.npz")):
    ds, proto, feats, rep, fold, mset = re.match(r"(\w+?)_(app|flow)_(\w+?)_r(\d)_f(\d)_(\w+)\.npz", fn.name).groups()
    z = np.load(fn, allow_pickle=True); key = dict(features=feats, repeat=int(rep), fold=int(fold), modelset=mset)
    for m in z["models"]:
        gt, gc = agg(z["idx_te"], z[f"te__{m}"]), agg(z["idx_cal"], z[f"cal__{m}"]); meta = dict(key, model=m, n_apps_test=len(gt))
        yt, st, yc, sc = gt.y.to_numpy(), gt.s.to_numpy(), gc.y.to_numpy(), gc.s.to_numpy()
        fP = platt(sc, yc); qt, qc = fP(st), fP(sc); thr = f1_thr(qc, yc); yh = qt >= thr
        rows["A"].append(dict(meta, roc_auc=roc_auc_score(yt, st), pr_auc=average_precision_score(yt, st), f1=f1_score(yt, yh),
                              mcc=matthews_corrcoef(yt, yh), threshold=thr, prevalence=yt.mean(),
                              flow_roc_auc=roc_auc_score(z["y_te"], z[f"te__{m}"])))
        for k in KS:
            aucs = [roc_auc_score(*(lambda g: (g.y, g.s))(agg(z["idx_te"], z[f"te__{m}"], k, sd))) for sd in range(5)]
            rows["K"].append(dict(meta, k=k, roc_auc=np.mean(aucs)))
        if feats != "main" or mset != "full": continue
        fI = iso(sc, yc)
        for meth, q in (("raw", st), ("platt", qt), ("isotonic", fI(st))):
            rows["AC"].append(dict(meta, method=meth, ece=ece(yt, q), brier=brier_score_loss(yt, q)))
        for rule in ("error_window", "cost_window", "symmetric"):
            for r in RATES:
                win = (window_error(qc, yc, thr, r) if rule == "error_window" else window_error(qc, yc, thr, r, cfn=10) if rule == "cost_window" else window_sym(qc, thr, r))
                rows["AS"].append(dict(meta, rule=rule, target=r, threshold=thr, tau_lo=win[0] if win else np.nan, tau_hi=win[1] if win else np.nan, **selective(qt, yt, thr, win)))
        rows["AS"].append(dict(meta, rule="aurc", aurc=aurc(qt, yt, thr)))
        for pi2 in (.10, .05, .01):
            w = np.where(yt == 1, pi2 / yt.mean(), (1 - pi2) / (1 - yt.mean())); q2 = prior_shift(qt, yc.mean(), pi2)
            t2 = f1_thr(prior_shift(qc, yc.mean(), pi2), yc)       # threshold re-chosen on CAL after the prior shift
            yh2 = q2 >= t2; tp = (w * (yh2 & (yt == 1))).sum(); fp = (w * (yh2 & (yt == 0))).sum(); fn_ = (w * (~yh2 & (yt == 1))).sum()
            rows["AP"].append(dict(meta, prior=pi2, precision=tp / max(tp + fp, 1e-12), recall=tp / max(tp + fn_, 1e-12),
                                   alerts_per1000=1e3 * (tp + fp) / w.sum(), fp_per1000=1e3 * fp / w.sum(),
                                   pr_auc=average_precision_score(yt, st, sample_weight=w), ece=ece(yt, q2, w=w)))
        for a in (.05, .10):
            rows["AM"].append(dict(meta, alpha=a, **mondrian(sc, yc, st, yt, a)))
        mc = mondrian(sc, yc, st, yt, .05); rr = mc["defer"]           # AV: reject rules at the conformal deferral rate
        if 0 < rr < 1:
            for rule in ("error_window", "cost_window", "symmetric"):
                win = (window_error(qc, yc, thr, rr) if rule == "error_window" else window_error(qc, yc, thr, rr, cfn=10)
                       if rule == "cost_window" else window_sym(qc, thr, rr))
                rows["AV"].append(dict(meta, rule=rule, **selective(qt, yt, thr, win)))
            rows["AV"].append(dict(meta, rule="conformal", defer=mc["defer"], err_auto=mc["err_singleton"]))
        if m == "PSO-XGBoost":
            gt2 = gt.assign(q=qt, flagged=yh)
            for (cat, fam), g in gt2[gt2.y == 1].groupby(["cat", "fam"]):
                rows["AF"].append(dict(meta, category=cat, family=fam, n_apps=len(g), detected=int(g.flagged.sum())))
for k, v in rows.items(): pd.DataFrame(v).to_csv(OUT / f"{k}.csv", index=False)
print({k: len(v) for k, v in rows.items()})
