"""Prevalence analysis (same procedure for both datasets). For each outer fold: Platt calibration on CAL; the TEST and
CAL items are importance-weighted to a target malware prevalence pi'; probabilities are prior-shift adjusted
(Saerens et al. 2002) from the CAL prevalence pi to pi'. Two operating thresholds:
  fixed  : the F1-optimal threshold chosen on CAL at the sampled prevalence (mapped through the prior shift, so the
           same items are flagged and recall is unchanged)
  reopt  : the threshold that maximises the importance-weighted F1 on CAL at prevalence pi'
Reported on TEST (weighted): precision, recall, alerts / false alarms / missed malware per 1000 items, PR-AUC,
ECE and Brier score of the prior-shifted probabilities.
CICAndMal2017: app level (app score = mean flow probability), 15 folds. CIC-AAGM2017: flow level, 5 folds."""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, sys, numpy as np, pandas as pd
sys.path.insert(0, ROOT + "/src")
from analysis_lib import platt, f1_thr, prior_shift, ece
from sklearn.metrics import average_precision_score
PRED = ROOT + "/predictions/"
AP = pd.read_parquet(ROOT + "/data/cicandmal_app_K20.parquet", columns=["app", "y"]); APP = AP.app.to_numpy(); YA = AP.y.to_numpy()
def agg(idx, p):
    g = pd.DataFrame({"app": APP[idx], "y": YA[idx], "p": p}).groupby("app").agg(y=("y", "first"), s=("p", "mean"))
    return g.y.to_numpy(), g.s.to_numpy()
def wf1_thr(q, y, w):
    best, bt = -1, .5
    for t in np.unique(np.quantile(q, np.linspace(.005, .995, 397))):
        h = q >= t; tp = (w * (h & (y == 1))).sum(); fp = (w * (h & (y == 0))).sum(); fn = (w * (~h & (y == 1))).sum()
        f = 2 * tp / max(2 * tp + fp + fn, 1e-12)
        if f > best: best, bt = f, t
    return bt
rows = []
for ds, pat, priors in (("CICAndMal2017", "andmal_app_main_r*_f*_full.npz", [None, .10, .05, .01]),
                        ("CIC-AAGM2017", "aagm_flow_main_r0_f*_full.npz", [None, .10, .05, .01, .001])):
    for fn in sorted(glob.glob(PRED + pat)):
        z = np.load(fn, allow_pickle=True)
        for m in ("PSO-XGBoost", "XGBoost"):
            if ds == "CICAndMal2017": yt, st = agg(z["idx_te"], z[f"te__{m}"]); yc, sc = agg(z["idx_cal"], z[f"cal__{m}"])
            else: yt, st, yc, sc = z["y_te"], z[f"te__{m}"], z["y_cal"], z[f"cal__{m}"]
            f = platt(sc, yc); qc, qt = f(sc), f(st); pi = yc.mean(); t0 = f1_thr(qc, yc)
            for p2 in priors:
                pt = yt.mean() if p2 is None else p2
                wt = np.ones_like(qt) if p2 is None else np.where(yt == 1, p2 / yt.mean(), (1 - p2) / (1 - yt.mean()))
                wc = np.ones_like(qc) if p2 is None else np.where(yc == 1, p2 / pi, (1 - p2) / (1 - pi))
                q2t = qt if p2 is None else prior_shift(qt, pi, p2); q2c = qc if p2 is None else prior_shift(qc, pi, p2)
                tf = t0 if p2 is None else prior_shift(np.array([t0]), pi, p2)[0]
                for tname, t in (("fixed", tf), ("reopt", wf1_thr(q2c, yc, wc))):
                    h = q2t >= t; W = wt.sum()
                    tp = (wt * (h & (yt == 1))).sum(); fp = (wt * (h & (yt == 0))).sum(); fn_ = (wt * (~h & (yt == 1))).sum()
                    rows.append(dict(dataset=ds, model=m, file=fn.split("/")[-1], prior="sampled" if p2 is None else p2, prior_value=pt,
                                     threshold=tname, precision=tp / max(tp + fp, 1e-12), recall=tp / max(tp + fn_, 1e-12),
                                     alerts_per1000=1e3 * (tp + fp) / W, fp_per1000=1e3 * fp / W, missed_per1000=1e3 * fn_ / W,
                                     pr_auc=average_precision_score(yt, qt, sample_weight=wt), ece=ece(yt, q2t, w=wt),
                                     brier=(wt * (q2t - yt) ** 2).sum() / W))
pd.DataFrame(rows).to_csv(ROOT + "/results/PREV.csv", index=False)
d = pd.DataFrame(rows); print(d[d.model == "PSO-XGBoost"].groupby(["dataset", "prior", "threshold"])[["precision", "recall", "alerts_per1000", "fp_per1000", "missed_per1000", "pr_auc", "ece", "brier"]].mean().round(3).to_string())
