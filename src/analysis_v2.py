"""Revision analysis (stage 2) from the per-fold probabilities saved by train_v2.py. Nothing here touches TEST labels
except for reporting. All calibrators, thresholds and conformal scores are fitted on a calibration split.

Blocks
  D  discrimination on TEST: ROC-AUC, PR-AUC, and F1 / MCC / balanced accuracy at the F1-optimal threshold chosen on CAL
  C  calibration (Platt, isotonic) fitted on held-out-app CAL vs seen-app CAL (app protocol): ECE, Brier on TEST
  S  selective classification: error-targeted window (ours) vs symmetric Chow band around the decision threshold;
     both set on CAL for target workloads r; automatic-decision error, FPR, FNR, errors intercepted; AURC of the
     symmetric confidence |q - 0.5|; end-to-end error with analyst accuracy a; cost per 10k flows
  P  prevalence: TEST reweighted to target malware prevalence pi' with prior-shift-adjusted probabilities
  M  Mondrian conformal (held-out-app vs seen-app CAL): per-class coverage with Clopper-Pearson CIs, singleton rate
  T  corrected repeated k-fold t-tests (Nadeau & Bengio 2003; Bouckaert & Frank 2004) with Holm adjustment
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, json, re, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, matthews_corrcoef, balanced_accuracy_score
from pathlib import Path
PRED = Path(ROOT + "/predictions"); OUT = Path(ROOT + "/results"); OUT.mkdir(exist_ok=True)
EPS = 1e-6; RATES = [0, .05, .10, .20, .30]; ALPHAS = [.05, .10]; PRIORS = [.10, .05, .01, .001]

from analysis_lib import logit, platt, iso, ece, f1_thr, window_error, window_sym

def selective(qt, yt, thr, win, a_analyst=1.0, cfn=10, cfp=1, crev=.5):
    defer = np.zeros(len(qt), bool) if win is None else (qt >= win[0]) & (qt <= win[1])
    yh = (qt >= thr).astype(int); k = ~defer; yk, pk = yt[k], yh[k]
    fp = ((pk == 1) & (yk == 0)).sum(); fn = ((pk == 0) & (yk == 1)).sum(); tp = ((pk == 1) & (yk == 1)).sum(); tn = ((pk == 0) & (yk == 0)).sum()
    e_all = yh != yt; n = len(yt); nd = defer.sum()
    err_e2e = (fp + fn + (1 - a_analyst) * nd) / n
    return {"defer": nd / n, "err_auto": (fp + fn) / max(k.sum(), 1), "fpr_auto": fp / max(fp + tn, 1), "fnr_auto": fn / max(fn + tp, 1),
            "prec_auto": tp / max(tp + fp, 1), "intercepted_pct": 100 * e_all[defer].sum() / max(e_all.sum(), 1),
            "fp_per10k": 1e4 * fp / n, "fn_per10k": 1e4 * fn / n, "e2e_err_a1": (fp + fn) / n,
            "e2e_err_a95": (fp + fn + .05 * nd) / n, "e2e_err_a90": (fp + fn + .10 * nd) / n,
            "cost_per10k": 1e4 * (cfn * fn + cfp * fp + crev * nd) / n}

def aurc(q, y, thr):
    conf = np.abs(q - thr); o = np.argsort(-conf); err = ((q[o] >= thr).astype(int) != y[o]).astype(float)
    return np.mean(np.cumsum(err) / np.arange(1, len(err) + 1))

def mondrian(pc, yc, pt, yt, a):
    sets = np.zeros((len(pt), 2), bool)
    for c in (0, 1):
        sc = 1 - (pc if c else 1 - pc)[yc == c]; st = 1 - (pt if c else 1 - pt)
        sets[:, c] = (len(sc) - np.searchsorted(np.sort(sc), st, side="left") + 1) / (len(sc) + 1) > a
    single = sets.sum(1) == 1
    return {"cov_mal_hits": int(sets[yt == 1, 1].sum()), "n_mal": int((yt == 1).sum()), "cov_ben_hits": int(sets[yt == 0, 0].sum()),
            "n_ben": int((yt == 0).sum()), "singleton": single.mean(), "err_singleton": (sets[single, 1].astype(int) != yt[single]).mean() if single.any() else np.nan,
            "defer": 1 - single.mean()}

def prior_shift(q, pi_from, pi_to):
    a = q * pi_to / pi_from; b = (1 - q) * (1 - pi_to) / (1 - pi_from); return a / (a + b)

rows = {k: [] for k in "DCSPMV"}
for fn in sorted(PRED.glob("*.npz")):
    ds, proto, feats, rep, fold, mset = re.match(r"(\w+?)_(app|flow)_(\w+?)_r(\d)_f(\d)_(\w+)\.npz", fn.name).groups()
    z = np.load(fn, allow_pickle=True); key = dict(dataset=ds, protocol=proto, features=feats, repeat=int(rep), fold=int(fold), modelset=mset)
    yt, yc = z["y_te"], z["y_cal"]; has_seen = "y_calseen" in z
    info = json.loads(str(z["info"]))
    for m in z["models"]:
        pt, pc = z[f"te__{m}"], z[f"cal__{m}"]
        meta = dict(key, model=m, n_train=int(z["n_train"]), n_test=len(yt))
        # D: discrimination; threshold chosen on CAL (held-out apps)
        t = f1_thr(pc, yc); yh = pt >= t
        rows["D"].append(dict(meta, roc_auc=roc_auc_score(yt, pt), pr_auc=average_precision_score(yt, pt),
                              f1=f1_score(yt, yh), mcc=matthews_corrcoef(yt, yh), bal_acc=balanced_accuracy_score(yt, yh),
                              prevalence=yt.mean(), inner_auc=info.get(m, {}).get("inner_auc", np.nan)))
        cal_sets = [("heldout", pc, yc)] + ([("seen", z[f"calseen__{m}"], z["y_calseen"])] if has_seen else [])
        for src, pcs, ycs in cal_sets:
            fP, fI = platt(pcs, ycs), iso(pcs, ycs)
            for meth, q in (("raw", pt), ("platt", fP(pt)), ("isotonic", fI(pt))):
                rows["C"].append(dict(meta, calsource=src, method=meth, ece=ece(yt, q), brier=brier_score_loss(yt, q),
                                      mean_q=q.mean(), prevalence=yt.mean()))
            qc, qt = fP(pcs), fP(pt); thr = f1_thr(qc, ycs)
            for rule in ("error_window", "cost_window", "symmetric"):
                for r in RATES:
                    win = (window_error(qc, ycs, thr, r) if rule == "error_window" else window_error(qc, ycs, thr, r, cfn=10) if rule == "cost_window" else window_sym(qc, thr, r))
                    rows["S"].append(dict(meta, calsource=src, rule=rule, target=r, threshold=thr, tau_lo=win[0] if win else np.nan, tau_hi=win[1] if win else np.nan, **selective(qt, yt, thr, win)))
            if src == "heldout":
                rows["S"].append(dict(meta, calsource=src, rule="aurc", target=np.nan, aurc=aurc(qt, yt, thr)))
            for a in ALPHAS:
                rows["M"].append(dict(meta, calsource=src, alpha=a, **mondrian(pcs, ycs, pt, yt, a)))
            # P: prevalence shift (held-out-app calibration only)
            if src == "heldout":
                pi = ycs.mean()
                for pi2 in PRIORS:
                    w = np.where(yt == 1, pi2 / yt.mean(), (1 - pi2) / (1 - yt.mean())); q2 = prior_shift(qt, pi, pi2)
                    for thr_name, thr in (("0.5", .5), ("cost10", 1 / 11)):
                        yh2 = q2 >= thr; tp = (w * (yh2 & (yt == 1))).sum(); fp = (w * (yh2 & (yt == 0))).sum()
                        fn_ = (w * (~yh2 & (yt == 1))).sum(); W = w.sum()
                        rows["P"].append(dict(meta, prior=pi2, threshold=thr_name, precision=tp / max(tp + fp, 1e-12),
                                              recall=tp / max(tp + fn_, 1e-12), alerts_per10k=1e4 * (tp + fp) / W,
                                              fp_per10k=1e4 * fp / W, missed_per10k=1e4 * fn_ / W,
                                              pr_auc=average_precision_score(yt, qt, sample_weight=w), ece=ece(yt, q2, w=w)))
                    # matched workload r = 20%: share of benign among deferred flows at prior pi2
                    win = window_sym(qc, thr, .20); dmask = (qt >= win[0]) & (qt <= win[1])
                    rows["P"].append(dict(meta, prior=pi2, threshold="defer20_sym", benign_share_deferred=
                                          (w * dmask * (yt == 0)).sum() / max((w * dmask).sum(), 1e-12),
                                          malware_deferred_per10k=1e4 * (w * dmask * (yt == 1)).sum() / w.sum()))
        # V: conformal vs reject at matched deferral (per fold, held-out CAL, alpha = 0.05)
        mc = mondrian(pc, yc, pt, yt, .05); fP = platt(pc, yc); qc, qt = fP(pc), fP(pt); thr = f1_thr(qc, yc)
        r = mc["defer"]
        if 0 < r < 1:
            for rule in ("error_window", "cost_window", "symmetric"):
                win = (window_error(qc, yc, thr, r) if rule == "error_window" else window_error(qc, yc, thr, r, cfn=10)
                       if rule == "cost_window" else window_sym(qc, thr, r))
                s = selective(qt, yt, thr, win); rows["V"].append(dict(meta, rule=rule, defer=s["defer"], err_auto=s["err_auto"]))
            rows["V"].append(dict(meta, rule="conformal", defer=mc["defer"], err_auto=mc["err_singleton"]))
for k, v in rows.items():
    pd.DataFrame(v).to_csv(OUT / f"{k}.csv", index=False)
print({k: len(v) for k, v in rows.items()})
