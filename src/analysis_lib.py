"""Shared functions for the revision analyses."""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
EPS = 1e-6
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def platt(pc, yc):
    m = LogisticRegression(C=1e6).fit(logit(pc)[:, None], yc); return lambda p: m.predict_proba(logit(p)[:, None])[:, 1]
def iso(pc, yc): return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pc, yc).predict
def ece(y, p, nb=15, w=None):
    w = np.ones_like(p) if w is None else w; b = np.minimum((p * nb).astype(int), nb - 1); W = w.sum()
    return sum(abs(np.average(y[b == k], weights=w[b == k]) - np.average(p[b == k], weights=w[b == k])) * w[b == k].sum() / W
               for k in range(nb) if (b == k).any())
def f1_thr(pc, yc):
    ts = np.unique(np.quantile(pc, np.linspace(.01, .99, 197)))
    return ts[np.argmax([f1_score(yc, pc >= t) for t in ts])]

def window_error(qc, yc, thr, r, cfn=1, cfp=1):
    """Error-targeted deferral window: among windows of round(r n) consecutive CAL items (sorted by q), the one with
    the largest total CAL error cost of the rule 1[q >= thr] (a false negative costs cfn, a false positive cfp;
    cfn = cfp = 1 counts errors); ties -> centre closest to thr."""
    n = len(qc); w = int(round(r * n))
    if w == 0: return None
    o = np.argsort(qc); qs = qc[o]; yh = (qs >= thr).astype(int); yo = yc[o]
    err = np.where((yh == 0) & (yo == 1), cfn, 0) + np.where((yh == 1) & (yo == 0), cfp, 0)
    cs = np.concatenate([[0], np.cumsum(err)]); inside = cs[w:] - cs[:-w]
    mid = np.abs((qs[:n - w + 1] + qs[w - 1:]) / 2 - thr); s = np.lexsort((mid, -inside))[0]
    return qs[s], qs[s + w - 1]
def window_sym(qc, thr, r):
    """Symmetric (Chow-type) band: defer the round(r n) CAL flows closest to the decision threshold."""
    if r == 0: return None
    d = np.abs(qc - thr); h = np.quantile(d, r); return thr - h, thr + h

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

