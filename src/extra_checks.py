"""Revision checks from saved predictions (no retraining), CICAndMal2017, app protocol, main features, 15 folds.
 (1) capture-day check: app-level ROC-AUC restricted to test apps captured on days that contain both classes
 (2) alternative aggregations of flow probabilities into an application score
 (3) decision-layer summary: cost and ECE effect sizes with corrected CIs"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, re, numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
from scipy import stats
sys.path.insert(0, ROOT + "/src")
D = pd.read_parquet(ROOT + "/data/cicandmal_app_K20.parquet", columns=["app", "date", "y"])
APP, DATE, Y = D.app.to_numpy(), D.date.to_numpy(), D.y.to_numpy()
t = D.groupby(["date", "y"]).app.nunique().unstack(fill_value=0); SHARED = set(t[(t[0] > 0) & (t[1] > 0)].index)
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
AGGS = {"mean": np.mean, "median": np.median, "max": np.max, "logit-mean": lambda p: np.mean(logit(p)),
        "share > 0.5": lambda p: np.mean(p > .5), "75th percentile": lambda p: np.percentile(p, 75)}
cap, agg = [], []
for fn in sorted(glob.glob(ROOT + "/predictions/andmal_app_main_r*_f*_full.npz")):
    rep, fold = map(int, re.search(r"_r(\d)_f(\d)", fn).groups()); z = np.load(fn, allow_pickle=True); idx = z["idx_te"]
    for m in ("PSO-XGBoost", "XGBoost", "PSO-LightGBM", "PSO-RF"):
        df = pd.DataFrame({"app": APP[idx], "y": Y[idx], "date": DATE[idx], "p": z[f"te__{m}"]})
        g = df.groupby("app").agg(y=("y", "first"), date=("date", "first"), s=("p", "mean"))
        sh = g[g.date.isin(SHARED)]
        cap.append(dict(model=m, repeat=rep, fold=fold, auc_all=roc_auc_score(g.y, g.s), n_shared=len(sh), mal_shared=int(sh.y.sum()),
                        auc_shared=roc_auc_score(sh.y, sh.s) if 0 < sh.y.sum() < len(sh) else np.nan,
                        y=sh.y.tolist(), s=sh.s.tolist()))
        for a, f in AGGS.items():
            ga = df.groupby("app").agg(y=("y", "first")).join(df.groupby("app").p.apply(lambda v: f(v.to_numpy())).rename("s"))
            agg.append(dict(model=m, repeat=rep, fold=fold, aggregation=a, auc=roc_auc_score(ga.y, ga.s)))
cap, agg = pd.DataFrame(cap), pd.DataFrame(agg)
def ctest(d, rho=.25):
    d = np.asarray(d, float); n = len(d); m, v = d.mean(), d.var(ddof=1); se = np.sqrt((1 / n + rho) * v)
    tt = m / se; h = stats.t.ppf(.975, n - 1) * se; return dict(diff=m, lo=m - h, hi=m + h, p=2 * stats.t.sf(abs(tt), n - 1))
rows = []
for m, g in cap.groupby("model"):
    # pooled per repetition (each shared-day app tested once per repetition), bootstrap CI over apps
    pooled = []
    for r, gr in g.groupby("repeat"):
        y = np.concatenate(gr.y.values); s = np.concatenate(gr.s.values); pooled.append((r, roc_auc_score(y, s), len(y), int(y.sum()), y, s))
    rng = np.random.RandomState(0); y, s = pooled[0][4], pooled[0][5]; bs = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): bs.append(roc_auc_score(y[i], s[i]))
    rows.append(dict(model=m, auc_all_mean=g.auc_all.mean(), auc_shared_pooled_mean=np.mean([p[1] for p in pooled]),
                     auc_shared_pooled_r0=pooled[0][1], ci_lo_r0=np.percentile(bs, 2.5), ci_hi_r0=np.percentile(bs, 97.5),
                     n_apps=pooled[0][2], n_mal=pooled[0][3], auc_shared_fold_mean=g.auc_shared.mean()))
CAP = pd.DataFrame(rows); CAP.to_csv(ROOT + "/results/CAPDAY.csv", index=False); print(CAP.round(3).to_string())
AG = agg.groupby(["model", "aggregation"]).auc.agg(["mean", "std"]).reset_index(); agg.to_csv(ROOT + "/results/AGG_folds.csv", index=False)
tests = []
for m in ("PSO-XGBoost",):
    pv = agg[agg.model == m].pivot_table(index=["repeat", "fold"], columns="aggregation", values="auc")
    for a in AGGS:
        if a != "mean": tests.append(dict(model=m, aggregation=a, **ctest(pv[a] - pv["mean"])))
T = pd.DataFrame(tests); from statsmodels.stats.multitest import multipletests; T["p_holm"] = multipletests(T.p, method="holm")[1]
AG.merge(T, on=["model", "aggregation"], how="left").to_csv(ROOT + "/results/AGG.csv", index=False)
print(AG[AG.model == "PSO-XGBoost"].round(3).to_string()); print(T.round(4).to_string())
