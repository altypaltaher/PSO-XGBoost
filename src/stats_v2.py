"""Block T: corrected repeated k-fold t-tests (Nadeau & Bengio 2003; Bouckaert & Frank 2004) with Holm adjustment.
t = mean(d) / sqrt((1/n + n_te/n_tr) * var(d)), df = n - 1, n = repeats x folds; 95% CI with the same corrected SE.
Comparisons (app protocol, main features, full model set):
  T1  every model vs PSO-XGBoost: app ROC-AUC, flow ROC-AUC, app F1, app MCC
  T2  error-targeted window vs symmetric band, app level, target workloads 10/20/30%: error of automatic decisions
  T3  held-out-app vs seen-app calibration, flow level (Platt): ECE
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
R = ROOT + "/results/"; RHO = 0.25   # test/train size ratio (1 of 5 app folds vs 4 of 5)

def ctest(d):
    d = np.asarray(d, float); n = len(d); m, v = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / n + RHO) * v) if v > 0 else np.nan
    t = m / se if se and se > 0 else np.nan; p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 1.0
    h = stats.t.ppf(.975, n - 1) * se if np.isfinite(se) else np.nan
    return dict(n=n, mean_diff=m, ci_lo=m - h, ci_hi=m + h, t=t, p=p)

def holm(df):
    df["p_holm"] = multipletests(df.p.fillna(1), method="holm")[1]; return df

out = []
A = pd.read_csv(R + "A.csv"); A = A[(A.features == "main") & (A.modelset == "full")]
for metric in ("roc_auc", "flow_roc_auc", "f1", "mcc"):
    piv = A.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    rows = []
    for m in piv.columns:
        if m == "PSO-XGBoost": continue
        d = (piv[m] - piv["PSO-XGBoost"]).dropna()
        rows.append(dict(block="T1", metric=metric, comparison=f"{m} - PSO-XGBoost", **ctest(d)))
    out.append(holm(pd.DataFrame(rows)))
S = pd.read_csv(R + "AS.csv"); S = S[S.rule.isin(["error_window", "cost_window", "symmetric"])]
rows = []
for model in ("PSO-XGBoost", "XGBoost"):
    for r in (.10, .20, .30):
        s = S[(S.model == model) & (S.target == r)].pivot_table(index=["repeat", "fold"], columns="rule", values="err_auto")
        rows.append(dict(block="T2", metric=f"err_auto@{int(r*100)}% ({model})", comparison="error_window - symmetric",
                         **ctest(s.error_window - s.symmetric)))
out.append(holm(pd.DataFrame(rows)))
rows = []
for model in ("PSO-XGBoost", "XGBoost"):
    for r in (.05, .10, .20, .30):
        s = S[(S.model == model) & (S.target == r)].pivot_table(index=["repeat", "fold"], columns="rule", values="cost_per10k")
        rows.append(dict(block="T2c", metric=f"cost_per10k@{int(r*100)}% ({model})", comparison="cost_window - symmetric",
                         **ctest(s.cost_window - s.symmetric)))
out.append(holm(pd.DataFrame(rows)))
try:
    C = pd.read_csv(R + "C.csv")
    C = C[(C.dataset == "andmal") & (C.protocol == "app") & (C.features == "main") & (C.method == "platt")]
    rows = []
    for model in ("PSO-XGBoost", "XGBoost"):
        c = C[C.model == model].pivot_table(index=["repeat", "fold"], columns="calsource", values="ece")
        rows.append(dict(block="T3", metric=f"ECE ({model})", comparison="heldout - seen", **ctest(c.heldout - c.seen)))
    out.append(holm(pd.DataFrame(rows)))
except FileNotFoundError:
    pass
T = pd.concat(out, ignore_index=True); T.to_csv(R + "T.csv", index=False)
pd.set_option("display.width", 200); print(T.round(4).to_string())
