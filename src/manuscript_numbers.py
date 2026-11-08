"""All numbers used in the revised manuscript, computed from the result CSVs and saved predictions (single source)."""
import json, glob, numpy as np, pandas as pd
from scipy.stats import beta
import os
ROOT = os.environ.get("TRIAGE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = ROOT + "/results/"; PRED = ROOT + "/predictions/"
MODELS = ["PSO-XGBoost", "RS-XGBoost", "TPE-XGBoost", "XGBoost", "PSO-LightGBM", "LightGBM", "PSO-RF", "PSO-ExtraTrees", "GBM", "AdaBoost"]
A = pd.read_csv(R + "A.csv"); K = pd.read_csv(R + "K.csv"); AC = pd.read_csv(R + "AC.csv"); AS = pd.read_csv(R + "AS.csv")
AM = pd.read_csv(R + "AM.csv"); AF = pd.read_csv(R + "AF.csv"); AV = pd.read_csv(R + "AV.csv")
D = pd.read_csv(R + "D.csv"); C = pd.read_csv(R + "C.csv"); S = pd.read_csv(R + "S.csv"); M = pd.read_csv(R + "M.csv")
V = pd.read_csv(R + "V.csv"); PREV = pd.read_csv(R + "PREV.csv")
AM_ = A[(A.features == "main") & (A.modelset == "full")]                      # app protocol, main features, 15 folds
DG = D[D.dataset == "aagm"]                                                   # CIC-AAGM2017 flow level, 5 folds
RHO = 0.25

def cp(k, n, a=.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

def ctest(d):
    from scipy import stats
    d = np.asarray(d, float); n = len(d); m, v = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / n + RHO) * v); t = m / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else 1.0; h = stats.t.ppf(.975, n - 1) * se
    return dict(n=n, diff=m, lo=m - h, hi=m + h, t=t, p=p, wins=int((d > 0).sum()), losses=int((d < 0).sum()))

def holm(ps):
    from statsmodels.stats.multitest import multipletests
    return multipletests(np.nan_to_num(np.asarray(ps, float), nan=1.0), method="holm")[1]

def tests(df, metric, idx, ref="PSO-XGBoost", models=None):
    piv = df.pivot_table(index=idx, columns="model", values=metric); rows = []
    for m in (models or [c for c in MODELS if c in piv.columns and c != ref]):
        d = (piv[m] - piv[ref]).dropna(); rows.append(dict(model=m, metric=metric, **ctest(d)))
    out = pd.DataFrame(rows); out["p_holm"] = holm(out.p); return out

N = {}
# ---------------------------------------------------------------- discrimination
def ms(df, m, col): x = df[df.model == m][col]; return x.mean(), x.std(ddof=1), len(x)
N["andmal_disc"] = {m: {c: ms(AM_, m, c) for c in ("flow_roc_auc", "roc_auc", "pr_auc", "f1", "mcc")} for m in MODELS}
N["aagm_disc"] = {m: {c: ms(DG, m, c) for c in ("roc_auc", "pr_auc", "f1", "mcc")} for m in MODELS}
N["andmal_prev_apps"] = AM_[AM_.model == "PSO-XGBoost"].prevalence.mean()
N["aagm_prev"] = DG[DG.model == "PSO-XGBoost"].prevalence.mean()
T_and = pd.concat([tests(AM_, c, ["repeat", "fold"]) for c in ("roc_auc", "flow_roc_auc", "pr_auc", "f1", "mcc")])
T_agm = pd.concat([tests(DG, c, ["fold"]) for c in ("roc_auc", "pr_auc", "f1", "mcc")])
N["T_and"], N["T_agm"] = T_and, T_agm
# tuner comparisons on the same 5 folds (repeat 0)
r0 = AM_[AM_.repeat == 0]
N["tuner_and"] = pd.concat([tests(r0, "roc_auc", ["fold"], models=["RS-XGBoost", "TPE-XGBoost", "XGBoost"]),
                            tests(r0, "flow_roc_auc", ["fold"], models=["RS-XGBoost", "TPE-XGBoost", "XGBoost"])])

# ---------------------------------------------------------------- protocol / feature ablation (PSO-XGBoost + defaults)
abl = {}
for feats in ("main", "ports", "noinitwin"):
    a = A[(A.features == feats) & (A.repeat == 0)]
    abl[feats] = {m: (a[a.model == m].flow_roc_auc.mean(), a[a.model == m].flow_roc_auc.std(ddof=1),
                      a[a.model == m].roc_auc.mean(), a[a.model == m].roc_auc.std(ddof=1)) for m in ("PSO-XGBoost", "XGBoost")}
fl = D[(D.dataset == "andmal") & (D.protocol == "flow")]
abl["random"] = {m: (fl[fl.model == m].roc_auc.mean(), fl[fl.model == m].roc_auc.std(ddof=1), np.nan, np.nan) for m in ("PSO-XGBoost", "XGBoost")}
N["abl"] = abl
def paired(a, b, col):
    x = a.set_index("fold")[col]; y = b.set_index("fold")[col]; return ctest((x - y).dropna())
pr = A[(A.repeat == 0) & (A.model == "PSO-XGBoost")]
N["abl_tests"] = {f: {c: paired(pr[pr.features == f], pr[(pr.features == "main") & (pr.modelset == "full")], c) for c in ("flow_roc_auc", "roc_auc")}
                  for f in ("ports", "noinitwin")}
rnd = fl[fl.model == "PSO-XGBoost"].set_index("fold").roc_auc; app0 = pr[(pr.features == "main") & (pr.modelset == "full")].set_index("fold").flow_roc_auc
N["abl_random_minus_app"] = ctest((rnd - app0).values)                          # different splits: descriptive only

# ---------------------------------------------------------------- evidence curve
Km = K[(K.features == "main") & (K.modelset == "full")]
N["evid"] = {m: Km[Km.model == m].groupby("k").roc_auc.agg(["mean", "std"]) for m in MODELS}

# ---------------------------------------------------------------- calibration
N["cal_and"] = AC[AC.features == "main"].groupby(["model", "method"])[["ece", "brier"]].agg(["mean", "std"])
Cg = C[(C.dataset == "aagm")]; N["cal_agm"] = Cg.groupby(["model", "method"])[["ece", "brier"]].agg(["mean", "std"])
Cs = C[(C.dataset == "andmal") & (C.protocol == "app") & (C.features == "main") & (C.modelset == "full") & (C.method == "platt")]
N["calsrc"] = {m: Cs[Cs.model == m].groupby("calsource").ece.agg(["mean", "std"]) for m in ("PSO-XGBoost", "XGBoost")}
pv = Cs.pivot_table(index=["repeat", "fold", "model"], columns="calsource", values="ece").reset_index()
src = []
for m in MODELS:
    x = pv[pv.model == m]
    if len(x): src.append(dict(model=m, **ctest((x.heldout - x.seen).values)))
src = pd.DataFrame(src); src["p_holm"] = holm(src.p); N["calsrc_tests"] = src
Cf = C[(C.dataset == "andmal") & (C.protocol == "app") & (C.features == "main") & (C.modelset == "full") & (C.calsource == "heldout")]
N["cal_flow_and"] = Cf[Cf.model == "PSO-XGBoost"].groupby("method")[["ece", "brier"]].mean()

# ---------------------------------------------------------------- deferral
RULES = ["cost_window", "symmetric", "error_window"]
ASm = AS[(AS.features == "main") & (AS.modelset == "full")]
N["def_and"] = ASm[ASm.rule.isin(RULES)].groupby(["model", "rule", "target"]).mean(numeric_only=True)
Sg = S[(S.dataset == "aagm") & (S.calsource == "heldout")]
N["def_agm"] = Sg[Sg.rule.isin(RULES)].groupby(["model", "rule", "target"]).mean(numeric_only=True)
N["aurc_and"] = ASm[ASm.rule == "aurc"].groupby("model").aurc.agg(["mean", "std"])
N["aurc_agm"] = Sg[Sg.rule == "aurc"].groupby("model").aurc.agg(["mean", "std"])
def rule_tests(df, idx, models=("PSO-XGBoost", "XGBoost"), col="cost_per10k", a="cost_window", b="symmetric", rates=(.05, .10, .20, .30)):
    rows = []
    for m in models:
        for r in rates:
            x = df[(df.model == m) & (df.target == r)].pivot_table(index=idx, columns="rule", values=col)
            rows.append(dict(model=m, target=r, **ctest((x[a] - x[b]).values)))
    out = pd.DataFrame(rows); out["p_holm"] = holm(out.p); return out
N["rt_and_cost"] = rule_tests(ASm, ["repeat", "fold"]); N["rt_agm_cost"] = rule_tests(Sg, ["fold"])
N["rt_and_err"] = rule_tests(ASm, ["repeat", "fold"], col="err_auto", a="error_window"); N["rt_agm_err"] = rule_tests(Sg, ["fold"], col="err_auto", a="error_window")
N["rt_and_fnr"] = rule_tests(ASm, ["repeat", "fold"], col="fnr_auto", a="error_window"); N["rt_agm_fnr"] = rule_tests(Sg, ["fold"], col="fnr_auto", a="error_window")
N["win_and"] = ASm[(ASm.model == "PSO-XGBoost") & (ASm.target == .2) & ASm.rule.isin(RULES)].groupby("rule")[["tau_lo", "tau_hi", "threshold", "defer"]].agg(["min", "max", "mean"])
N["win_agm"] = Sg[(Sg.model == "PSO-XGBoost") & (Sg.target == .2) & Sg.rule.isin(RULES)].groupby("rule")[["tau_lo", "tau_hi", "threshold", "defer"]].agg(["min", "max", "mean"])
N["thr_and"] = ASm[ASm.model == "PSO-XGBoost"].threshold.agg(["min", "max", "mean"])
N["V_and"] = AV[(AV.features == "main") & (AV.model == "PSO-XGBoost")].groupby("rule")[["defer", "err_auto", "fnr_auto", "fpr_auto", "cost_per10k"]].mean()
N["V_agm"] = V[(V.dataset == "aagm") & (V.model == "PSO-XGBoost")].groupby("rule")[["defer", "err_auto"]].mean()

# ---------------------------------------------------------------- conformal
def conf_rows(df, keys):
    g = df.groupby(keys)[["cov_mal_hits", "n_mal", "cov_ben_hits", "n_ben"]].sum()
    g2 = df.groupby(keys)[["singleton", "err_singleton", "defer"]].mean(); return g.join(g2)
Am = AM[(AM.features == "main") & (AM.model == "PSO-XGBoost")]
N["conf_and_rep"] = conf_rows(Am, ["repeat", "alpha"])
Mg = M[(M.dataset == "aagm") & (M.model == "PSO-XGBoost")]
N["conf_agm"] = conf_rows(Mg, ["alpha"])
N["conf_and_models"] = conf_rows(AM[(AM.features == "main") & (AM.repeat == 0)], ["model", "alpha"])
N["conf_agm_models"] = conf_rows(M[M.dataset == "aagm"], ["model", "alpha"])
N["cp"] = cp

# ---------------------------------------------------------------- prevalence
N["prev"] = PREV.groupby(["dataset", "model", "prior", "threshold"]).mean(numeric_only=True)

# ---------------------------------------------------------------- families
f = AF[AF.features == "main"] if "features" in AF else AF
N["fam_cat"] = f.groupby("category")[["n_apps", "detected"]].sum().assign(rate=lambda x: x.detected / x.n_apps)
N["fam"] = f.groupby(["category", "family"])[["n_apps", "detected"]].sum().assign(rate=lambda x: x.detected / x.n_apps)
fp0 = ASm[(ASm.model == "PSO-XGBoost") & (ASm.rule == "symmetric") & (ASm.target == 0)]
N["app_fpr0"], N["app_fnr0"] = fp0.fpr_auto.mean(), fp0.fnr_auto.mean()

# ---------------------------------------------------------------- PSO search details
def pso_info(pat, kinds=("PSO-XGBoost", "RS-XGBoost", "TPE-XGBoost", "PSO-LightGBM", "PSO-RF", "PSO-ExtraTrees")):
    rows = []
    for fn in sorted(glob.glob(PRED + pat)):
        info = json.loads(str(np.load(fn, allow_pickle=True)["info"]))
        for k in kinds:
            if k in info: rows.append(dict(file=fn.split("/")[-1], model=k, minutes=info[k]["minutes"], inner_auc=info[k]["inner_auc"], **info[k]["params"]))
    return pd.DataFrame(rows)
N["pso_and"] = pso_info("andmal_app_main_r*_f*_full.npz"); N["pso_agm"] = pso_info("aagm_flow_main_r0_f*_full.npz")
SPACE = {"xgb": dict(learning_rate=(.01, .3), max_depth=(3, 12), n_estimators=(50, 400), subsample=(.5, 1.), colsample_bytree=(.5, 1.)),
         "lgb": dict(learning_rate=(.01, .3), num_leaves=(15, 255), n_estimators=(50, 400), subsample=(.5, 1.), colsample_bytree=(.5, 1.)),
         "rf": dict(n_estimators=(100, 300), max_depth=(5, 30), max_features=(.05, .5), min_samples_leaf=(1, 10)),
         "et": dict(n_estimators=(100, 300), max_depth=(5, 30), max_features=(.05, 1.), min_samples_leaf=(1, 10))}
N["SPACE"] = SPACE
def bound_hits(df, model, kind):
    x = df[df.model == model]; out = {}
    for p, (lo, hi) in SPACE[kind].items():
        v = x[p].astype(float); out[p] = (int((np.isclose(v, lo)).sum()), int((np.isclose(v, hi)).sum()), len(v))
    return out
N["bounds"] = {ds: {m: bound_hits(N[f"pso_{ds}"], m, k) for m, k in (("PSO-XGBoost", "xgb"), ("RS-XGBoost", "xgb"), ("TPE-XGBoost", "xgb"),
               ("PSO-LightGBM", "lgb"), ("PSO-RF", "rf"), ("PSO-ExtraTrees", "et"))} for ds in ("and", "agm")}

# ---------------------------------------------------------------- SHAP
N["shap_and"] = pd.read_csv(R + "shap_andmal_app_main.csv"); N["shap_ports"] = pd.read_csv(R + "shap_andmal_app_ports.csv")
N["shap_agm"] = pd.read_csv(R + "shap_aagm_flow_main.csv")
N["errtest"] = {k: pd.read_csv(R + f"shap_errtest_{k}.csv").iloc[0] for k in ("andmal_app_main", "andmal_app_ports", "aagm_flow_main")}

# ---------------------------------------------------------------- flow-level triage on CICAndMal2017 (for contrast)
Sa = S[(S.dataset == "andmal") & (S.protocol == "app") & (S.features == "main") & (S.modelset == "full") & (S.calsource == "heldout")]
N["def_and_flow"] = Sa[Sa.rule.isin(RULES)].groupby(["model", "rule", "target"]).mean(numeric_only=True)

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for k, v in N.items():
        if k in ("cp",): continue
        print("=====", k); print(v if not isinstance(v, pd.DataFrame) else v.round(4).to_string())
