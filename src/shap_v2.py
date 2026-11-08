"""SHAP for PSO-XGBoost (repeat 0) and a test of whether SHAP evidence conflict predicts errors beyond |q - 0.5|.
Usage: python shap_v2.py <dataset> <protocol> <features> <modelset>
For every outer fold the split is regenerated exactly as in train_v2.py, the model is refitted with the saved PSO
hyperparameters, and its test probabilities are checked against the saved ones. TreeSHAP (exact, log-odds) is computed
for all test flows. Error-prediction test: target = error of the Platt-calibrated rule 1[q >= 0.5]; predictors
(a) |q - 0.5|, (b) |q - 0.5| + conflict K + total evidence E; logistic regression, 5-fold CV (grouped by app when
available); out-of-fold ROC-AUC of (a) vs (b), with a paired bootstrap CI of the difference (resampling folds' flows
by group)."""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, numpy as np, pandas as pd, xgboost as xgb, shap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
sys.path.insert(0, ROOT + "/src")
from train_v2 import load, make
def platt(pc, yc, EPS=1e-6):
    lg = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    m = LogisticRegression(C=1e6).fit(lg(pc)[:, None], yc); return lambda p: m.predict_proba(lg(p)[:, None])[:, 1]
ds, proto, feats, mset = sys.argv[1:5]; rep = 0; seed0 = 42
X, y, g, fam, cols = load(ds, feats); OUT = Path(ROOT + "/results"); FIG = Path(ROOT + "/figures"); FIG.mkdir(exist_ok=True)
outer = (StratifiedGroupKFold(5, shuffle=True, random_state=seed0).split(X, y, g) if proto == "app"
         else StratifiedKFold(5, shuffle=True, random_state=seed0).split(X, y))
S, XT, YT, Q, G = [], [], [], [], []
for f, (tr, te) in enumerate(outer):
    seed = seed0 + f; z = np.load(ROOT + f"/predictions/{ds}_{proto}_{feats}_r0_f{f}_{mset}.npz", allow_pickle=True)
    if proto == "app":
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X[tr], y[tr], g[tr]))
        fit_apps, cal = tr[a], tr[b]; trn, _ = train_test_split(fit_apps, test_size=.2, random_state=seed, stratify=y[fit_apps])
    else:
        trn, cal = train_test_split(tr, test_size=.25, random_state=seed, stratify=y[tr])
    assert np.array_equal(z["idx_te"], te) and np.array_equal(z["idx_cal"], cal)
    p = json.loads(str(z["info"]))["PSO-XGBoost"]["params"]; m = make("xgb", p, seed).fit(X[trn], y[trn])
    pt = m.predict_proba(X[te])[:, 1]; assert np.allclose(pt, z["te__PSO-XGBoost"], atol=1e-5), "not reproduced"
    bst = m.get_booster(); D = xgb.DMatrix(X[te]); C = bst.predict(D, pred_contribs=True)
    assert np.allclose(C.sum(1), bst.predict(D, output_margin=True), atol=1e-3)
    q = platt(z["cal__PSO-XGBoost"], z["y_cal"])(pt)
    S.append(C[:, :-1]); XT.append(X[te]); YT.append(y[te]); Q.append(q); G.append(g[te] if g is not None else te)
    print(ds, proto, feats, "fold", f, "reproduced", flush=True)
S, XT, YT, Q, G = map(np.concatenate, (S, XT, YT, Q, G))
imp = np.abs(S).mean(0); share = 100 * imp / imp.sum()
pd.DataFrame({"feature": cols, "mean_abs_shap": imp, "share_pct": share}).sort_values("share_pct", ascending=False) \
  .to_csv(OUT / f"shap_{ds}_{proto}_{feats}.csv", index=False)
pos = np.clip(S, 0, None).sum(1); neg = -np.clip(S, None, 0).sum(1); E = pos + neg; K = np.minimum(pos, neg) / E
err = ((Q >= .5).astype(int) != YT).astype(int); conf = np.abs(Q - .5)
cv = GroupKFold(5) if proto == "app" else StratifiedKFold(5, shuffle=True, random_state=0)
def oof(F):
    o = np.zeros(len(err))
    for a, b in (cv.split(F, err, G) if proto == "app" else cv.split(F, err)):
        o[b] = LogisticRegression(max_iter=1000).fit(F[a], err[a]).predict_proba(F[b])[:, 1]
    return o
Fa = np.column_stack([conf]); Fb = np.column_stack([conf, K, np.log1p(E)])
oa, ob = oof(Fa), oof(Fb); auc_a, auc_b = roc_auc_score(err, oa), roc_auc_score(err, ob)
rng = np.random.RandomState(0); ug = np.unique(G); gi = {u: np.where(G == u)[0] for u in ug}; diffs = []
for _ in range(1000):
    idx = np.concatenate([gi[u] for u in rng.choice(ug, len(ug), replace=True)])
    if err[idx].min() == err[idx].max(): continue
    diffs.append(roc_auc_score(err[idx], ob[idx]) - roc_auc_score(err[idx], oa[idx]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
res = {"dataset": ds, "protocol": proto, "features": feats, "n": len(err), "error_rate": err.mean(),
       "auc_conf_only": auc_a, "auc_conf_plus_shap": auc_b, "delta": auc_b - auc_a, "delta_ci_lo": lo, "delta_ci_hi": hi,
       "corr_K_conf": np.corrcoef(K, conf)[0, 1], "top10_share": np.sort(share)[::-1][:10].sum()}
pd.DataFrame([res]).to_csv(OUT / f"shap_errtest_{ds}_{proto}_{feats}.csv", index=False); print(res)
ex = shap.Explanation(values=S, data=XT, feature_names=cols)
plt.figure(); shap.plots.beeswarm(ex, max_display=12, show=False); plt.title(f"{ds} ({proto} split, {feats})", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / f"shap_{ds}_{proto}_{feats}.png", dpi=300, bbox_inches="tight"); plt.close("all")
