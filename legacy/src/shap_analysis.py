"""SHAP analysis of PSO-XGBoost for the manuscript.
For each outer fold, PSO-XGBoost is re-fitted on the same FIT split with the PSO-selected hyperparameters saved by
calib_train.py; its test probabilities are checked against the saved ones (must be identical). Exact TreeSHAP
(XGBoost pred_contribs, log-odds space, additivity checked) is computed for every test flow; since every flow is in
exactly one test fold, pooling the five folds explains every flow once with a model that never saw it.
Deferred vs automatic flows: the Chow window at a 20% target workload (written by analysis.py).
Usage: python src/shap_analysis.py   (after analysis.py)
Per flow: total evidence E = sum_j |phi_j|, conflict ratio K = min(sum phi+, |sum phi-|) / E  (0 = all features push
the same way, 0.5 = supporting and opposing evidence cancel)."""
import json, numpy as np, pandas as pd, xgboost as xgb, shap, matplotlib
from scipy import stats
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from common import load, PRED as R, RES as OUT, FIG, DATA, SEED
from train import make, outer_folds, fit_cal_split
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
rows, summ = [], []
pooled = {}
for ds in ["CICAndMal2017", "CIC-AAGM2017"]:
    X, y, cols = load(ds)
    Ss, Xs, Ys, Ks = [], [], [], []
    for f, (tr, te) in enumerate(outer_folds(X, y)):
        z = np.load(R / f"{ds}_fold{f}.npz", allow_pickle=True)
        fi, ca = fit_cal_split(tr, y, f)
        p = {k: v for k, v in json.loads(str(z["params"]))["PSO-XGBoost"].items() if k in ("learning_rate", "max_depth", "n_estimators", "subsample")}
        m = make("xgb", p).fit(X[fi], y[fi]); pt = m.predict_proba(X[te])[:, 1]
        assert np.array_equal(y[te], z["y_te"]) and np.allclose(pt, z["te__PSO-XGBoost"], atol=1e-6), f"mismatch {ds} {f}"
        b = m.get_booster(); Dm = xgb.DMatrix(X[te]); Cn = b.predict(Dm, pred_contribs=True)
        assert np.allclose(Cn.sum(1), b.predict(Dm, output_margin=True), atol=1e-3)
        k = np.load(R / f"keep20_{ds}_fold{f}.npz"); assert np.array_equal(k["y"], y[te])
        Ss.append(Cn[:, :-1]); Xs.append(X[te]); Ys.append(y[te]); Ks.append(k["keep"])
        print(ds, f, "reproduced, additivity OK", flush=True)
    S, Xt, Yt, K = map(np.concatenate, (Ss, Xs, Ys, Ks)); pooled[ds] = (S, Xt, Yt, K, cols)
    imp = np.abs(S).mean(0); share = 100 * imp / imp.sum()
    E = np.abs(S).sum(1); pos = np.clip(S, 0, None).sum(1); neg = -np.clip(S, None, 0).sum(1); Kc = np.minimum(pos, neg) / E
    dfr = ~K
    t1 = stats.mannwhitneyu(E[dfr], E[K]); t2 = stats.mannwhitneyu(Kc[dfr], Kc[K])
    summ.append({"dataset": ds, "pos_def_median": np.median(pos[dfr]), "neg_def_median": np.median(neg[dfr]),
                 "pos_auto_median": np.median(pos[K]), "neg_auto_median": np.median(neg[K]),
                 "E_def_over_auto_%": 100 * np.median(E[dfr]) / np.median(E[K]), "n_test_flows": len(Yt), "deferred_%": 100 * dfr.mean(),
                 "E_deferred_median": np.median(E[dfr]), "E_auto_median": np.median(E[K]), "E_MWU_p": t1.pvalue,
                 "K_deferred_median": np.median(Kc[dfr]), "K_auto_median": np.median(Kc[K]), "K_MWU_p": t2.pvalue,
                 "top10_share_%": np.sort(share)[::-1][:10].sum()})
    imp_d, imp_a = np.abs(S[dfr]).mean(0), np.abs(S[K]).mean(0)
    t = pd.DataFrame({"feature": cols, "mean_abs_shap": imp, "share_%": share, "mean_abs_deferred": imp_d,
                      "mean_abs_auto": imp_a, "ratio_def_auto": imp_d / imp_a}).sort_values("mean_abs_shap", ascending=False)
    t["rank"] = range(1, len(t) + 1); t.insert(0, "dataset", ds); rows.append(t)
    # rank agreement of feature importance between deferred and automatic flows
    rho_s = stats.spearmanr(imp_d, imp_a).statistic; summ[-1]["spearman_imp_def_vs_auto"] = rho_s
T = pd.concat(rows); T.to_csv(OUT / "shap_importance.csv", index=False)
Sm = pd.DataFrame(summ); Sm.to_csv(OUT / "shap_deferral_summary.csv", index=False)
pd.set_option("display.width", 220)
print(Sm.round(4).to_string(index=False))
for ds in pooled: print(T[T.dataset == ds].head(12).round(4).to_string(index=False))

# Figure 4: SHAP beeswarm for both datasets
fig = plt.figure(figsize=(11, 5.2))
for i, ds in enumerate(pooled):
    S, Xt, Yt, K, cols = pooled[ds]
    ax = fig.add_subplot(1, 2, i + 1); plt.sca(ax)
    ex = shap.Explanation(values=S, data=Xt, feature_names=cols)
    shap.plots.beeswarm(ex, max_display=12, show=False, plot_size=None, color_bar=(i == 1))
    ax.set_title(f"({'ab'[i]}) {ds}", fontsize=10); ax.tick_params(axis="y", labelsize=8); ax.set_xlabel("SHAP value (log-odds of malware)", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "fig4_shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close("all")

# Figure 5: supporting vs opposing SHAP evidence, deferred vs automatic flows
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
rng = np.random.RandomState(0)
for i, ds in enumerate(pooled):
    S, Xt, Yt, K, cols = pooled[ds]
    pos = np.clip(S, 0, None).sum(1); neg = -np.clip(S, None, 0).sum(1)
    for msk, col, lab in ((K, "#2471a3", "Decided automatically"), (~K, "#c0392b", "Deferred to analyst")):
        idx = np.where(msk)[0]; idx = rng.choice(idx, min(2500, len(idx)), replace=False)
        ax[i].scatter(pos[idx], neg[idx], s=3, alpha=.35, color=col, label=f"{lab} (n = {msk.sum()})", rasterized=True)
    lim = np.percentile(np.r_[pos, neg], 99.5); ax[i].plot([0, lim], [0, lim], "k--", lw=.8)
    ax[i].set_xlim(0, lim); ax[i].set_ylim(0, lim); ax[i].set_aspect("equal")
    ax[i].set_xlabel("Evidence for malware  Σφ⁺ (log-odds)"); ax[i].set_ylabel("Evidence for benign  |Σφ⁻| (log-odds)")
    ax[i].set_title(f"({'ab'[i]}) {ds}", fontsize=9); leg = ax[i].legend(fontsize=7, loc="upper left", markerscale=4); ax[i].grid(alpha=.25)
plt.tight_layout(); plt.savefig(FIG / "fig5_shap_deferral.png", dpi=300); plt.close("all"); print("figures saved")
