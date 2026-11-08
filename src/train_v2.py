"""Revision experiments, stage 1: tuning, fitting and saving per-fold probabilities.

Usage: python train_v2.py <dataset> <protocol> <features> <repeat> <modelset>
  dataset  : andmal (CICAndMal2017 app-level, groups = app) | aagm (CIC-AAGM2017, natural prevalence, no app ids)
  protocol : app  -> outer StratifiedGroupKFold(5) by app; training apps split by app into FIT apps (75%) and CAL apps
                     (25%; 'held-out-app calibration'); 20% of the FIT-app flows are withheld from training as
                     'seen-app calibration'; inner tuning CV grouped by app.
             flow -> outer StratifiedKFold(5); training part split 75/25 at random into FIT and CAL.
  features : main (no ports) | ports (with Source/Destination Port) | noinitwin (no ports, no Init_Win_bytes_*)
  repeat   : 0, 1, 2 (outer seed 42 + 100*repeat)
  modelset : full  -> PSO-{XGBoost,LightGBM,RF,ExtraTrees} + defaults (+ RS-XGBoost, TPE-XGBoost when repeat == 0)
             small -> PSO-XGBoost + defaults
Equal tuning budget: 48 configurations per search (PSO: 8 particles x (1 + 5 iterations); RS: 48 uniform draws;
TPE: 48 Optuna trials), each scored by mean ROC-AUC of a 3-fold inner CV on the training flows (grouped by app
for the app protocol). ROC-AUC is used instead of F1 because it does not depend on the 0.5 threshold or the class prior.
"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb, optuna
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os
SMOKE = os.environ.get("SMOKE") == "1"
NJ = 2; BUDGET_IT = 1 if SMOKE else 5; NP = 2 if SMOKE else 8
OUT = Path(ROOT + "/smoke" if SMOKE else ROOT + "/predictions"); OUT.mkdir(exist_ok=True)
META = ["app", "family", "category", "date", "y"]

def load(ds, feats):
    if ds == "andmal":
        d = pd.read_parquet(ROOT + "/data/cicandmal_app_K20.parquet")
    else:
        d = pd.read_parquet(ROOT + "/data/aagm_natural20k.parquet")
    drop = [c for c in META if c in d]
    if feats in ("main", "noinitwin"): drop += [c for c in ("Source Port", "Destination Port") if c in d]
    if feats == "noinitwin": drop += [c for c in d if c.startswith("Init_Win_bytes")]
    X = d.drop(columns=drop).fillna(0).to_numpy(np.float32)
    g = d["app"].to_numpy() if "app" in d else None
    fam = d["family"].to_numpy() if "family" in d else np.array(["-"] * len(d))
    return X, d["y"].to_numpy().astype(int), g, fam, [c for c in d.columns if c not in drop]

# ------------------------------------------------------------------ search spaces (lo, hi, integer?)
SPACE = {
 "xgb": [("learning_rate", .01, .3, 0), ("max_depth", 3, 12, 1), ("n_estimators", 50, 400, 1),
         ("subsample", .5, 1., 0), ("colsample_bytree", .5, 1., 0)],
 "lgb": [("learning_rate", .01, .3, 0), ("num_leaves", 15, 255, 1), ("n_estimators", 50, 400, 1),
         ("subsample", .5, 1., 0), ("colsample_bytree", .5, 1., 0)],
 "rf":  [("n_estimators", 100, 300, 1), ("max_depth", 5, 30, 1), ("max_features", .05, .5, 0), ("min_samples_leaf", 1, 10, 1)],
 "et":  [("n_estimators", 100, 300, 1), ("max_depth", 5, 30, 1), ("max_features", .05, 1., 0), ("min_samples_leaf", 1, 10, 1)]}
def decode(kind, v):
    return {n: (int(round(np.clip(x, lo, hi))) if isint else float(np.clip(x, lo, hi)))
            for (n, lo, hi, isint), x in zip(SPACE[kind], v)}
def make(kind, p, seed):
    if kind == "xgb": return xgb.XGBClassifier(**p, tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=NJ)
    if kind == "lgb": return lgb.LGBMClassifier(**p, subsample_freq=1, random_state=seed, n_jobs=NJ, verbose=-1)
    if kind == "rf": return RandomForestClassifier(**p, random_state=seed, n_jobs=NJ)
    return ExtraTreesClassifier(**p, random_state=seed, n_jobs=NJ)
def defaults(seed):
    return {"XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=.1, subsample=.8,
                                         eval_metric="logloss", random_state=seed, n_jobs=NJ),
            "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=.1, random_state=seed, n_jobs=NJ, verbose=-1),
            "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=.1, random_state=seed),
            "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=.1, random_state=seed)}

def inner_folds(X, y, g, seed):
    if g is None: return list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    return list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g))

def objective(kind, X, y, folds, seed):
    def f(p): return 1 - np.mean([roc_auc_score(y[b], make(kind, p, seed).fit(X[a], y[a]).predict_proba(X[b])[:, 1]) for a, b in folds])
    return f

def search(method, kind, X, y, folds, seed):
    obj = objective(kind, X, y, folds, seed); B = np.array([[lo, hi] for _, lo, hi, _ in SPACE[kind]]); d = len(B)
    rng = np.random.RandomState(seed); hist = []
    if method == "PSO":
        w, c1, c2 = .7, 1.5, 1.5
        pos = rng.uniform(B[:, 0], B[:, 1], (NP, d)); vel = np.zeros_like(pos)
        pb = pos.copy(); ps = np.array([obj(decode(kind, q)) for q in pos]); gb, gs = pb[ps.argmin()].copy(), ps.min(); hist.append(gs)
        for _ in range(BUDGET_IT):
            for i in range(NP):
                r1, r2 = rng.rand(d), rng.rand(d)
                vel[i] = w * vel[i] + c1 * r1 * (pb[i] - pos[i]) + c2 * r2 * (gb - pos[i])
                pos[i] = np.clip(pos[i] + vel[i], B[:, 0], B[:, 1]); s = obj(decode(kind, pos[i]))
                if s < ps[i]: pb[i], ps[i] = pos[i].copy(), s
            if ps.min() < gs: gb, gs = pb[ps.argmin()].copy(), ps.min()
            hist.append(gs)
        return decode(kind, gb), gs, hist
    n = NP * (1 + BUDGET_IT)
    if method == "RS":
        best = (np.inf, None)
        for t in range(n):
            v = rng.uniform(B[:, 0], B[:, 1]); s = obj(decode(kind, v))
            if s < best[0]: best = (s, v)
            if (t + 1) % NP == 0: hist.append(best[0])
        return decode(kind, best[1]), best[0], hist
    def trial(tr):
        p = {nm: (tr.suggest_int(nm, lo, hi) if isint else tr.suggest_float(nm, lo, hi)) for nm, lo, hi, isint in SPACE[kind]}
        return obj(p)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)); st.optimize(trial, n_trials=n)
    vals = [t.value for t in st.trials]; hist = [min(vals[:k]) for k in range(NP, n + 1, NP)]
    return {k: v for k, v in st.best_params.items()}, st.best_value, hist

if __name__ == "__main__":
    ds, proto, feats, rep, mset = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    X, y, g, fam, cols = load(ds, feats); seed0 = 42 + 100 * rep
    print(ds, proto, feats, rep, mset, X.shape, f"malware {y.mean():.3f}", flush=True)
    outer = (StratifiedGroupKFold(5, shuffle=True, random_state=seed0).split(X, y, g) if proto == "app"
             else StratifiedKFold(5, shuffle=True, random_state=seed0).split(X, y))
    for f, (tr, te) in enumerate(outer):
        fn = OUT / f"{ds}_{proto}_{feats}_r{rep}_f{f}_{mset}.npz"
        if fn.exists(): continue
        t0 = time.time(); seed = seed0 + f; parts = {}
        if proto == "app":
            a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X[tr], y[tr], g[tr]))
            fit_apps, cal_held = tr[a], tr[b]
            trn, cal_seen = train_test_split(fit_apps, test_size=.2, random_state=seed, stratify=y[fit_apps])
            assert not set(g[te]) & set(g[tr]) and not set(g[cal_held]) & set(g[fit_apps])
            parts = {"cal": cal_held, "calseen": cal_seen, "te": te}; gi = g[trn]
        else:
            trn, cal = train_test_split(tr, test_size=.25, random_state=seed, stratify=y[tr])
            parts = {"cal": cal, "te": te}; gi = None
        folds = inner_folds(X[trn], y[trn], gi, seed)
        jobs = [("PSO-XGBoost", "PSO", "xgb")]
        if mset == "full":
            jobs += [("PSO-LightGBM", "PSO", "lgb"), ("PSO-RF", "PSO", "rf"), ("PSO-ExtraTrees", "PSO", "et")]
            if rep == 0: jobs += [("RS-XGBoost", "RS", "xgb"), ("TPE-XGBoost", "TPE", "xgb")]
        P, info = {}, {}
        for name, meth, kind in jobs:
            t = time.time(); p, best, hist = search(meth, kind, X[trn], y[trn], folds, seed)
            m = make(kind, p, seed).fit(X[trn], y[trn])
            P[name] = {k: m.predict_proba(X[v])[:, 1] for k, v in parts.items()}
            info[name] = {"params": p, "inner_auc": 1 - best, "convergence_auc": [1 - h for h in hist], "minutes": (time.time() - t) / 60}
            print(f"  f{f} {name}: inner AUC {1 - best:.4f} {p} ({(time.time() - t) / 60:.1f} min)", flush=True)
        for name, clf in defaults(seed).items():
            clf.fit(X[trn], y[trn]); P[name] = {k: clf.predict_proba(X[v])[:, 1] for k, v in parts.items()}
        save = {"models": np.array(list(P)), "info": json.dumps(info), "cols": np.array(cols)}
        for k, v in parts.items():
            save[f"y_{k}"] = y[v]; save[f"idx_{k}"] = v
            for mname in P: save[f"{k}__{mname}"] = P[mname][k]
        save["fam_te"] = fam[te]; save["n_train"] = len(trn)
        np.savez_compressed(fn, **save)
        print(f"{fn.name} saved: train={len(trn)} " + " ".join(f"{k}={len(v)}" for k, v in parts.items()) +
              f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if SMOKE: break
    print("DONE", ds, proto, feats, rep, mset, flush=True)
