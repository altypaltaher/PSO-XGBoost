"""Stage 1: per outer fold, PSO search + model fitting on FIT, probabilities saved for CAL and TEST.

Per outer fold f of a stratified 5-fold CV (seed 42):
  training part -> stratified 75/25 split (seed 42 + f) into FIT (60% of all flows) and CAL (20%)
  test part (20%) is used for reporting only.
PSO-XGBoost: 8 particles x 10 iterations, w = 0.7, c1 = c2 = 1.5, positions clipped to the bounds, zero initial
velocities; fitness = 1 - mean F1 of a stratified 3-fold CV on FIT (seed 42 + f); the best particle is refitted on FIT.
Search space: learning rate [0.01, 0.3], max depth {3..10}, trees {50..300}, subsample [0.6, 1.0].
Also trained: PSO-LightGBM with the same budget (the preliminary tuned-LightGBM comparison mentioned in the
limitations of the paper; not part of the reported tables) and the four fixed-setting baselines.
Output: results/fold_predictions/<dataset>_fold<f>.npz (resumable per fold).
Usage: python src/train.py <dataset>      dataset in {CICAndMal2017, CIC-AAGM2017}
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, xgboost as xgb, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from common import load, bases, SEED, NJ, PRED

BOUNDS = np.array([[0.01, 0.3], [3, 10], [50, 300], [0.6, 1.0]])

def decode(v):
    return {"learning_rate": float(np.clip(v[0], .01, .3)), "max_depth": int(np.clip(round(v[1]), 3, 10)),
            "n_estimators": int(np.clip(round(v[2]), 50, 300)), "subsample": float(np.clip(v[3], .6, 1.))}

def make(kind, p):
    if kind == "xgb":
        return xgb.XGBClassifier(**p, eval_metric="logloss", random_state=SEED, n_jobs=NJ)
    return lgb.LGBMClassifier(**p, subsample_freq=1, random_state=SEED, n_jobs=NJ, verbose=-1)

def outer_folds(X, y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))

def fit_cal_split(tr, y, f):
    return train_test_split(tr, test_size=.25, random_state=SEED + f, stratify=y[tr])

def pso(kind, X, y, seed, n_particles=8, n_iter=10, w=.7, c1=1.5, c2=1.5):
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    def fitness(v):
        p = decode(v)
        return 1 - np.mean([f1_score(y[b], make(kind, p).fit(X[a], y[a]).predict(X[b])) for a, b in inner])
    d = len(BOUNDS); rng = np.random.RandomState(seed)
    pos = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_particles, d)); vel = np.zeros_like(pos)
    pb, ps = pos.copy(), np.array([fitness(q) for q in pos]); gb, gs = pb[ps.argmin()].copy(), ps.min()
    for _ in range(n_iter):
        for i in range(n_particles):
            r1, r2 = rng.rand(d), rng.rand(d)
            vel[i] = w * vel[i] + c1 * r1 * (pb[i] - pos[i]) + c2 * r2 * (gb - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], BOUNDS[:, 0], BOUNDS[:, 1]); s = fitness(pos[i])
            if s < ps[i]: pb[i], ps[i] = pos[i].copy(), s
        if ps.min() < gs: gb, gs = pb[ps.argmin()].copy(), ps.min()
    return decode(gb), gs

if __name__ == "__main__":
    ds = sys.argv[1]; X, y, _ = load(ds); PRED.mkdir(parents=True, exist_ok=True)
    print(ds, X.shape, np.bincount(y), flush=True)
    for f, (tr, te) in enumerate(outer_folds(X, y)):
        fn = PRED / f"{ds}_fold{f}.npz"
        if fn.exists(): continue
        t0 = time.time(); fi, ca = fit_cal_split(tr, y, f); P, params = {}, {}
        for name, kind in (("PSO-XGBoost", "xgb"), ("PSO-LightGBM", "lgb")):
            t = time.time(); p, best = pso(kind, X[fi], y[fi], SEED + f)
            m = make(kind, p).fit(X[fi], y[fi]); P[name] = (m.predict_proba(X[ca])[:, 1], m.predict_proba(X[te])[:, 1])
            params[name] = {**p, "pso_best_1-F1": best, "pso_min": (time.time() - t) / 60}
            print(f"  {ds} fold{f} {name} {p} inner-CV F1={1 - best:.4f} ({(time.time() - t) / 60:.1f} min)", flush=True)
        for name, clf in bases().items():
            clf.fit(X[fi], y[fi]); P[name] = (clf.predict_proba(X[ca])[:, 1], clf.predict_proba(X[te])[:, 1])
        np.savez_compressed(fn, y_cal=y[ca], y_te=y[te], models=np.array(list(P)), params=json.dumps(params),
                            **{f"cal__{k}": v[0] for k, v in P.items()}, **{f"te__{k}": v[1] for k, v in P.items()})
        print(f"{ds} fold{f} saved (fit={len(fi)} cal={len(ca)} test={len(te)}) {(time.time() - t0) / 60:.1f} min", flush=True)
