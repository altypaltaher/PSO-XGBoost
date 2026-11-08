"""Shared settings: paths, seed, data loader and the four baseline classifiers (fixed settings)."""
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
PRED = RES / "fold_predictions"
SEED, NJ = 42, 2
DATASETS = ["CICAndMal2017", "CIC-AAGM2017"]

def load(ds):
    """Prepared dataset (see prepare_data.py) -> numeric matrix X, labels y (1 = malware)."""
    d = pd.read_parquet(DATA / f"{ds}.parquet")
    y = d.pop("y").astype(int).to_numpy()
    X = d.select_dtypes("number").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    return X, y, list(d.columns)

def bases():
    """Baselines with fixed, commonly used settings (Section 3.5 of the paper)."""
    return {"XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=.1, subsample=.8,
                                         eval_metric="logloss", random_state=SEED, n_jobs=NJ),
            # subsample has no effect in LightGBM without subsample_freq > 0, i.e. no bagging
            "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=.1, subsample=.8,
                                           random_state=SEED, n_jobs=NJ, verbose=-1),
            "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=.1, random_state=SEED),
            "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=.1, random_state=SEED)}
