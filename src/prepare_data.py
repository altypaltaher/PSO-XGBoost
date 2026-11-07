"""Build the two analysis datasets from the original CIC files (Section 3.2).

  CICAndMal2017 : class-balanced flow-level CSV (CICFlowMeter, label column 'Label1');
                  identifiers Flow ID / Source IP / Destination IP / Timestamp removed, ports kept.
  CIC-AAGM2017  : TotalFeatures-ISCXFlowMeter.csv (label column 'calss'); exact duplicates and feature vectors with
                  conflicting labels removed, adware + general malware = malware, seeded class-balanced sample of
                  20,000 flows (10,000 per class).
Both: numeric columns only, +/-inf -> NaN, all-missing and constant columns dropped.

Usage: python src/prepare_data.py <CICAndMal2017.csv> <TotalFeatures-ISCXFlowMeter.csv>
Output: data/CICAndMal2017.parquet, data/CIC-AAGM2017.parquet
"""
import sys, numpy as np, pandas as pd
from common import DATA, SEED

def clean(df, y):
    X = df.drop(columns=[c for c in df.columns if c.lower() in {"flow id", "source ip", "destination ip", "timestamp"}])
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]; X = X.loc[:, X.nunique(dropna=True) > 1]
    return pd.concat([X.reset_index(drop=True).astype(float), pd.Series(np.asarray(y), name="y").astype(int)], axis=1)

def andmal(path):
    df = pd.read_csv(path, low_memory=False); df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=["Label1"])
    y = pd.to_numeric(df["Label1"], errors="coerce"); df, y = df[y.notna()], y[y.notna()].astype(int)
    return clean(df.drop(columns=["Label1"]), y)

def aagm(path):
    d = pd.read_csv(path, low_memory=False); d.columns = d.columns.str.strip()
    d = d.drop_duplicates(); feat = [c for c in d.columns if c != "calss"]
    d = d[~d.duplicated(subset=feat, keep=False)]                       # drop label-conflicting feature vectors
    d["y"] = (d["calss"].str.strip().str.lower() != "benign").astype(int)
    s = pd.concat([d[d.y == 1].sample(10000, random_state=SEED), d[d.y == 0].sample(10000, random_state=SEED)])
    s = s.sample(frac=1, random_state=SEED).drop(columns="calss").reset_index(drop=True)
    return clean(s.drop(columns="y"), s["y"])

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, fn, path in (("CICAndMal2017", andmal, sys.argv[1]), ("CIC-AAGM2017", aagm, sys.argv[2])):
        t = fn(path); t.to_parquet(DATA / f"{name}.parquet")
        print(name, t.shape[0], "flows,", t.shape[1] - 1, "features,", t.y.value_counts().to_dict())
