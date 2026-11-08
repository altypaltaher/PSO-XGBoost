"""CIC-AAGM2017 (TotalFeatures-ISCXFlowMeter.csv, label column 'calss') -> uniform sample at the natural malware share.
Exact duplicate rows and feature vectors that occur with conflicting labels are removed; adware ('asware' in the file)
and general malware form the malware class; a uniform random sample of 20,000 flows (seed 42) keeps the natural share;
+/-inf -> NaN; constant columns dropped. Output columns: features..., y, family (= original 'calss').
Usage: python prepare_aagm.py <TotalFeatures-ISCXFlowMeter.csv> <out.parquet>"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, low_memory=False); d.columns = d.columns.str.strip(); n0 = len(d)
d = d.drop_duplicates(); n1 = len(d); feat = [c for c in d.columns if c != "calss"]
d = d[~d.duplicated(subset=feat, keep=False)]; n2 = len(d)
d["y"] = (d["calss"].str.strip().str.lower() != "benign").astype(int)
print(f"flows {n0}: exact duplicates {n0 - n1}, conflicting-label rows {n1 - n2}, unique {n2}, malware share {d.y.mean():.4f}")
s = d.sample(20000, random_state=42).reset_index(drop=True)
X = s[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X = X.loc[:, X.notna().any()]; X = X.loc[:, X.nunique(dropna=True) > 1]
out_df = pd.concat([X.astype(float), s["y"], s["calss"].rename("family")], axis=1); out_df.to_parquet(out)
print(out_df.shape, "malware share", round(out_df.y.mean(), 4))
