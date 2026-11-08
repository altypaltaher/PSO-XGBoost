"""CICAndMal2017 (complete per-app CSVs, 2,126 apps) -> app-level dataset with app, family, category and capture date.
Identifiers (Flow ID, IPs, Timestamp) are removed. Ports are kept as columns here so that the port ablation can use
them; the main feature set excludes them. Exact duplicate rows (main features + label) are removed, and feature
vectors that occur with both labels are removed. Then at most K flows per app are drawn (seed 42).
Usage: python prepare_cicandmal_app.py <Data dir> <K> <out.parquet>"""
import os as _os
ROOT = _os.environ.get("TRIAGE_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, glob, os, numpy as np, pandas as pd
root, K, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ID = {"flow id", "source ip", "destination ip", "timestamp"}; frames = []; cols = None
for f in sorted(glob.glob(os.path.join(root, "*", "*", "*.csv"))):
    cat, fam, fn = f.split(os.sep)[-3:]
    d = pd.read_csv(f, low_memory=False); d.columns = d.columns.str.strip()
    d = d.drop(columns=[c for c in d.columns if c.lower() in ID or c == "Label"])
    if cols is None: cols = list(d.columns)
    d = d.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    d["app"], d["family"], d["category"] = fn.replace(".pcap_ISCX.csv", ""), fam, cat
    d["date"] = fn[:10]; d["y"] = np.int8(cat != "Benign")
    for c in ("app", "family", "category", "date"): d[c] = d[c].astype("category")
    frames.append(d)
D = pd.concat(frames, ignore_index=True); del frames
for c in ("app", "family", "category", "date"): D[c] = D[c].astype("category")
n0 = len(D); feat = cols
main = [c for c in feat if c not in ("Source Port", "Destination Port")]
h = pd.util.hash_pandas_object(D[main], index=False).to_numpy()
key = pd.DataFrame({"h": h, "y": D.y.to_numpy()})
dup = key.duplicated(subset=["h", "y"]).to_numpy()                        # exact duplicates (main features + label)
nl = key[~dup].groupby("h").y.nunique(); bad = set(nl[nl > 1].index)       # vectors seen with both labels
conf = np.isin(h, list(bad)); keep = ~dup & ~conf
print(f"flows {n0}: exact duplicates {int(dup.sum())}, conflicting-label rows {int((conf & ~dup).sum())}, kept {int(keep.sum())}")
D = D[keep].reset_index(drop=True)
D = D.sample(frac=1, random_state=42).reset_index(drop=True)              # random order, then first K per app
S = D[D.groupby("app", observed=True).cumcount() < K].copy()
for c in ("app", "family", "category", "date"): S[c] = S[c].astype(str)
S = S.loc[:, S.notna().any()]
const = [c for c in feat if c in S and S[c].nunique(dropna=True) <= 1]; S = S.drop(columns=const)
S.reset_index(drop=True).to_parquet(out)
print("dropped constant:", const)
print(S.shape, "malware share", round(S.y.mean(), 4), "apps", S.app.nunique(), S.groupby("y").app.nunique().to_dict(),
      "families", S[S.y == 1].family.nunique())
print(S.groupby("category").size().to_dict())
