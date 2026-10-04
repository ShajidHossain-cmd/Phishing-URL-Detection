
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from feature_engineering import FEATURES  # teammate 1's 21 feature columns

SEED = 42
K_VALUES = [2, 4]   # the two candidates we are comparing
TOP_N = 3           # how many standout features to report per cluster

#Load data, keep only the phishing class 
HERE = os.path.dirname(os.path.abspath(__file__))  
path = os.path.join(HERE, "processed_merged_urls_v2.csv")
if not os.path.exists(path):
    raise SystemExit(f"Dataset not found. Put processed_merged_urls_v2.csv in: {HERE}")
df = pd.read_csv(path)
phishing = df[df["label"] == 1].reset_index(drop=True)  # label only used for this filter

raw = phishing[FEATURES].copy()   # untouched values, kept so we can quote readable numbers
X = raw.copy()

#Same preprocessing as choose_k.py: log-transform skewed columns, then scale 
skewed_cols = ["url_length", "host_length", "path_length", "query_length",
               "dot_count", "hyphen_count", "at_count", "percent_count",
               "digit_count", "subdomain_count", "path_depth", "query_param_count"]
for col in skewed_cols:
    X[col] = np.log1p(X[col])
X_scaled = StandardScaler().fit_transform(X)
scaled = pd.DataFrame(X_scaled, columns=FEATURES)

binary_cols = [c for c in FEATURES if c.startswith("has_")]  # yes/no flags


def describe(value, col):
    """Readable value: a rate for yes/no flags, a median for everything else."""
    return f"{value:.0%}" if col in binary_cols else f"{value:.1f}"


for k in K_VALUES:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
    labels = km.labels_
    rows = []
    print(f"\n{'=' * 70}\nk = {k}\n{'=' * 70}")

    for c in range(k):
        members = np.where(labels == c)[0]
        print(f"\n--- Cluster {c}: {len(members):,} URLs ({len(members) / len(labels):.1%}) ---")

        # Confounder check: does this cluster just mean "one dataset"?
        src = phishing.loc[members, "source"].value_counts(normalize=True)
        print("Source mix:", ", ".join(f"{s} {p:.0%}" for s, p in src.items()))

        # Scaled data has mean 0 over the whole phishing class, so a cluster's mean
        # on a scaled feature IS how many standard deviations it sits from the average.
        diff = scaled.loc[members].mean()
        top = diff.reindex(diff.abs().sort_values(ascending=False).index).head(TOP_N)
        print(f"Top {TOP_N} features vs phishing average:")
        for col, d in top.items():
            cl_val = raw.loc[members, col].mean() if col in binary_cols else raw.loc[members, col].median()
            all_val = raw[col].mean() if col in binary_cols else raw[col].median()
            word = "mean" if col not in binary_cols else "rate"
            kind = "median" if col not in binary_cols else "rate"
            print(f"  {col:20s} {d:+.2f} sd   (cluster {kind} {describe(cl_val, col)} "
                  f"vs overall {describe(all_val, col)})")

        # Representative examples = the 3 URLs closest to the cluster centre
        dist = np.linalg.norm(X_scaled[members] - km.cluster_centers_[c], axis=1)
        nearest = members[np.argsort(dist)[:3]]
        print("Examples:")
        for i in nearest:
            print("  ", phishing.loc[i, "url"][:100])

        rows.append({"cluster": c, "size": len(members), **diff.round(3).to_dict()})

    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"cluster_profile_k{k}.csv"), index=False)
    print(f"\nSaved cluster_profile_k{k}.csv")
