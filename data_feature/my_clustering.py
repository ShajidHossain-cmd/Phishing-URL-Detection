
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from feature_engineering import FEATURES, extract_features  # teammate 1's code

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "phishing_kmeans.joblib")

N_CLUSTERS = 4
SEED = 42      # fixed so results can be reproduced
N_INIT = 10    # k-means restarts from 10 random starts and keeps the best

# Length/count columns with long right tails; log-transformed before scaling.
SKEWED_COLS = ["url_length", "host_length", "path_length", "query_length",
               "dot_count", "hyphen_count", "at_count", "percent_count",
               "digit_count", "subdomain_count", "path_depth", "query_param_count"]


def log_transform(features):
    out = features.copy()
    for col in SKEWED_COLS:
        out[col] = np.log1p(out[col])
    return out


def load_phishing():
    """Load dataset and keep only phishing rows (label == 1)."""
    path = os.path.join(HERE, "processed_merged_urls_v2.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Dataset not found. Put processed_merged_urls_v2.csv in: {HERE}")
    df = pd.read_csv(path)
    return df[df["label"] == 1].reset_index(drop=True)


def fit():
    phishing = load_phishing()
    X = log_transform(phishing[FEATURES])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=SEED).fit(X_scaled)

    # Save scaler + model together to cluster a new URL.
    joblib.dump({"scaler": scaler, "kmeans": km}, MODEL_PATH)

    # Every phishing URL with its cluster with source
    out = phishing[["url", "source"]].copy()
    out["cluster"] = km.labels_
    out.to_csv(os.path.join(HERE, "phishing_cluster_assignments.csv"), index=False)

    # Mean scaled per cluster = standard deviations from the average.
    profile = pd.DataFrame(X_scaled, columns=FEATURES).groupby(km.labels_).mean().round(3)
    profile.insert(0, "size", np.bincount(km.labels_))
    profile.to_csv(os.path.join(HERE, "phishing_cluster_profile.csv"), index_label="cluster")

    print("Cluster sizes:\n", out["cluster"].value_counts().sort_index().to_string())
    print("\nSource mix per cluster (%):")
    print((pd.crosstab(out["cluster"], out["source"], normalize="index") * 100).round(0).to_string())
    print(f"\nSaved model to {MODEL_PATH}")


def predict(url):
    """Assign one new URL to a cluster using the saved scaler + model."""
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("No saved model yet. Run: python cluster_phishing.py")
    saved = joblib.load(MODEL_PATH)
    row = pd.DataFrame([extract_features(url)])[FEATURES]   # one-row table of the 21 features
    scaled = saved["scaler"].transform(log_transform(row))
    cluster = int(saved["kmeans"].predict(scaled)[0])
    dist = float(np.linalg.norm(scaled - saved["kmeans"].cluster_centers_[cluster]))
    print(f"URL: {url}\nCluster: {cluster}  (distance to centre: {dist:.2f})")
    return cluster


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        fit()
