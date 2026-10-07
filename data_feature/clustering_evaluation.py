import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from feature_engineering import FEATURES

# Load dataset
df = pd.read_csv("processed_merged_urls_v2.csv")

# Keep phishing URLs only
phishing = df[df["label"] == 1]

# Select clustering features
X = phishing[FEATURES].copy()

# Features that need log transformation
SKEWED_COLS = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "at_count",
    "percent_count",
    "digit_count",
    "subdomain_count",
    "path_depth",
    "query_param_count"
]

# Log transformation
for col in SKEWED_COLS:
    X[col] = np.log1p(X[col])

# Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 7 - K=4 evaluation
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)

score = silhouette_score(
    X_scaled,
    cluster_labels,
    sample_size=10000,
    random_state=42
)

print("Silhouette score:", score)

# STEP 8 - Stability evaluation
seeds = [1, 7, 21, 99]

for random_seed in seeds:
    test_model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=random_seed
    )

    test_labels = test_model.fit_predict(X_scaled)

    stability = adjusted_rand_score(
        cluster_labels,
        test_labels
    )

    print("Seed:", random_seed, "ARI:", stability) 