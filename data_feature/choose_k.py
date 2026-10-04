"""

Inputs : processed_merged_urls_v2.csv (or .csv.gz) from teammate 1
Outputs: k_selection_results.csv and k_selection.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets the script save plots without needing a display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from feature_engineering import FEATURES

SEED = 42            # fixed so anyone re-running gets the same numbers
K_RANGE = range(2, 11)
SIL_SAMPLE = 15000   # silhouette is slow on 194k rows, so score a random sample

# load the data and keep only the phishing class
HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "processed_merged_urls_v2.csv")
# path = "processed_merged_urls_v2.csv"
# if not os.path.exists(path):
#     path = "processed_merged_urls_v2.csv"
df = pd.read_csv(path)
phishing = df[df["label"] == 1].copy()  # label is only used for this filter, never for clustering
X = phishing[FEATURES].copy()           # url/label/source are left out of the feature matrix
print("Clustering input:", X.shape)

# log-transform the skewed length/count columns
skewed_cols = ["url_length", "host_length", "path_length", "query_length",
               "dot_count", "hyphen_count", "at_count", "percent_count",
               "digit_count", "subdomain_count", "path_depth", "query_param_count"]
for col in skewed_cols:
    X[col] = np.log1p(X[col])

# ---- Step 3: standardise so every feature has mean 0 and std 1 ----
X_scaled = StandardScaler().fit_transform(X)

# ---- Step 4: run k-means for each k and record both measures ----
rng = np.random.RandomState(SEED)
sample_idx = rng.choice(len(X_scaled), SIL_SAMPLE, replace=False)

rows = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(X_scaled)
    sil = silhouette_score(X_scaled[sample_idx], km.labels_[sample_idx])
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    print(f"k={k:2d}  inertia={km.inertia_:10.0f}  silhouette={sil:.3f}")

results = pd.DataFrame(rows)
results.to_csv("k_selection_results.csv", index=False)

#plot both curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(results["k"], results["inertia"], marker="o")
ax[0].set_title("Elbow: inertia vs k")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(results["k"], results["silhouette"], marker="o", color="tab:red")
ax[1].set_title(f"Silhouette vs k ({SIL_SAMPLE:,}-row sample)")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig("k_selection.png", dpi=150)
print("Saved k_selection_results.csv and k_selection.png")
