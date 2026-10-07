# URLGuard: Phishing URL Detection (COS30049 A2)

## 1. Environment (conda)

```bash
conda create -n urlguard python=3.11 pandas numpy matplotlib scikit-learn -y
conda activate urlguard
```

## 2. Data processing

The prepared dataset is `data_feature/processed_merged_urls_v2.csv` (757,375 URLs, `label` 1 = phishing, 21 features listed in `FEATURES` in `feature_engineering.py`).

Load it for further processing:

```python
import pandas as pd
from feature_engineering import FEATURES      # run inside data_feature/
df = pd.read_csv("processed_merged_urls_v2.csv")
X, y = df[FEATURES], df["label"]
```

To rebuild it from the raw Kaggle and UCI CSVs (optional):

```bash
cd data_feature
python build_merged_dataset.py --kaggle malicious_phish.csv --uci PhiUSIIL_Phishing_URL_Dataset.csv --output processed_merged_urls_v2.csv --audit processing_audit_v2.csv
```

## 3. Train the models

Classification (run from the project root):

```bash
python classification.py
```

Clustering of phishing URLs, k-means with k = 4 (run from `data_feature/`):

```bash
cd data_feature
python choose_k.py          # choose k (elbow + silhouette)
python profile_clusters.py  # describe the clusters
python my_clustering.py     # fit and save phishing_kmeans.joblib
```

## 4. Prediction

Cluster type for a URL (run from `data_feature/`):

```bash
python my_clustering.py "http://www.getelectrum.com"
```
