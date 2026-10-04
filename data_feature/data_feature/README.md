# URLGuard — Data & Feature Engineering handoff

**Final shared dataset:** `processed_merged_urls_v2.csv` (757,375 unique URLs; `label=1` phishing, `label=0` benign). It has `url`, `label`, `source` and 21 numeric features. Model input is **only** the columns in `FEATURES` in `feature_engineering.py`. `url` is for audit/error analysis; `source` is for source-aware evaluation, never a training feature. The website can compute the same features with `extract_features(url)` entirely offline. The file contains no destination page content, UCI's precomputed webpage features or source-derived popularity statistics.

## Raw sources and schema

1. Kaggle, *Malicious URLs dataset*, `malicious_phish.csv`: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset . Source columns `url,type`; keep only `benign` and `phishing`; exclude `malware` and `defacement` (128,977 rows) because the agreed task is phishing detection.
2. UCI Machine Learning Repository, *PhiUSIIL Phishing URL (Website)*, `PhiUSIIL_Phishing_URL_Dataset.csv`: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset . Source columns `URL,label`; its **1 means legitimate and 0 means phishing**. Reverse this convention to the shared `0=benign, 1=phishing`. Ignore all UCI precomputed and webpage features so every feature can be extracted from a pasted URL at prediction time.

Downloaded 29 September 2026. Cite **both** sources in the submitted report. Labels are inherited from source datasets, not independently verified by our team.

## Reproduce from original CSVs

```bash
conda create -n urlguard python=3.11 pandas matplotlib scikit-learn -y
conda activate urlguard
python build_merged_dataset.py --kaggle malicious_phish.csv --uci PhiUSIIL_Phishing_URL_Dataset.csv --output processed_merged_urls_v2.csv --audit processing_audit_v2.csv
python eda.py --input processed_merged_urls_v2.csv --outdir eda_v2
```

The two raw CSVs must be downloaded separately from the cited sources. `feature_engineering.py` contains the shared offline parser/extractor. `build_merged_dataset.py` is the **two-source reproduction entry point**. The older one-source `processed_urls.csv` and previous merged CSV are superseded by `processed_merged_urls_v2.csv`.

## Cleaning and joining decisions

- Select Kaggle phishing/benign and UCI's raw URL/label only; trim outer whitespace and unify column names and binary labels.
- Concatenate rows; discard empty URLs, all URLs with contradictory labels, and duplicate exact URL strings (Kaggle row retained if a repeated URL occurs in both). Discard unparseable URLs without a hostname and URLs containing control characters. Do not lowercase, unquote or rewrite raw URL strings because case and encoded characters carry signals.
- A scheme-less URL is temporarily prefixed with `http://` **for parsing only**; `has_https_scheme` checks the original text. This is a source-dependent missing-scheme indicator and needs cautious interpretation.
- Generate all 21 features without inspecting labels. No global scaling, imputation, feature selection, or training was performed. For scaling, fit a scikit-learn Pipeline on training rows only.
- Exact URL deduplication does not remove shared domains or near duplicates; avoid overly optimistic random-split claims. Compare a domain-grouped split and a held-out-source test. Any exact URL overlap should be removed **before** the split.

| Audit | Count |
| --- | ---: |
| Kaggle selected phishing/benign | 522,214 |
| UCI selected | 235,795 |
| Combined before cleaning | 758,009 |
| Distinct URLs with conflicting labels | 6 |
| After conflict removal and exact-URL deduplication | 757,477 |
| Invalid/unparseable/control-character URLs removed | 102 (18 parse failures and 84 control-character URLs) |
| **Final unique URLs** | **757,375** |
| Final benign / phishing | 562,921 / 194,454 |

See `processing_audit_v2.csv` for per-source labels. `eda_v2/class_by_source.csv` and `eda_v2/feature_summary_by_source_label.csv` show the EDA details.

## Feature dictionary

| Columns | Definition and motivation |
| --- | --- |
| `url_length`, `host_length`, `path_length`, `query_length` | Character lengths of full input and parsed components; describe URL complexity. |
| `dot_count`, `hyphen_count`, `at_count`, `percent_count` | Counts of literal delimiters, separators and percent signs; possible nesting or encoding signals. |
| `digit_count`, `digit_ratio`, `special_char_ratio` | Number or proportion of digits/non-alphanumeric characters; possible opaque or generated strings. Ratios use original URL length. |
| `subdomain_count` | Approximate DNS labels beyond final two, zero for IP literals; public suffixes such as `.co.uk` are **not** handled exactly. |
| `path_depth`, `query_param_count` | Number of nonempty slash-separated path parts and ampersand-separated query fields. |
| `has_ip_host`, `has_port` | Whether hostname is an IPv4/IPv6 literal or a valid explicit port is present. |
| `has_https_scheme` | Original text explicitly begins `https://`; this is not proof of safety and is strongly affected by source formatting. |
| `has_punycode` | Hostname contains `xn--`, a possible internationalised-name encoding; benign names can too. |
| `has_login_keyword`, `has_verify_keyword` | Host/path/query tokens contain login/signin or verify/verification respectively; possible credential-themed lure. |
| `url_entropy` | Shannon entropy (bits per character) of the original URL, a crude unpredictability signal. |

All features are **hypotheses**, not independent security verdicts. No true visual confusable-character detection or verified unusual-TLD reputation feature is implemented. Avoid claiming those functions in the report/UI. Sources motivating lexical feature categories: Tamal et al. (2024), https://doi.org/10.3389/fcomp.2024.1308634 ; and the original PhiUSIIL dataset description by Prasad and Chandra (2024), https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset .

## EDA and handoff to model teammates

Source-specific formatting is a major confounder: Kaggle benign URLs have mean `has_https_scheme=0.005`, versus UCI benign URLs `1.000`. The pooled `https` rate therefore reflects collection format and must **not** be described as phishing behavior. Kaggle benign mean URL length is 57.68, UCI benign 27.23. See plots and summary CSVs in `eda_v2/`. These descriptive statistics are not evaluation results.

```python
import pandas as pd
from feature_engineering import FEATURES, extract_features

df = pd.read_csv('processed_merged_urls_v2.csv')
X = df[FEATURES]
y = df['label'].astype(int)
source = df['source']  # group/holdout analysis only
single_url_features = extract_features('https://example.org/login')
```

Classification teammate: preserve source and URL alongside predictions for false positive/negative review; compare models on the same test set, report precision/recall/F1/confusion matrix and stress tests. In a strict cross-source test, use one source for training and the other for testing, and report both directions if feasible. Clustering teammate: filter to `label==1` first, then drop `label`, `source`, and `url` from the clustering feature matrix; profile each cluster against the phishing-class overall mean and show representative URL examples safely as text.

The handoff includes **data engineering and EDA only**. The final assignment ZIP still needs the team's classifier, clusterer, saved models, prediction command, integrated README, report and meeting/contribution PDFs.
