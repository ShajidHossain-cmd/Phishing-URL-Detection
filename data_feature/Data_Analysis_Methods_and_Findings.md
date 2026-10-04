# Data Analysis — methods and findings for report integration

Use this as a draft, shorten it to fit the team's 4,000-word report. All values come from `processed_merged_urls_v2.csv`, and the supporting tables and plots are in `eda_extended/`. The script `eda_extended.py` regenerates them. This is descriptive EDA, not a trained model or a causal test.

## Method

We first audited the processed file: 757,375 unique URLs, no null cells, and binary labels (`0=benign`, `1=phishing`). We counted rows in each class and each source, then calculated each class's percentage **within its source**. This matters because Kaggle contributes 522,058 URLs (82.0% benign) while UCI contributes 235,317 (57.3% benign). A pooled summary alone would give the larger Kaggle source more weight.

For each of 21 numeric features, we calculated the mean and median by `source × label`. For URL length, path length and entropy, we also calculated the 10th, 50th, 90th and 99th percentiles. Medians and upper quantiles show typical values and long tails when a mean is affected by unusually long addresses. For binary indicators, we reported the fraction equal to 1 in each group. We additionally derived two descriptive flags from the saved URL: whether an explicit scheme (`http://` or `https://`) is present and whether the path is empty. These flags help audit the input format and are **not** extra training features.

We compared phishing and benign values **inside each dataset source** before interpreting pooled differences. The figures show class composition, URL/path lengths by source and label, and format-related rates. We used correlations between numeric features to identify redundancy, rather than treating every extracted column as independent evidence. For example, Spearman rank correlation is approximately 0.998 for query length versus query parameter count, 0.981 for digit count versus digit ratio, and 0.904 for path length versus path depth. Such dependence can matter when interpreting coefficients or feature importance; it does not mean the features must automatically be removed.

## Findings and interpretation

1. **The classes and sources differ in size.** The final data has 562,921 benign URLs (74.3%) and 194,454 phishing URLs (25.7%). Kaggle has 428,071 benign and 93,987 phishing URLs; UCI has 134,850 benign and 100,467 phishing URLs. Accuracy alone could conceal missed phishing cases, so the classification teammate should report phishing precision, recall, F1 and a confusion matrix by source as well as overall.

2. **URL length changes direction across sources.** In Kaggle, benign URLs average 57.68 characters and phishing URLs 45.78 (medians 46 and 35). In UCI, benign URLs average 27.23 and phishing URLs 46.22 (medians 27 and 34). The pooled means (benign 50.38, phishing 46.01) hide the UCI relationship. A universal rule that a longer URL is phishing would therefore be misleading for these data.

3. **URL structure reveals a collection artifact.** Every UCI benign row has an empty path and an explicit HTTPS scheme. In contrast, only 0.066% of Kaggle benign rows have an empty path, 8.265% have an explicit scheme, and 0.456% explicitly use HTTPS. UCI phishing URLs also all have an explicit scheme, but only 48.65% use HTTPS and 40.33% have an empty path. These differences may arise from how the source datasets were assembled; they should not be presented as general properties of safe or phishing links.

4. **Some signals may be useful but are not decisive.** Phishing URLs have longer hostnames on average in both sources (Kaggle: 19.30 versus 16.77 characters; UCI: 24.44 versus 19.23). IP-address hosts are uncommon, even among phishing URLs (0.35% in Kaggle, 0.60% in UCI). A rule based only on IP hosts would miss most phishing cases. Character entropy changes direction by source: Kaggle phishing mean 4.009 versus benign 4.210, while UCI phishing mean 4.105 versus benign 3.853.

## Implication for modelling and evaluation

The EDA supports comparing a simple linear baseline with a model able to handle interactions, but it **does not prove** either will perform well. Source formatting can be predictive of both class and source, so a random row split may overstate real-world generalisation. Keep `source` outside the training features, report results separately for Kaggle and UCI, and test a model trained on one source against the other where feasible. A group or domain-aware split can also reduce near-duplicate domain leakage. Inspect false positives and false negatives in the unusual groups, especially short HTTPS phishing URLs and benign URLs with long paths or queries. Any scaler, feature selector or threshold must be fitted/tuned only on training/validation data.

## Reproduction

```bash
python eda_extended.py --input processed_merged_urls_v2.csv --outdir eda_extended
```

Outputs: `class_counts.csv`, `numeric_feature_summary.csv`, `structure_rates.csv`, `feature_quantiles.csv`, `class_mean_differences.csv`, and three labelled PNG figures. The original v2 dataset and its 21 training features have not changed.
