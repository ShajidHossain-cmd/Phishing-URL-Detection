# Data Collection and Processing — report draft

Adapt this text to the team's final models and cite all sources in the chosen Harvard style. Counts and claims below refer to the current merged data version.

## Data Collection

The project targets people who want to assess a suspicious link before opening it. The planned application performs URL-only, offline analysis. We collected two open datasets with raw URL strings and phishing labels. The basic Kaggle *Malicious URLs dataset* provides `url` and `type` and contains 651,191 records in four categories: 428,103 benign, 94,111 phishing, 96,457 defacement and 32,520 malware. We retained only benign and phishing (522,214 records) to keep the class definition aligned with phishing detection; the other two threats should not be relabelled as phishing. [Kaggle source](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset).

The additional UCI *PhiUSIIL Phishing URL (Website)* dataset contains 235,795 records: 134,850 legitimate and 100,945 phishing. It has a different schema (`URL`, `label`) and a reversed label convention: source label 1 denotes legitimate, while 0 denotes phishing. Although the source also supplies precomputed and webpage-derived fields, we deliberately used only the raw URL and label. The web application cannot obtain page content because its workflow must be offline. The second source improves coverage and creates a useful external-source evaluation opportunity, while introducing collection differences that must be measured. [UCI source](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).

Both files were downloaded from their public dataset pages on 29 September 2026. The team did not collect live URLs or independently verify the inherited labels. No timestamp was present in the two-column Kaggle source, so a genuine chronological split cannot be constructed from this version.

## Data Processing and Feature Engineering

We harmonised the two schemas to `url`, `label` and `source`, mapping benign/legitimate to 0 and phishing to 1. After concatenation there were 758,009 selected records. We stripped outer whitespace, removed all URLs appearing with contradictory labels (six distinct URL strings), kept one occurrence of each exact duplicate URL, and removed 18 unparseable URLs and 84 URLs containing control characters after deduplication. The final dataset has 757,375 unique URL strings: 562,921 benign and 194,454 phishing. The `source` field supports auditing and external evaluation; it is excluded from the model input.

A common offline parser turns each raw URL into 21 numerical lexical and structural features. It temporarily adds an HTTP prefix to schemeless input for parsing without changing the saved original URL. Features include component lengths and depth, punctuation/encoding counts, digit and special-character ratios, literal IP hostname, Punycode marker, login/verification terms and character entropy. These capture possible address complexity, obfuscation or credential-themed lures identified in URL-based phishing research (Tamal et al., 2024; Prasad and Chandra, 2024). The indicators are not guarantees of maliciousness: legitimate applications can use long queries, HTTPS and internationalised domains. The approximate subdomain counter does not recognise multi-label public suffixes. We do not claim visual look-alike character detection or TLD reputation assessment.

No missing values remain in the final file. We do not normalise features before splitting: a model needing scaling must fit its scaler only on training data. The `url`, `source` and `label` columns are excluded from the input feature matrix. The exact same `extract_features` function can generate features from a pasted URL at inference time, preventing a train/serve mismatch.

## EDA findings and limitation

Class counts by source and distributions of URL length and entropy are supplied as separate reproducible plots. A key limitation is a strong source-format artifact: the mean explicit-HTTPS indicator among benign Kaggle URLs is 0.005, versus 1.000 among benign UCI URLs. Mean benign URL lengths also differ substantially (57.68 versus 27.23 characters). Thus a model may infer the dataset's origin instead of general phishing patterns. Random row splitting alone can hide this failure. The classification teammate should report a held-out-source evaluation and inspect false positives/negatives for the missing-scheme pattern. These observations describe the data; they do not claim model performance.

### References to format in Harvard style

- Kaggle (n.d.) *Malicious URLs dataset*. Available at: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset (Accessed: 29 September 2026).
- Prasad, A. and Chandra, S. (2024) *PhiUSIIL Phishing URL (Website)*. UCI Machine Learning Repository. Available at: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset (Accessed: 29 September 2026).
- Tamal, M.A. et al. (2024) 'Dataset of suspicious phishing URL detection', *Frontiers in Computer Science*. Available at: https://doi.org/10.3389/fcomp.2024.1308634 . Verify the full author list and bibliography details before submission.
