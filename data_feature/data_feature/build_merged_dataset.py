"""Build offline phishing-vs-benign dataset from Kaggle and UCI source CSVs."""
import argparse
from pathlib import Path
import pandas as pd
from feature_engineering import FEATURES, extract_features, parseable_url


def load_sources(kaggle, uci):
    k = pd.read_csv(kaggle, usecols=['url', 'type'], dtype=str, keep_default_na=False)
    k['type'] = k.type.str.strip().str.lower()
    k = k[k.type.isin(['benign', 'phishing'])].copy()
    k['label'] = k.type.map({'benign': 0, 'phishing': 1})
    k = k[['url', 'label']].assign(source='kaggle_sid321axn')
    u = pd.read_csv(uci, usecols=['URL', 'label'], dtype=str, keep_default_na=False)
    u.columns = ['url', 'original_label']
    u['original_label'] = u.original_label.str.strip()
    if not u.original_label.isin(['0', '1']).all():
        raise ValueError('UCI label contains unexpected values')
    # UCI PhiUSIIL: 1 = legitimate, 0 = phishing. Invert to positive=phishing.
    u['label'] = u.original_label.map({'1': 0, '0': 1})
    u = u[['url', 'label']].assign(source='uci_phiusiil')
    return k, u


def build(kaggle, uci, output, audit):
    k, u = load_sources(kaggle, uci)
    counts = {'kaggle_selected': len(k), 'uci_selected': len(u)}
    d = pd.concat([k, u], ignore_index=True)
    d.url = d.url.str.strip()
    d = d[d.url.ne('')].copy()
    counts['nonempty'] = len(d)
    labels_per_url = d.groupby('url', sort=False).label.nunique()
    conflict_urls = set(labels_per_url[labels_per_url > 1].index)
    counts['conflicting_distinct_urls'] = len(conflict_urls)
    d = d[~d.url.isin(conflict_urls)].copy()
    # Keep Kaggle first when exact URL occurs in both sources; source is provenance only.
    d = d.drop_duplicates('url', keep='first').reset_index(drop=True)
    counts['after_dedup'] = len(d)
    valid = d.url.map(parseable_url)
    counts['invalid_urls'] = int((~valid).sum())
    d = d[valid].reset_index(drop=True)
    features = pd.DataFrame([extract_features(x) for x in d.url], columns=FEATURES)
    result = pd.concat([d[['url', 'label', 'source']], features], axis=1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    counts['final'] = len(result)
    for (source, label), n in result.groupby(['source', 'label']).size().items():
        counts[f'{source}_label_{label}'] = int(n)
    pd.DataFrame([counts]).to_csv(audit, index=False)
    print(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kaggle', required=True, help='malicious_phish.csv')
    parser.add_argument('--uci', required=True, help='PhiUSIIL_Phishing_URL_Dataset.csv')
    parser.add_argument('--output', default='processed_merged_urls.csv')
    parser.add_argument('--audit', default='processing_audit.csv')
    a = parser.parse_args()
    build(a.kaggle, a.uci, a.output, a.audit)
