"""Offline lexical URL feature extraction. Python 3.10+, pandas required."""
import argparse
import ipaddress
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

FEATURES = [
    'url_length', 'host_length', 'path_length', 'query_length',
    'dot_count', 'hyphen_count', 'at_count', 'percent_count',
    'digit_count', 'digit_ratio', 'special_char_ratio', 'subdomain_count',
    'path_depth', 'query_param_count', 'has_ip_host', 'has_https_scheme',
    'has_port', 'has_punycode', 'has_login_keyword', 'has_verify_keyword',
    'url_entropy',
]


def extract_features(value):
    """Return stable numeric features without DNS, network calls, or label access."""
    raw = str(value).strip()
    candidate = raw if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://', raw) else 'http://' + raw
    parsed = urlsplit(candidate)
    host = (parsed.hostname or '').lower().rstrip('.')
    path = parsed.path or ''
    query = parsed.query or ''
    digits = sum(c.isdigit() for c in raw)
    special = sum(not c.isalnum() for c in raw)
    counts = Counter(raw)
    entropy = -sum((n / len(raw)) * math.log2(n / len(raw)) for n in counts.values()) if raw else 0.0
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0
    labels = [part for part in host.split('.') if part]
    # Approximation: assumes final two DNS labels form the registered domain.
    subdomains = max(len(labels) - 2, 0) if not is_ip else 0
    words = re.split(r'[^a-z0-9]+', (host + path + '?' + query).lower())
    return {
        'url_length': len(raw), 'host_length': len(host), 'path_length': len(path),
        'query_length': len(query), 'dot_count': raw.count('.'),
        'hyphen_count': raw.count('-'), 'at_count': raw.count('@'),
        'percent_count': raw.count('%'), 'digit_count': digits,
        'digit_ratio': digits / len(raw) if raw else 0.0,
        'special_char_ratio': special / len(raw) if raw else 0.0,
        'subdomain_count': subdomains, 'path_depth': len([x for x in path.split('/') if x]),
        'query_param_count': len([x for x in query.split('&') if x]),
        'has_ip_host': is_ip, 'has_https_scheme': int(parsed.scheme.lower() == 'https' and raw.lower().startswith('https://')),
        'has_port': int(parsed.port is not None) if _valid_port(parsed) else 0,
        'has_punycode': int('xn--' in host),
        'has_login_keyword': int('login' in words or 'signin' in words),
        'has_verify_keyword': int('verify' in words or 'verification' in words),
        'url_entropy': entropy,
    }


def parseable_url(value):
    raw = str(value).strip()
    # Control bytes cannot be pasted as a normal browser URL and may reflect
    # corrupted source records. Exclude them before feature generation.
    if any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in raw):
        return False
    candidate = raw if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://', raw) else 'http://' + raw
    try:
        parsed = urlsplit(candidate)
        return bool(parsed.hostname)
    except ValueError:
        return False


def _valid_port(parsed):
    try:
        return parsed.port is not None
    except ValueError:
        return False


def process(input_path, output_path, url_col, label_col, mapping, filter_to_mapped=False):
    p = Path(input_path)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    for col in (url_col, label_col):
        if col not in df:
            raise ValueError(f'Missing column {col!r}; available: {list(df.columns)}')
    clean = pd.DataFrame({'url': df[url_col].str.strip(), 'original_label': df[label_col].str.strip().str.lower()})
    clean = clean[(clean.url != '') & (clean.original_label != '')].copy()
    excluded_count = 0
    if filter_to_mapped:
        included = clean.original_label.isin(mapping)
        excluded_count = int((~included).sum())
        clean = clean[included].copy()
    clean['label'] = clean.original_label.map(mapping)
    unknown = sorted(clean.loc[clean.label.isna(), 'original_label'].unique())
    if unknown:
        raise ValueError(f'Unmapped labels: {unknown}. Supply explicit --positive/--negative labels.')
    conflicting = clean.groupby('url')['label'].nunique()
    conflict_urls = set(conflicting[conflicting > 1].index)
    clean = clean[~clean.url.isin(conflict_urls)].drop_duplicates('url', keep='first').reset_index(drop=True)
    valid = clean.url.map(parseable_url)
    invalid_count = int((~valid).sum())
    clean = clean[valid].reset_index(drop=True)
    features = pd.DataFrame([extract_features(u) for u in clean.url], columns=FEATURES)
    result = pd.concat([clean[['url', 'label']], features], axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f'Rows: {len(result)}; excluded other classes: {excluded_count}; removed conflicting URLs: {len(conflict_urls)}; invalid/unparseable URLs: {invalid_count}; labels: {result.label.value_counts().to_dict()}')
    print(f'Output: {output_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='processed_urls.csv')
    ap.add_argument('--url-column', default='url')
    ap.add_argument('--label-column', default='type')
    ap.add_argument('--positive', nargs='+', default=['phishing'], help='Raw labels mapped to 1')
    ap.add_argument('--negative', nargs='+', default=['benign'], help='Raw labels mapped to 0')
    ap.add_argument('--filter-to-mapped-labels', action='store_true', help='Explicitly exclude other classes before processing')
    args = ap.parse_args()
    mapping = {str(x).strip().lower(): 1 for x in args.positive}
    for x in args.negative:
        key = str(x).strip().lower()
        if key in mapping:
            ap.error(f'Label {key!r} appears in both classes')
        mapping[key] = 0
    process(args.input, args.output, args.url_column, args.label_column, mapping, args.filter_to_mapped_labels)


if __name__ == '__main__':
    main()
