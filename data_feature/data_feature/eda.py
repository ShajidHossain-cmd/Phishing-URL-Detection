"""Generate reproducible descriptive statistics and figures; no model training."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from feature_engineering import FEATURES


def run(path, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(path)
    assert set(d.label.unique()) == {0, 1}
    assert not d.isna().any().any()
    d.groupby(['source', 'label']).size().rename('count').to_csv(out/'class_by_source.csv')
    d.groupby(['source', 'label'])[FEATURES].agg(['mean', 'median']).to_csv(out/'feature_summary_by_source_label.csv')
    d[FEATURES].describe(percentiles=[.5,.9,.99]).T.to_csv(out/'overall_feature_summary.csv')
    groups = d.groupby(['source', 'label']).size().unstack(fill_value=0).rename(columns={0:'benign',1:'phishing'})
    ax = groups.plot.bar(rot=0, figsize=(8,4), color=['#2d79a8','#db7045'])
    ax.set(xlabel='Source',ylabel='Unique URLs',title='Class counts by dataset source')
    plt.tight_layout(); plt.savefig(out/'class_by_source.png',dpi=180); plt.close()
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for i, col in enumerate(['url_length','url_entropy']):
        capped = d[col].clip(upper=d[col].quantile(.99)) if col=='url_length' else d[col]
        axes[i].hist([capped[d.label==0],capped[d.label==1]],bins=45,label=['benign','phishing'],alpha=.8)
        axes[i].set(xlabel=col,ylabel='Count',title=col+' distribution')
        axes[i].legend()
    fig.tight_layout(); fig.savefig(out/'feature_distributions.png',dpi=180); plt.close(fig)
    rates = d.groupby(['source','label'])[['has_https_scheme','has_ip_host','has_punycode']].mean()
    rates.to_csv(out/'binary_feature_rates.csv')
    print('Rows:',len(d),'Labels:',d.label.value_counts().to_dict(),'Missing:',int(d.isna().sum().sum()))
    print('Source/label counts:\n',groups.to_string())
    print('Selected means:\n',d.groupby(['source','label'])[['url_length','url_entropy','has_https_scheme']].mean().round(3).to_string())


if __name__ == '__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',default='processed_merged_urls.csv')
    a.add_argument('--outdir',default='eda')
    x=a.parse_args();run(x.input,x.outdir)
