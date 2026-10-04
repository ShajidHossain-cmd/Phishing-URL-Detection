"""Source-stratified EDA for the v2 URL dataset; no model is fitted."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES = ['url_length','host_length','path_length','query_length','digit_ratio',
            'special_char_ratio','subdomain_count','url_entropy','has_https_scheme',
            'has_ip_host','has_login_keyword','has_verify_keyword']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='processed_merged_urls_v2.csv')
    ap.add_argument('--outdir',default='eda_extended')
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input)
    required={'url','source','label',*FEATURES}
    if not required.issubset(d): raise ValueError(f'Missing columns: {required-set(d)}')
    d['label_name']=d.label.map({0:'benign',1:'phishing'})
    d['source_name']=d.source.map({'kaggle_sid321axn':'Kaggle','uci_phiusiil':'UCI PhiUSIIL'})
    d['has_explicit_scheme']=d.url.str.match(r'^[A-Za-z][A-Za-z0-9+.-]*://').astype(int)
    d['zero_path']=(d.path_length==0).astype(int)
    d['long_url_100']=(d.url_length>=100).astype(int)
    d['has_query']=(d.query_length>0).astype(int)
    d['has_digit']=(d.digit_ratio>0).astype(int)
    counts=d.groupby(['source_name','label_name']).size().rename('count').reset_index()
    counts['within_source_pct']=100*counts['count']/counts.groupby('source_name')['count'].transform('sum')
    counts.to_csv(out/'class_counts.csv',index=False)
    summary=d.groupby(['source_name','label_name'])[FEATURES].agg(['mean','median']).round(5)
    summary.to_csv(out/'numeric_feature_summary.csv')
    rates=d.groupby(['source_name','label_name'])[['has_explicit_scheme','has_https_scheme','zero_path','has_query','has_digit','long_url_100','has_ip_host','has_login_keyword','has_verify_keyword']].mean().round(5)
    rates.to_csv(out/'structure_rates.csv')
    q=d.groupby(['source_name','label_name'])[['url_length','path_length','url_entropy']].quantile([.1,.5,.9,.99]).round(4)
    q.to_csv(out/'feature_quantiles.csv')
    # Pooled and source-specific class differences show reversals hidden by aggregation.
    rows=[]
    for source,group in [('All',d),*list(d.groupby('source_name'))]:
        means=group.groupby('label')[FEATURES].mean()
        for f in FEATURES:
            rows.append({'source':source,'feature':f,'benign_mean':means.loc[0,f],
                         'phishing_mean':means.loc[1,f],
                         'phishing_minus_benign':means.loc[1,f]-means.loc[0,f]})
    pd.DataFrame(rows).round(5).to_csv(out/'class_mean_differences.csv',index=False)
    # Figure 1: source composition matters before interpreting any pooled mean.
    pivot=counts.pivot(index='source_name',columns='label_name',values='count')[['benign','phishing']]
    ax=pivot.plot.bar(figsize=(7.3,4),rot=0,color=['#2878a8','#d66a40'])
    ax.set(xlabel='',ylabel='Unique URL count',title='Class composition varies by source')
    ax.legend(title='Label');plt.tight_layout();plt.savefig(out/'class_composition.png',dpi=180);plt.close()
    # Figure 2: show class directions within each source with means and medians.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,col,title in zip(axes,['url_length','path_length'],['Mean URL length','Mean path length']):
        p=d.groupby(['source_name','label_name'])[col].mean().unstack()[['benign','phishing']]
        p.plot.bar(ax=ax,rot=0,color=['#2878a8','#d66a40'],legend=False)
        ax.set(xlabel='',ylabel='Characters',title=title)
    axes[1].legend(title='Label');fig.tight_layout();fig.savefig(out/'length_by_source_and_label.png',dpi=180);plt.close(fig)
    # Figure 3: source format artifact; rates are not causal phishing indicators.
    p=100*rates[['has_explicit_scheme','has_https_scheme','zero_path']]
    ax=p.plot.bar(figsize=(10,4.5),rot=0)
    ax.set(xlabel='Source and label',ylabel='Share of URLs (%)',title='URL format differs sharply by dataset source',ylim=(0,105))
    plt.tight_layout();plt.savefig(out/'format_artifacts.png',dpi=180);plt.close()
    print(counts.to_string(index=False))
    print(rates[['has_explicit_scheme','has_https_scheme','zero_path']].to_string())

if __name__=='__main__': main()
