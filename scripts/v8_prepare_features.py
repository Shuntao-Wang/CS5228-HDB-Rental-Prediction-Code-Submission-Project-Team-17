"""Prepare compact V8 feature-family caches on top of unchanged V5 cache."""
from __future__ import annotations
import json, pickle
import pandas as pd
from run_v2_experiments import ROOT, DATA, SPLITS, SPLIT_END
from v8_features import shape_features, premium_features, trend_features, SHAPE_CATS

BASE=ROOT/'experiments/v5/feature_cache.pkl'
OUT=ROOT/'experiments/v8'

def make_extras(x, reference, query, cutoff):
    joined=pd.concat([shape_features(x),premium_features(x),
                      trend_features(reference,query,cutoff_month_index=cutoff)],axis=1)
    if len(joined)!=len(x) or joined.columns.duplicated().any():
        raise AssertionError('V8 feature alignment failed')
    return joined

def write(split,base,reference,query,cutoff,target):
    print(f'V8 preparing {split}: train={len(reference):,}, query={len(query):,}',flush=True)
    xtr=make_extras(base['xtr'],reference,reference,cutoff)
    xva=make_extras(base[target],reference,query,cutoff)
    if xtr.columns.tolist()!=xva.columns.tolist():
        raise AssertionError('V8 feature column misalignment')
    payload={'xtr':xtr,target:xva}
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/f'extras_{split}.pkl'; temp=dest.with_suffix('.pkl.tmp')
    with temp.open('wb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
    temp.replace(dest)
    print(f'Saved {dest.name}: {len(xtr.columns)} extra features, {dest.stat().st_size/1e6:.1f} MB',flush=True)

def main():
    if not BASE.is_file(): raise FileNotFoundError('Run python scripts/v5_cache_features.py first')
    with BASE.open('rb') as f:base=pickle.load(f)
    tr=pd.read_csv(DATA/'train_features.csv.gz');te=pd.read_csv(DATA/'test_features.csv.gz')
    for split in ('long','short'):
        ref=tr[tr.RENT_APPROVAL_DATE<SPLITS[split]].copy()
        val=tr[(tr.RENT_APPROVAL_DATE>=SPLITS[split])&(tr.RENT_APPROVAL_DATE<=SPLIT_END)].copy()
        write(split,base[split],ref,val,int(val.MONTH_INDEX.min()),'xva')
    write('full',base['full'],tr,te,int(tr.MONTH_INDEX.max())+1,'xte')
    (OUT/'feature_metadata.json').write_text(json.dumps({
        'source':'V5 and prior-month original training labels',
        'external_data':False,'test_labels':False,'base_features':47,
        'shape_cat_features':SHAPE_CATS},indent=2)+'\n',encoding='utf-8')
    print('V8 cache preparation complete.',flush=True)

if __name__=='__main__':main()
