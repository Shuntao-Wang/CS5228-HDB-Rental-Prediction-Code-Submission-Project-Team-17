# V5 feature cache — submitted by ruihe.
# Builds the V4 'extended' features once (optionally + geo neighbour features) so that
# model runs take seconds instead of recomputing the time-safe history aggregates.
"""Build and cache V4-extended feature matrices for the two chronological holdouts and the
full train->test run, so V5 model experiments can iterate without recomputing history features.
Run from repo root: python scripts/v5_cache_features.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v2_experiments import ROOT, DATA, SPLITS, SPLIT_END
from run_v4_experiments import build_xy
from v5_geo_features import make_geo_features
import argparse

OUT = ROOT / 'experiments' / 'v5' / 'feature_cache.pkl'


def build(ref, qry, geo):
    x, cats = build_xy(ref, qry, 'extended')
    if geo:
        g = make_geo_features(ref, qry)
        for c in g.columns:
            x[c] = g[c].to_numpy(float)
    return x, cats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--geo', action='store_true'); a = ap.parse_args()
    out = OUT.with_name('feature_cache_geo.pkl') if a.geo else OUT
    train = pd.read_csv(DATA / 'train_features.csv.gz')
    test = pd.read_csv(DATA / 'test_features.csv.gz')
    cache = {}
    for split in ['long', 'short']:
        sub = train[train.RENT_APPROVAL_DATE < SPLITS[split]].copy()
        val = train[(train.RENT_APPROVAL_DATE >= SPLITS[split]) & (train.RENT_APPROVAL_DATE <= SPLIT_END)].copy()
        xtr, cats = build(sub, sub, a.geo)
        xva, _ = build(sub, val, a.geo)
        cache[split] = dict(xtr=xtr, ytr=sub.MONTHLY_RENT.to_numpy(float), mtr=sub.MONTH_INDEX.to_numpy(),
                            xva=xva, yva=val.MONTHLY_RENT.to_numpy(float), mva=val.MONTH_INDEX.to_numpy(),
                            dva=val.RENT_APPROVAL_DATE.to_numpy(), cats=cats)
        print(split, xtr.shape, xva.shape, flush=True)
    xtr, cats = build(train, train, a.geo)
    xte, _ = build(train, test, a.geo)
    cache['full'] = dict(xtr=xtr, ytr=train.MONTHLY_RENT.to_numpy(float), mtr=train.MONTH_INDEX.to_numpy(),
                         xte=xte, mte=test.MONTH_INDEX.to_numpy(), dte=test.RENT_APPROVAL_DATE.to_numpy(), cats=cats)
    print('full', xtr.shape, xte.shape, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(cache, f)
    print('saved', out)

if __name__ == '__main__':
    main()
