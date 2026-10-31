# V7 features — detrended (market-relative) history features.
"""Build V7 feature matrices for the two chronological holdouts and the full train->test run.

Idea: rents rose ~50% between 2021 and 2023, so V2-V4 cumulative means since 2021 mix price levels.
V7 divides every label by the monthly market level L(m) of the reference fold and computes
group statistics of that *ratio* (optionally exponentially time-decayed). Models are then trained
on the ratio target y / L(m); predictions are multiplied by the level forecast for the query month.

Leakage rules (same as V2-V5): a query row from month M only sees reference labels from months < M;
validation features only use the training fold's labels. L(m) of the training fold is used only for
the target transform and for normalising past labels; for query months after the fold it is the
persistence forecast (mean level of the last LEVEL_WINDOW months).

Run from repo root: python scripts/v7_features.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v2_experiments import ROOT, DATA, SPLITS, SPLIT_END, NUM_COLS, CAT_COLS

OUT = ROOT / 'experiments' / 'v7' / 'feature_cache.pkl'
LEVEL_WINDOW = 3
TAUS = [None, 12.0]       # None = plain cumulative mean, 12 = exp decay with 12-month time constant
SMOOTH = 10.0              # pseudo-count shrinkage towards the parent group's ratio
# name: (keys, parent name or None for global ratio 1.0)
GROUPS = {
    'type': (['FLAT_TYPE'], None),
    'type_area': (['FLAT_TYPE', '_AREA_BAND'], 'type'),
    'town_type': (['TOWN', 'FLAT_TYPE'], 'type'),
    'town_type_area': (['TOWN', 'FLAT_TYPE', '_AREA_BAND'], 'town_type'),
    'pa_type_model': (['PLANNING_AREA', 'FLAT_TYPE', 'FLAT_MODEL'], 'town_type'),
    'street_type': (['TOWN', 'STREET', 'FLAT_TYPE'], 'town_type'),
    'street_type_area': (['TOWN', 'STREET', 'FLAT_TYPE', '_AREA_BAND'], 'street_type'),
    'block': (['BLOCK_KEY'], None),
    'block_type': (['BLOCK_KEY', 'FLAT_TYPE'], 'street_type'),
    'block_type_area': (['BLOCK_KEY', 'FLAT_TYPE', '_AREA_BAND'], 'block_type'),
    'block_type_model_area': (['BLOCK_KEY', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM'], 'block_type_area'),
}


def market_level(ref: pd.DataFrame) -> pd.Series:
    """Monthly mean rent of the reference fold, indexed by MONTH_INDEX."""
    return ref.groupby('MONTH_INDEX')['MONTHLY_RENT'].mean().sort_index()


def level_for(months: np.ndarray, level: pd.Series, window: int = LEVEL_WINDOW) -> np.ndarray:
    """L(m) for in-fold months; persistence forecast (mean of last `window` months) after the fold."""
    last = level.index.max()
    fut = level.loc[level.index > last - window].mean()
    out = pd.Series(months).map(level).to_numpy(float)
    out[months > last] = fut
    if np.isnan(out).any():
        raise ValueError('Month without market level inside the reference fold')
    return out


def _asof_sums(ref: pd.DataFrame, qry: pd.DataFrame, by: list[str], vals: list[str]) -> np.ndarray:
    """Per query row, sums of `vals` over reference rows of the same group with month < query month."""
    g = ref.groupby(by + ['MONTH_INDEX'], sort=True, observed=True, dropna=False)[vals].sum().reset_index()
    g[vals] = g.groupby(by, sort=False, observed=True, dropna=False)[vals].cumsum()
    g = g.sort_values('MONTH_INDEX', kind='stable')
    look = qry[by + ['MONTH_INDEX']].copy()
    look['_i'] = np.arange(len(qry))
    look = look.sort_values('MONTH_INDEX', kind='stable')
    m = pd.merge_asof(look, g, on='MONTH_INDEX', by=by, direction='backward', allow_exact_matches=False)
    m = m.sort_values('_i', kind='stable')
    return np.nan_to_num(m[vals].to_numpy(float), nan=0.0)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in CAT_COLS:
        d[c] = d[c].fillna('__missing__').astype(str)
    d['_AREA_BAND'] = np.floor(d['FLOOR_AREA_SQM'] / 10.0).astype('int32')
    return d


def ratio_features(reference: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    ref, qry = _prep(reference), _prep(query)
    level = market_level(ref)
    ref['_r'] = ref['MONTHLY_RENT'].to_numpy(float) / level_for(ref['MONTH_INDEX'].to_numpy(), level)
    ref['_one'] = 1.0
    mq = qry['MONTH_INDEX'].to_numpy(float)
    feats = {}
    for tau in TAUS:
        sfx = '' if tau is None else f'_D{int(tau)}'
        if tau is None:
            ref['_w'] = 1.0
        else:
            # exp(m/tau) weights: the exp(-M/tau) factor cancels in means, restored for the decayed count.
            ref['_w'] = np.exp(ref['MONTH_INDEX'].to_numpy(float) / tau)
        ref['_wr'] = ref['_w'] * ref['_r']
        means = {}
        for name, (keys, parent) in GROUPS.items():
            sw, swr, cnt = _asof_sums(ref, qry, keys, ['_w', '_wr', '_one']).T
            prior = np.ones(len(qry)) if parent is None else means[parent]
            # Shrink with an effective sample size so decayed and plain means use comparable strength.
            neff = cnt if tau is None else sw * np.exp(-mq / tau)
            raw = np.divide(swr, sw, out=np.zeros(len(qry)), where=sw > 0)
            mean = (neff * raw + SMOOTH * prior) / (neff + SMOOTH)
            means[name] = mean
            feats[f'R_{name.upper()}{sfx}'] = mean
            feats[f'N_{name.upper()}{sfx}'] = np.log1p(neff)
    x = pd.DataFrame(feats, index=query.index)
    return x


SUPPLY_KEYS = {
    'M': ['MONTH_INDEX'],
    'M_TOWN_TYPE': ['MONTH_INDEX', 'TOWN', 'FLAT_TYPE'],
    'M_BLOCK': ['MONTH_INDEX', 'BLOCK_KEY'],
    'M_BLOCK_TYPE': ['MONTH_INDEX', 'BLOCK_KEY', 'FLAT_TYPE'],
    'BLOCK': ['BLOCK_KEY'],
    'BLOCK_TYPE_AREA': ['BLOCK_KEY', 'FLAT_TYPE', 'FLOOR_AREA_SQM'],
}


def supply_features(query: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Listing counts over `rows` (train + test descriptors, no labels): rental supply/popularity."""
    q, r = _prep(query), _prep(rows)
    out = pd.DataFrame(index=query.index)
    for name, keys in SUPPLY_KEYS.items():
        cnt = r.groupby(keys, observed=True, dropna=False).size().rename('_c').reset_index()
        out[f'SUPPLY_{name}'] = q[keys].merge(cnt, on=keys, how='left')['_c'].fillna(0).to_numpy(float)
    out['SUPPLY_BLOCK_SHARE'] = out['SUPPLY_M_BLOCK'] / out['SUPPLY_M']
    return out


ALL_ROWS: pd.DataFrame | None = None   # set in main() when --supply is on


def build(reference: pd.DataFrame, query: pd.DataFrame, with_v5: bool = False):
    """Feature matrix, categorical columns and per-row level multiplier for `query`."""
    if with_v5:
        from run_v4_experiments import build_xy
        x, _ = build_xy(reference, query, 'extended')
    else:
        x = query[NUM_COLS + CAT_COLS].copy()
        for c in CAT_COLS:
            x[c] = x[c].fillna('__missing__').astype(str)
    r = ratio_features(reference, query)
    for c in r.columns:
        x[c] = r[c].to_numpy(float)
    if ALL_ROWS is not None:
        sup = supply_features(query, ALL_ROWS)
        for c in sup.columns:
            x[c] = sup[c].to_numpy(float)
    lev = level_for(query['MONTH_INDEX'].to_numpy(), market_level(reference))
    return x, CAT_COLS.copy(), lev


def main():
    global SMOOTH, TAUS, ALL_ROWS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--smooth', type=float, default=SMOOTH)
    ap.add_argument('--taus', type=float, nargs='*', default=[12.0], help='decay constants in addition to the plain mean')
    ap.add_argument('--with-v5', action='store_true', help='also append the V4-extended raw-level history columns')
    ap.add_argument('--supply', action='store_true', help='add label-free listing-count features (train+test)')
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    SMOOTH, TAUS = a.smooth, [None] + a.taus
    train = pd.read_csv(DATA / 'train_features.csv.gz')
    test = pd.read_csv(DATA / 'test_features.csv.gz')
    if a.supply:
        ALL_ROWS = pd.concat([train.drop(columns='MONTHLY_RENT'), test], ignore_index=True)
    cache = {}
    for split in ['long', 'short']:
        sub = train[train.RENT_APPROVAL_DATE < SPLITS[split]].copy()
        val = train[(train.RENT_APPROVAL_DATE >= SPLITS[split]) & (train.RENT_APPROVAL_DATE <= SPLIT_END)].copy()
        assert sub.MONTH_INDEX.max() < val.MONTH_INDEX.min()
        xtr, cats, ltr = build(sub, sub, a.with_v5)
        xva, _, lva = build(sub, val, a.with_v5)
        cache[split] = dict(xtr=xtr, ytr=sub.MONTHLY_RENT.to_numpy(float), ltr=ltr, mtr=sub.MONTH_INDEX.to_numpy(),
                            xva=xva, yva=val.MONTHLY_RENT.to_numpy(float), lva=lva, mva=val.MONTH_INDEX.to_numpy(),
                            dva=val.RENT_APPROVAL_DATE.to_numpy(), cats=cats)
        print(split, xtr.shape, xva.shape, flush=True)
    xtr, cats, ltr = build(train, train, a.with_v5)
    xte, _, lte = build(train, test, a.with_v5)
    cache['full'] = dict(xtr=xtr, ytr=train.MONTHLY_RENT.to_numpy(float), ltr=ltr, mtr=train.MONTH_INDEX.to_numpy(),
                         xte=xte, lte=lte, mte=test.MONTH_INDEX.to_numpy(), dte=test.RENT_APPROVAL_DATE.to_numpy(), cats=cats)
    print('full', xtr.shape, xte.shape, flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(cache, f)
    print('saved', out)


if __name__ == '__main__':
    main()
