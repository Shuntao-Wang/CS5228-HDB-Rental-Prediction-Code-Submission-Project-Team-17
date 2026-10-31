# V5 geographic neighbour features — submitted by ruihe.
# Result: neutral on both holdouts (see docs/v5_results.md); kept for the ablation record.
"""Time-safe geographic-neighbour rent features (V5).

For each query row (block b, flat type f, month M) we aggregate reference-fold labels of
the same flat type located within `radius_km` of block b, restricted to reference months
strictly earlier than M (and, for a recent-window variant, within the last `window` months).
Shrunk towards the V2 town x type prior mean with pseudo-count `smoothing`.
No same-month, validation or test labels are used anywhere.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v2_experiments import make_past_rent_features

EARTH_KM = 6371.0088


def make_geo_features(reference: pd.DataFrame, query: pd.DataFrame, radius_km=(0.5, 1.0),
                      window: int = 12, smoothing: float = 25.0) -> pd.DataFrame:
    ref = reference[['BLOCK_KEY', 'FLAT_TYPE', 'MONTH_INDEX', 'MONTHLY_RENT', 'LATITUDE', 'LONGITUDE']].copy()
    qry = query[['BLOCK_KEY', 'FLAT_TYPE', 'MONTH_INDEX', 'LATITUDE', 'LONGITUDE']].copy()
    # Block coordinate table (union of reference and query blocks).
    blocks = (pd.concat([ref[['BLOCK_KEY', 'LATITUDE', 'LONGITUDE']], qry[['BLOCK_KEY', 'LATITUDE', 'LONGITUDE']]])
              .dropna().drop_duplicates('BLOCK_KEY').reset_index(drop=True))
    bidx = {k: i for i, k in enumerate(blocks.BLOCK_KEY)}
    types = sorted(set(ref.FLAT_TYPE) | set(qry.FLAT_TYPE))
    tidx = {t: i for i, t in enumerate(types)}
    n_months = int(max(ref.MONTH_INDEX.max(), qry.MONTH_INDEX.max())) + 2
    nb, nt = len(blocks), len(types)
    # Per (block, type, month) sums and counts, then cumulative over months.
    S = np.zeros((nb, nt, n_months)); N = np.zeros((nb, nt, n_months))
    r = ref.dropna(subset=['LATITUDE', 'LONGITUDE'])
    bi = r.BLOCK_KEY.map(bidx).to_numpy(); ti = r.FLAT_TYPE.map(tidx).to_numpy(); mi = r.MONTH_INDEX.to_numpy(int)
    np.add.at(S, (bi, ti, mi), r.MONTHLY_RENT.to_numpy(float)); np.add.at(N, (bi, ti, mi), 1.0)
    CS = np.cumsum(S, axis=2); CN = np.cumsum(N, axis=2)
    # pad so that index m-1 = -1 maps to zeros
    CS = np.concatenate([np.zeros((nb, nt, 1)), CS], axis=2); CN = np.concatenate([np.zeros((nb, nt, 1)), CN], axis=2)
    tree = BallTree(np.deg2rad(blocks[['LATITUDE', 'LONGITUDE']].to_numpy()), metric='haversine')
    base = make_past_rent_features(reference, query, smoothing=smoothing)
    parent = base['HIST_TOWN_TYPE_MEAN'].to_numpy(float)
    out = pd.DataFrame(index=query.index)
    qb = qry.BLOCK_KEY.map(bidx).to_numpy(); qt = qry.FLAT_TYPE.map(tidx).to_numpy(); qm = qry.MONTH_INDEX.to_numpy(int)
    valid = ~pd.isna(qb)
    qb = np.where(valid, qb, 0).astype(int)
    for rad in radius_km:
        neigh = tree.query_radius(np.deg2rad(blocks[['LATITUDE', 'LONGITUDE']].to_numpy()), r=rad / EARTH_KM)
        for name, lo_off in [('all', None), (f'w{window}', window)]:
            s = np.zeros(len(qry)); n = np.zeros(len(qry))
            for i in range(len(qry)):
                if not valid[i]:
                    s[i] = np.nan; continue
                nbrs = neigh[qb[i]]
                hi = qm[i]  # cumulative index hi == months < qm[i]
                s_hi = CS[nbrs, qt[i], hi].sum(); n_hi = CN[nbrs, qt[i], hi].sum()
                if lo_off is not None:
                    lo = max(hi - lo_off, 0)
                    s_hi -= CS[nbrs, qt[i], lo].sum(); n_hi -= CN[nbrs, qt[i], lo].sum()
                s[i] = s_hi; n[i] = n_hi
            mean = np.divide(s + smoothing * np.nan_to_num(parent, nan=0.0), n + smoothing,
                             out=np.full(len(qry), np.nan), where=np.isfinite(parent) & np.isfinite(s))
            tag = f'GEO_{int(rad*1000)}M_{name.upper()}'
            out[f'{tag}_MEAN'] = mean
            out[f'{tag}_LOG_COUNT'] = np.log1p(np.nan_to_num(n, nan=0.0))
    return out


if __name__ == '__main__':
    # Leakage self-test: a query row's own month must not contribute.
    import time
    from run_v2_experiments import DATA
    tr = pd.read_csv(DATA / 'train_features.csv.gz')
    sub = tr[tr.RENT_APPROVAL_DATE < '2021-06'].copy()
    t0 = time.time(); f = make_geo_features(sub, sub); print(f.describe().T, time.time() - t0)
    first = sub.MONTH_INDEX == sub.MONTH_INDEX.min()
    assert f.loc[first, 'GEO_1000M_ALL_MEAN'].isna().all(), 'first month must have no history'
    # Perturb one row's label in month 2; features of rows in month 2 must be unchanged.
    sub2 = sub.copy(); i = sub2.index[sub2.MONTH_INDEX == 2][0]; sub2.loc[i, 'MONTHLY_RENT'] += 100000
    f2 = make_geo_features(sub2, sub2)
    m2 = sub2.MONTH_INDEX == 2
    assert np.allclose(f.loc[m2].fillna(-1), f2.loc[m2].fillna(-1)), 'same-month leakage'
    assert not np.allclose(f.loc[sub2.MONTH_INDEX == 3].fillna(-1), f2.loc[sub2.MONTH_INDEX == 3].fillna(-1)), 'later months should see the change'
    print('geo feature leakage tests passed')
