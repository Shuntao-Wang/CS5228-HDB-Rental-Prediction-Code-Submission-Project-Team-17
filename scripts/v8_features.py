"""V8 feature additions using ONLY supplied attributes and earlier training rents.

Three independent families:
  shape: known flat/type/location interactions, no targets;
  premium: ratios/differences of already time-safe V5 historical means;
  trend: rolling historical rentals, frozen at each fold's train cutoff.

Never inspects Kaggle test target or any leakage-affected submission.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

SHAPE_CATS = ['V8_SHAPE_TOWN_TYPE_CAT', 'V8_SHAPE_TYPE_MODEL_CAT']
GROUPS = {
    'TOWN_TYPE': ['TOWN', 'FLAT_TYPE'],
    'TYPE': ['FLAT_TYPE'],
    'TOWN': ['TOWN'],
}


def _safe_ratio(a: pd.Series, b: pd.Series, additive: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a + additive, b + additive,
                     out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan),
                     where=np.isfinite(a) & np.isfinite(b) & ((b + additive) > 0))


def shape_features(x: pd.DataFrame) -> pd.DataFrame:
    """Features built only from existing flat descriptors (no labels)."""
    required = {'FLAT_TYPE', 'TOWN', 'FLAT_MODEL', 'FLOOR_AREA_SQM',
                'LEASE_AGE', 'MRT_NEAREST_KM'}
    if missing := required - set(x):
        raise KeyError(f'V8 shape source columns missing: {sorted(missing)}')
    out = pd.DataFrame(index=x.index)
    ft = x['FLAT_TYPE'].fillna('__missing__').astype(str).str.lower().str.strip()
    town = x['TOWN'].fillna('__missing__').astype(str).str.lower().str.strip()
    model = x['FLAT_MODEL'].fillna('__missing__').astype(str).str.lower().str.strip()
    out[SHAPE_CATS[0]] = (town + '|' + ft).astype(str)
    out[SHAPE_CATS[1]] = (ft + '|' + model).astype(str)

    room = pd.to_numeric(ft.str.extract(r'\b([1-5])\s*[- ]?room\b', expand=False), errors='coerce')
    room = room.mask(ft.str.contains('executive', regex=False), 6.0)
    area = pd.to_numeric(x['FLOOR_AREA_SQM'], errors='coerce').to_numpy(dtype=float)
    age = pd.to_numeric(x['LEASE_AGE'], errors='coerce').to_numpy(dtype=float)
    mrt = pd.to_numeric(x['MRT_NEAREST_KM'], errors='coerce').to_numpy(dtype=float)
    ords = room.to_numpy(dtype=float)
    out['V8_SHAPE_ROOM_LEVEL'] = ords
    out['V8_SHAPE_IS_EXECUTIVE'] = ft.str.contains('executive', regex=False).astype('int8').to_numpy()
    out['V8_SHAPE_LOG_AREA'] = np.where(area > 0, np.log(np.maximum(area, 1)), np.nan)
    out['V8_SHAPE_AREA_PER_LEVEL'] = np.divide(area, ords,
                                             out=np.full(len(area), np.nan),
                                             where=np.isfinite(ords) & (ords > 0))
    out['V8_SHAPE_AREA_X_AGE'] = area * age
    out['V8_SHAPE_LOG1P_MRT'] = np.where(mrt >= 0, np.log1p(np.maximum(mrt, 0)), np.nan)
    return out.reset_index(drop=True)


PREMIUM_PAIRS = (
    ('TOWN_TYPE_VS_TYPE_AREA', 'HIST_TOWN_TYPE_MEAN', 'COMP_TYPE_AREA_MEAN'),
    ('STREET_TYPE_VS_TOWN_TYPE', 'COMP_STREET_TYPE_MEAN', 'HIST_TOWN_TYPE_MEAN'),
    ('STREET_TYPE_AREA_VS_TOWN_TYPE', 'COMP_STREET_TYPE_AREA_MEAN', 'HIST_TOWN_TYPE_MEAN'),
    ('BLOCK_TYPE_AREA_VS_STREET_TYPE_AREA', 'COMP_BLOCK_TYPE_AREA_MEAN', 'COMP_STREET_TYPE_AREA_MEAN'),
)


def premium_features(x: pd.DataFrame) -> pd.DataFrame:
    """Transform V5's existing, earlier-month rent aggregates only."""
    missing = set(col for _, a, b in PREMIUM_PAIRS for col in (a, b)) - set(x)
    if missing:
        raise KeyError(f'V8 premium needs V5 extended columns: {sorted(missing)}')
    out = pd.DataFrame(index=x.index)
    for tag, lhs, rhs in PREMIUM_PAIRS:
        a = pd.to_numeric(x[lhs], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(x[rhs], errors='coerce').to_numpy(dtype=float)
        out[f'V8_PREMIUM_{tag}_RATIO'] = _safe_ratio(a, b, additive=200.0)
        out[f'V8_PREMIUM_{tag}_DIFF'] = a-b
    return out.reset_index(drop=True)


def _asof(monthly: pd.DataFrame, query_keys: pd.DataFrame, group_cols: list[str],
          cutoff_inclusive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast grouped cumulative sum/count <= cutoff; unmatched groups -> zeros."""
    n = len(query_keys)
    if monthly.empty:
        return np.zeros(n), np.zeros(n)
    table = pd.concat([monthly[group_cols], query_keys[group_cols]], ignore_index=True)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(table), sort=False)
    rc = codes[:len(monthly)].astype(np.int64, copy=False)
    qc = codes[len(monthly):].astype(np.int64, copy=False)
    rm = monthly['MONTH_INDEX'].to_numpy(dtype=np.int64)
    qm = np.asarray(cutoff_inclusive, dtype=np.int64)
    low = int(min(rm.min(), qm.min())) if n else int(rm.min())
    high = int(max(rm.max(), qm.max())) if n else int(rm.max())
    span = high-low+1
    refs = rc * span + (rm-low)
    order = np.argsort(refs, kind='stable')
    positions = np.searchsorted(refs[order], qc*span + (qm-low), side='right')-1
    ok = positions >= 0
    idx = np.zeros(n, dtype=np.int64)
    idx[ok] = order[positions[ok]]
    ok &= rc[idx] == qc
    sums = np.where(ok, monthly['_S_CUM'].to_numpy(dtype=float)[idx], 0.0)
    counts = np.where(ok, monthly['_N_CUM'].to_numpy(dtype=float)[idx], 0.0)
    return sums, counts


def trend_features(reference: pd.DataFrame, query: pd.DataFrame,
                   *, cutoff_month_index: int) -> pd.DataFrame:
    """Earlier-month 6/12-month rent means; freeze at the fold's train cutoff."""
    required = {'MONTH_INDEX', 'MONTHLY_RENT', 'TOWN', 'FLAT_TYPE'}
    needed_q = required - {'MONTHLY_RENT'}
    if (miss := required - set(reference)) or (miss_q := needed_q - set(query)):
        raise KeyError(f'V8 trend missing reference={miss}, query={miss_q}')
    if len(reference) == 0:
        raise ValueError('Empty reference data')
    if int(reference['MONTH_INDEX'].max()) >= cutoff_month_index:
        raise ValueError('Reference extends beyond fold forecast origin (would leak)')
    anchors = np.minimum(query['MONTH_INDEX'].to_numpy(dtype=np.int64), cutoff_month_index)
    ref = reference[['MONTH_INDEX','MONTHLY_RENT','TOWN','FLAT_TYPE']].copy()
    q = query[['TOWN','FLAT_TYPE']].copy().reset_index(drop=True)
    for key in ('TOWN', 'FLAT_TYPE'):
        ref[key] = ref[key].fillna('__missing__').astype(str)
        q[key] = q[key].fillna('__missing__').astype(str)
    result = pd.DataFrame(index=range(len(query)))
    for tag, cols in GROUPS.items():
        tic = time.perf_counter()
        monthly = (ref.groupby(cols+['MONTH_INDEX'], sort=False, observed=True, dropna=False)
                   .agg(_SUM=('MONTHLY_RENT','sum'), _COUNT=('MONTHLY_RENT','size'))
                   .reset_index())
        monthly['_S_CUM'] = monthly.groupby(cols, sort=False, observed=True, dropna=False)['_SUM'].cumsum()
        monthly['_N_CUM'] = monthly.groupby(cols, sort=False, observed=True, dropna=False)['_COUNT'].cumsum()
        full_s, full_n = _asof(monthly,q,cols,anchors-1)
        means = {}
        for w in (6, 12):
            old_s, old_n = _asof(monthly,q,cols,anchors-1-w)
            s = full_s-old_s
            n = np.maximum(full_n-old_n,0.0)
            hist_mean = np.divide(full_s, full_n,
                                  out=np.full(len(q), np.nan), where=full_n>0)
            rolling = np.divide(s+10*np.nan_to_num(hist_mean), n+10,
                                out=np.full(len(q), np.nan), where=full_n>0)
            result[f'V8_TREND_{tag}_MEAN_{w}M'] = rolling
            result[f'V8_TREND_{tag}_LOG_N_{w}M'] = np.log1p(n)
            means[w] = rolling
        result[f'V8_TREND_{tag}_MOMENTUM_6V12'] = means[6]-means[12]
        print(f'  V8 trend {tag}: {time.perf_counter()-tic:.2f}s', flush=True)
    return result
