"""Run from repository root: python scripts/test_v7_features.py"""
import numpy as np
import pandas as pd
from run_v2_experiments import DATA
from v7_features import ratio_features, level_for, market_level

train = pd.read_csv(DATA / 'train_features.csv.gz')
sub = train[train.RENT_APPROVAL_DATE < '2022-07'].reset_index(drop=True)
M = int(sub.MONTH_INDEX.max()) - 2
a = ratio_features(sub, sub)
# Scrambling every label from month M on must not change features of rows dated <= M.
rng = np.random.default_rng(0)
sub2 = sub.copy()
late = sub2.MONTH_INDEX >= M
sub2.loc[late, 'MONTHLY_RENT'] = rng.permutation(sub2.loc[late, 'MONTHLY_RENT'].to_numpy()) * 3 + 777
b = ratio_features(sub2, sub2)
keep = (sub.MONTH_INDEX <= M).to_numpy()
pd.testing.assert_frame_equal(a[keep], b[keep])
# ...but later rows do see the changed history (the test above is not vacuous).
assert not np.allclose(a[~keep & (sub.MONTH_INDEX > M).to_numpy()], b[~keep & (sub.MONTH_INDEX > M).to_numpy()])
# Query labels are never read.
q = sub.iloc[:500].assign(MONTHLY_RENT=-1e9)
pd.testing.assert_frame_equal(ratio_features(sub, q), a.iloc[:500])
# First month has no history: ratios fall back to the global prior 1.0 with zero counts.
first = (sub.MONTH_INDEX == sub.MONTH_INDEX.min()).to_numpy()
assert np.allclose(a.loc[first, 'R_BLOCK_TYPE'], 1.0) and np.allclose(a.loc[first, 'N_BLOCK_TYPE'], 0.0)
# Query months after the fold use the persistence level of the fold's last 3 months.
lev = market_level(sub)
fut = level_for(np.array([lev.index.max() + 5]), lev)
assert np.isclose(fut[0], lev.iloc[-3:].mean())
print('PASS: V7 ratio features use only earlier-month labels; query labels ignored; level forecast uses fold labels only')
