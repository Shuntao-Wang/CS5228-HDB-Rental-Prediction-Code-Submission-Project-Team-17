# V7 model runs on the detrended feature cache (scripts/v7_features.py).
"""Train on the V7 cache with a choice of target transform and store holdout/test predictions.

    python scripts/run_v7_experiments.py --variant cat_reg --target ratio
    python scripts/run_v7_experiments.py --variant cat_reg --target raw      # same features, raw target
    python scripts/run_v7_experiments.py --variant lgb_reg --target ratio --seeds 3 --submit

Targets: raw = y; ratio = y / L(m), prediction * L(m); resid = y - L(m), prediction + L(m).
Output layout matches V5 (experiments/v7/runs/<name>/{valid_long,valid_short}.csv, test_pred.csv),
so scripts/v5_blend.py-style blending works with --runs-dir.
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v2_experiments import ROOT, score
from run_v5_experiments import VARIANTS, fit_predict

CACHE = ROOT / 'experiments' / 'v7' / 'feature_cache.pkl'
OUT = ROOT / 'experiments' / 'v7' / 'runs'
TIME_COLS = ['APPROVAL_YEAR', 'MONTH_INDEX']


def to_target(y, lev, kind):
    return {'raw': y, 'ratio': y / lev, 'resid': y - lev}[kind]


def from_target(p, lev, kind):
    return {'raw': p, 'ratio': p * lev, 'resid': p + lev}[kind]


def select(x: pd.DataFrame, drop: list[str]) -> pd.DataFrame:
    cols = [c for c in x.columns if not any(c == d or (d.endswith('*') and c.startswith(d[:-1])) for d in drop)]
    return x[cols]


def drift_factor(ytr, mtr, mq, damp, slope_win=12, win=3):
    """exp(damp * g * h): g = log-slope of the fold's monthly mean rent over its last `slope_win`
    months, h = months after the centre of the persistence window (see scripts/v7_level_check.py)."""
    if not damp:
        return np.ones(len(mq))
    lev = pd.Series(ytr).groupby(mtr).mean().sort_index()
    last = lev.index.max()
    r = lev.loc[lev.index > last - slope_win]
    g = np.polyfit(r.index.to_numpy(float), np.log(r.to_numpy(float)), 1)[0]
    h = np.clip(mq - (last - (win - 1) / 2), 0, None)
    return np.where(mq > last, np.exp(damp * g * h), 1.0)


def run_fold(lib, params, c, xkey, lkey, target, seeds, threads, drop, drift=0.0):
    xtr, xq = select(c['xtr'], drop), select(c[xkey], drop)
    cats = [k for k in c['cats'] if k in xtr.columns]
    ytr = to_target(c['ytr'], c['ltr'], target)
    pred = np.zeros(len(xq))
    for s in range(seeds):
        pred += fit_predict(lib, params, xtr, ytr, [xq], cats, 42 + s, threads)[0]
    mq = c['mva' if xkey == 'xva' else 'mte']
    lev = c[lkey] * (drift_factor(c['ytr'], c['mtr'], mq, drift) if target != 'raw' else 1.0)
    return from_target(pred / seeds, lev, target), xtr.shape[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    p.add_argument('--target', default='ratio', choices=['raw', 'ratio', 'resid'])
    p.add_argument('--drop', nargs='*', default=[], help="feature columns to drop; 'time' = year/month index; prefix*")
    p.add_argument('--drift', type=float, default=0.5, help='damping of the trailing-12-month level drift (ratio/resid)')
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--threads', type=int, default=10)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--tag', default='')
    p.add_argument('--cache', default=str(CACHE))
    a = p.parse_args()
    drop = [d for x in a.drop for d in (TIME_COLS if x == 'time' else [x])]
    lib, params = VARIANTS[a.variant]
    name = f'{a.variant}_{a.target}' + (f'_s{a.seeds}' if a.seeds > 1 else '') + (f'_{a.tag}' if a.tag else '')
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    with open(a.cache, 'rb') as f:
        cache = pickle.load(f)
    record = dict(name=name, library=lib, params=params, target=a.target, drift=a.drift, drop=drop, seeds=a.seeds,
                  cache=Path(a.cache).name, holdouts=[])
    for split in ['long', 'short']:
        c = cache[split]
        t0 = time.time()
        pred, nf = run_fold(lib, params, c, 'xva', 'lva', a.target, a.seeds, a.threads, drop, a.drift)
        m = score(c['yva'], pred)
        m.update(split=split, n_features=nf, seconds=round(time.time() - t0, 1))
        record['holdouts'].append(m)
        print(f'[{name} {split}] RMSE={m["rmse"]:.3f} MAE={m["mae"]:.3f} bias={m["mean_error_pred_minus_actual"]:+.2f} '
              f'({nf} feats, {m["seconds"]}s)', flush=True)
        pd.DataFrame({'month': c['dva'], 'actual': c['yva'], 'pred': pred}).to_csv(
            out / f'valid_{split}.csv', index=False, float_format='%.4f')
    if a.submit:
        c = cache['full']
        pred, _ = run_fold(lib, params, c, 'xte', 'lte', a.target, a.seeds, a.threads, drop, a.drift)
        assert len(pred) == 50000 and np.isfinite(pred).all()
        ids = pd.read_csv(ROOT / 'examples' / 'submission_reference.csv')['Id'].to_numpy()
        pd.DataFrame({'Id': ids, 'Predicted': pred}).to_csv(out / 'test_pred.csv', index=False, float_format='%.6f')
        print(f'[{name} submit] test mean pred={pred.mean():.1f}', flush=True)
    (out / 'metrics.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
