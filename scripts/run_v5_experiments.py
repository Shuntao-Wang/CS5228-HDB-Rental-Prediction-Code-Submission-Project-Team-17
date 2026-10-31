# V5 model experiments — submitted by ruihe (2026-10-20).
# Reads the cached V4-extended feature matrices and compares CatBoost/LightGBM settings,
# recency weighting and trend extrapolation on the two chronological holdouts.
"""V5 model-side experiments on the cached V4-extended feature matrices.

Run from the repository root (after scripts/v5_cache_features.py):
    python scripts/run_v5_experiments.py --variant cat_control
    python scripts/run_v5_experiments.py --variant cat_big
    python scripts/run_v5_experiments.py --variant lgb_big
    python scripts/run_v5_experiments.py --variant cat_big --recency 24
    python scripts/run_v5_experiments.py --variant cat_big --trend damped

Each run stores out-of-time validation predictions (long + short holdout) and, with
--submit, test predictions, so blends can be evaluated later without refitting.
Features are the same as V4 extended; only model/weights/target handling change.
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v2_experiments import ROOT, score

CACHE = ROOT / 'experiments' / 'v5' / 'feature_cache.pkl'
OUT = ROOT / 'experiments' / 'v5' / 'runs'

VARIANTS = {
    # name: (library, params)
    'cat_control': ('catboost', dict(iterations=300, depth=6, learning_rate=0.06, l2_leaf_reg=5)),
    'cat_big': ('catboost', dict(iterations=3000, depth=8, learning_rate=0.03, l2_leaf_reg=10,
                                 bootstrap_type='Bernoulli', subsample=0.8, rsm=0.8)),
    'cat_deep': ('catboost', dict(iterations=4000, depth=10, learning_rate=0.02, l2_leaf_reg=20,
                                  bootstrap_type='Bernoulli', subsample=0.8, rsm=0.7, min_data_in_leaf=20)),
    'lgb_big': ('lightgbm', dict(n_estimators=3000, learning_rate=0.02, num_leaves=127, min_child_samples=50,
                                 colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=10.0,
                                 max_cat_to_onehot=8, cat_smooth=20, max_bin=255)),
    # Regularised variants: higher L2 + row/feature subsampling beat both the 300-tree control
    # and the 3000-tree 'cat_big' (which overfits noise). These two are used in the final blend.
    'cat_reg': ('catboost', dict(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=20,
                                 bootstrap_type='Bernoulli', subsample=0.8, rsm=0.8)),
    'cat_small': ('catboost', dict(iterations=600, depth=5, learning_rate=0.05, l2_leaf_reg=10)),
    'cat_mid': ('catboost', dict(iterations=800, depth=7, learning_rate=0.04, l2_leaf_reg=30,
                                 bootstrap_type='Bernoulli', subsample=0.7, rsm=0.7, min_data_in_leaf=50)),
    # Heavily regularised LightGBM: slightly weaker alone but adds diversity to the blend.
    'lgb_reg': ('lightgbm', dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                 colsample_bytree=0.7, subsample=0.7, subsample_freq=1, reg_lambda=20.0,
                                 max_cat_to_onehot=8, cat_smooth=50, min_data_per_group=100, cat_l2=20)),
    'lgb_mid': ('lightgbm', dict(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                                 colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                                 max_cat_to_onehot=8, cat_smooth=20)),
}


def recency_weights(month_idx: np.ndarray, tau: float | None) -> np.ndarray | None:
    if not tau:
        return None
    age = month_idx.max() - month_idx
    return np.exp(-age / tau)


def fit_predict(lib: str, params: dict, xtr, ytr, xevals: list, cats, seed, threads, weights=None):
    if lib == 'catboost':
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor(loss_function='RMSE', random_seed=seed, thread_count=threads,
                                  verbose=False, allow_writing_files=False, **params)
        model.fit(Pool(xtr, ytr, cat_features=cats, weight=weights))
        return [model.predict(x) for x in xevals]
    if lib == 'lightgbm':
        from lightgbm import LGBMRegressor
        xtr = xtr.copy(); xevals = [x.copy() for x in xevals]
        for c in cats:
            dt = pd.CategoricalDtype(categories=pd.Index(xtr[c].unique()))
            xtr[c] = xtr[c].astype(dt)
            for x in xevals:
                x[c] = x[c].astype(dt)
        model = LGBMRegressor(objective='regression', random_state=seed, n_jobs=threads, verbosity=-1, **params)
        model.fit(xtr, ytr, sample_weight=weights, categorical_feature=cats)
        return [model.predict(x) for x in xevals]
    raise ValueError(lib)


def market_level(y: np.ndarray, m: np.ndarray) -> pd.Series:
    """Monthly mean rent (market level) of the training fold."""
    return pd.Series(y).groupby(m).mean()


# NOTE: trend extrapolation was tested and rejected (long-horizon bias +71/+126 SGD);
# the final submission uses mode='none'.
def trend_adjust(pred: np.ndarray, m_query: np.ndarray, level: pd.Series, mode: str, damp: float = 0.5) -> np.ndarray:
    """Additive extrapolation of the market trend beyond the last training month.

    Trees hold the level of the last training months; this adds slope*horizon where slope is
    the least-squares slope (SGD/month) of the last 12 training months, damped by `damp`.
    """
    if mode == 'none':
        return pred
    last = int(level.index.max())
    recent = level.loc[level.index > last - 12]
    slope = np.polyfit(recent.index.to_numpy(float), recent.to_numpy(float), 1)[0]
    horizon = np.clip(m_query - last, 0, None).astype(float)
    if mode == 'damped':
        slope *= damp
    return pred + slope * horizon


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    p.add_argument('--recency', type=float, default=0.0, help='exp decay tau in months (0=off)')
    p.add_argument('--trend', default='none', choices=['none', 'damped', 'full'])
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--tag', default='')
    p.add_argument('--geo', action='store_true', help='use feature_cache_geo.pkl (adds neighbour rent features)')
    a = p.parse_args()
    lib, params = VARIANTS[a.variant]
    name = a.variant + (f'_rec{a.recency:g}' if a.recency else '') + (f'_trend-{a.trend}' if a.trend != 'none' else '') \
        + (f'_s{a.seeds}' if a.seeds > 1 else '') + ('_geo' if a.geo else '') + (f'_{a.tag}' if a.tag else '')
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    with open(CACHE.with_name('feature_cache_geo.pkl') if a.geo else CACHE, 'rb') as f:
        cache = pickle.load(f)
    record = dict(name=name, geo=a.geo, library=lib, params=params, recency_tau=a.recency, trend=a.trend, seeds=a.seeds, holdouts=[])
    preds_store = {}
    for split in ['long', 'short']:
        c = cache[split]
        t0 = time.time()
        w = recency_weights(c['mtr'], a.recency)
        preds = np.zeros(len(c['yva']))
        for s in range(a.seeds):
            preds += fit_predict(lib, params, c['xtr'], c['ytr'], [c['xva']], c['cats'], 42 + s, a.threads, w)[0]
        preds /= a.seeds
        preds = trend_adjust(preds, c['mva'], market_level(c['ytr'], c['mtr']), a.trend)
        m = score(c['yva'], preds)
        m.update(split=split, n_train=len(c['ytr']), n_valid=len(c['yva']), seconds=round(time.time() - t0, 1))
        record['holdouts'].append(m)
        preds_store[split] = preds
        print(f'[{name} {split}] RMSE={m["rmse"]:.3f} MAE={m["mae"]:.3f} bias={m["mean_error_pred_minus_actual"]:+.2f} ({m["seconds"]}s)', flush=True)
        pd.DataFrame({'month': c['dva'], 'actual': c['yva'], 'pred': preds}).to_csv(out / f'valid_{split}.csv', index=False, float_format='%.4f')
    if a.submit:
        c = cache['full']
        w = recency_weights(c['mtr'], a.recency)
        preds = np.zeros(len(c['mte']))
        for s in range(a.seeds):
            preds += fit_predict(lib, params, c['xtr'], c['ytr'], [c['xte']], c['cats'], 42 + s, a.threads, w)[0]
        preds /= a.seeds
        preds = trend_adjust(preds, c['mte'], market_level(c['ytr'], c['mtr']), a.trend)
        assert len(preds) == 50000 and np.isfinite(preds).all()
        ids = pd.read_csv(ROOT / 'examples' / 'submission_reference.csv')['Id'].to_numpy()
        pd.DataFrame({'Id': ids, 'Predicted': preds}).to_csv(out / 'test_pred.csv', index=False, float_format='%.6f')
        print(f'[{name} submit] test mean pred={preds.mean():.1f}', flush=True)
    (out / 'metrics.json').write_text(json.dumps(record, indent=2) + '\n')
    print('saved', out)


if __name__ == '__main__':
    main()
