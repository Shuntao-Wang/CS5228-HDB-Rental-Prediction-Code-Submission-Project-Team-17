"""Compare LightGBM/CatBoost on chronological holdouts, optionally with prior-month rent statistics.

Examples (from repository root):
    python scripts/run_v2_experiments.py --model lightgbm --history none --split both
    python scripts/run_v2_experiments.py --model lightgbm --history prior --split both
    python scripts/run_v2_experiments.py --model catboost --history prior --split short
    python scripts/run_v2_experiments.py --model lightgbm --history prior --split both --submit

Data: V1 data/processed/{train_features,test_features}.csv.gz (NOT numeric processed files).
Every aggregate for each row uses strictly EARLIER MONTH labels only; for validation,
only the training partition's labels are included in the reference history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
NUM_COLS = [
    'APPROVAL_YEAR', 'APPROVAL_MONTH', 'MONTH_INDEX', 'MONTH_SIN', 'MONTH_COS',
    'FLOOR_AREA_SQM', 'LEASE_COMMENCE_DATE', 'LEASE_AGE', 'LEASE_AGE_INVALID',
    'LATITUDE', 'LONGITUDE', 'MAX_FLOOR', 'YEAR_COMPLETED', 'BUILDING_AGE',
    'MAX_FLOOR_MISSING', 'YEAR_COMPLETED_MISSING', 'MRT_NEAREST_KM', 'MRT_WITHIN_1KM',
    'PRIMARY_SCHOOL_NEAREST_KM', 'PRIMARY_SCHOOL_WITHIN_1KM', 'MALL_NEAREST_KM',
    'MALL_WITHIN_2KM',
]
CAT_COLS = ['TOWN', 'FLAT_TYPE', 'FLAT_MODEL', 'REGION', 'PLANNING_AREA', 'STREET', 'BLOCK_KEY']
GROUPS = {
    'town_type': ['TOWN', 'FLAT_TYPE'],
    'block_type': ['BLOCK_KEY', 'FLAT_TYPE'],
    'town': ['TOWN'],
}
SPLITS = {'long': '2024-01', 'short': '2024-07'}
SPLIT_END = '2025-03'


def _month_asof(ref: pd.DataFrame, qry: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """For each query, return group sum/count of reference labels from months < query month.

    Does NOT assume the query data are unlabeled. In particular, computing a training
    feature from ref==query excludes the query's ENTIRE month (incl. its own label).
    """
    grouped = (ref.groupby(by + ['MONTH_INDEX'], sort=True, observed=True, dropna=False)
               .agg(_n=('MONTHLY_RENT', 'size'), _s=('MONTHLY_RENT', 'sum')).reset_index())
    grouped['_n'] = grouped.groupby(by, sort=False, observed=True, dropna=False)['_n'].cumsum()
    grouped['_s'] = grouped.groupby(by, sort=False, observed=True, dropna=False)['_s'].cumsum()
    # Pandas merge_asof requires *globally* ordered on-key, even with by groups.
    grouped = grouped.sort_values('MONTH_INDEX', kind='stable')
    lookup = qry[by + ['MONTH_INDEX']].copy()
    lookup['_index'] = np.arange(len(qry), dtype=np.int32)
    lookup = lookup.sort_values('MONTH_INDEX', kind='stable')
    matched = pd.merge_asof(
        lookup, grouped, on='MONTH_INDEX', by=by,
        direction='backward', allow_exact_matches=False,
    ).sort_values('_index', kind='stable')
    return matched[['_n', '_s']].reset_index(drop=True)


def make_past_rent_features(reference: pd.DataFrame, query: pd.DataFrame, smoothing: float = 25.0) -> pd.DataFrame:
    """Create time-safe aggregate features. Reference is ONLY the known-label period.

    For *every* query month M, use solely reference labels dated < M.
    For rows from earliest month, past means are NaN, which models can handle.
    Shrink grouped rent means toward global past mean using pseudo-count smoothing.
    """
    if smoothing < 0:
        raise ValueError('smoothing must be nonnegative')
    if len(reference) == 0:
        raise ValueError('Reference dataset is empty')
    for field in ['MONTH_INDEX', 'MONTHLY_RENT', *CAT_COLS]:
        if field not in reference:
            raise KeyError(f'Reference missing {field}')
    if 'MONTH_INDEX' not in query:
        raise KeyError('Query missing MONTH_INDEX')
    # Strictly earlier month global reference counts/means.
    glob = (reference.groupby('MONTH_INDEX')['MONTHLY_RENT']
            .agg(_global_n='size', _global_s='sum').reset_index()
            .sort_values('MONTH_INDEX'))
    glob['_global_n'] = glob['_global_n'].cumsum()
    glob['_global_s'] = glob['_global_s'].cumsum()
    q = query[['MONTH_INDEX']].copy().reset_index(drop=True)
    q['_index'] = np.arange(len(q))
    tmp = pd.merge_asof(
        q.sort_values('MONTH_INDEX'), glob, on='MONTH_INDEX',
        direction='backward', allow_exact_matches=False,
    ).sort_values('_index')
    global_n = tmp['_global_n'].fillna(0).to_numpy(dtype=float)
    global_s = tmp['_global_s'].fillna(0).to_numpy(dtype=float)
    global_mean = np.divide(global_s, global_n, out=np.full(len(q), np.nan), where=global_n > 0)
    result = pd.DataFrame(index=query.index)
    result['HIST_GLOBAL_MEAN'] = global_mean
    result['HIST_GLOBAL_LOG_COUNT'] = np.log1p(global_n)

    for name, cols in GROUPS.items():
        p = _month_asof(reference, query, cols)
        count = p['_n'].fillna(0).to_numpy(dtype=float)
        summ = p['_s'].fillna(0).to_numpy(dtype=float)
        # If no global previous-month labels exist, leave as NaN; never use future labels.
        shrunk = np.divide(
            summ + smoothing * np.nan_to_num(global_mean, nan=0.0),
            count + smoothing,
            out=np.full(len(q), np.nan), where=(global_n > 0) & ((count + smoothing) > 0),
        )
        result[f'HIST_{name.upper()}_MEAN'] = shrunk
        result[f'HIST_{name.upper()}_LOG_COUNT'] = np.log1p(count)
    if not result.index.equals(query.index):
        raise AssertionError('Historical aggregate index changed unexpectedly')
    return result


def make_xy(reference: pd.DataFrame, query: pd.DataFrame, history: str) -> tuple[pd.DataFrame, list[str]]:
    required = set(NUM_COLS + CAT_COLS + ['MONTH_INDEX'])
    if missing := required - set(query.columns):
        raise ValueError(f'Missing features: {sorted(missing)}')
    x = query[NUM_COLS + CAT_COLS].copy()
    for col in CAT_COLS:
        x[col] = x[col].fillna('__missing__').astype(str)
    if history == 'prior':
        # Reference must be restricted to the corresponding training fold.
        aux = make_past_rent_features(reference, query)
        if len(aux) != len(x):
            raise AssertionError('History feature row mismatch')
        for col in aux.columns:
            x[col] = aux[col].to_numpy()
    return x, CAT_COLS.copy()


def _category_encoding(xtr: pd.DataFrame, xval: pd.DataFrame, categories: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Categorical vocab fitted ONLY on this chronological training partition.
    a, b = xtr.copy(), xval.copy()
    for col in categories:
        dt = pd.CategoricalDtype(categories=pd.Index(a[col].unique()), ordered=False)
        a[col] = a[col].astype(dt)
        b[col] = b[col].astype(dt)
    return a, b


def fit_predict(model_name: str, xtr: pd.DataFrame, ytr: np.ndarray, xval: pd.DataFrame,
                categories: list[str], args: argparse.Namespace) -> np.ndarray:
    if model_name == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise RuntimeError('Run pip install -r requirements-models-v2.txt') from e
        a, b = _category_encoding(xtr, xval, categories)
        model = LGBMRegressor(
            objective='regression', n_estimators=args.lgb_estimators, learning_rate=0.04,
            num_leaves=31, min_child_samples=70, max_cat_to_onehot=8,
            colsample_bytree=0.9, reg_lambda=4.0, random_state=42,
            n_jobs=args.threads, verbosity=-1,
        )
        model.fit(a, ytr, categorical_feature=categories)
        return model.predict(b)
    if model_name == 'catboost':
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise RuntimeError('Run pip install -r requirements-models-v2.txt') from e
        model = CatBoostRegressor(
            iterations=args.cat_iterations, depth=6, learning_rate=0.06,
            l2_leaf_reg=5, loss_function='RMSE', random_seed=42,
            thread_count=args.threads, verbose=False, allow_writing_files=False,
        )
        # CatBoost requires named columns with corresponding category strings.
        model.fit(xtr, ytr, cat_features=categories)
        return model.predict(xval)
    raise ValueError(model_name)


def score(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        'rmse': float(np.sqrt(mean_squared_error(actual, pred))),
        'mae': float(mean_absolute_error(actual, pred)),
        'mean_error_pred_minus_actual': float(np.mean(pred - actual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['lightgbm', 'catboost'])
    parser.add_argument('--history', choices=['none', 'prior'], default='none')
    parser.add_argument('--split', choices=['long', 'short', 'both'], default='both')
    parser.add_argument('--submit', action='store_true', help='Refit on all labeled training data and write Kaggle CSV')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--lgb-estimators', type=int, default=300)
    parser.add_argument('--cat-iterations', type=int, default=300)
    args = parser.parse_args()
    if min(args.threads, args.lgb_estimators, args.cat_iterations) < 1:
        parser.error('--threads, --lgb-estimators and --cat-iterations must be >=1')

    train = pd.read_csv(DATA / 'train_features.csv.gz')
    test = pd.read_csv(DATA / 'test_features.csv.gz') if args.submit else None
    if 'MONTHLY_RENT' not in train or len(train) != 150000:
        raise ValueError('Expected 150,000 rows and MONTHLY_RENT in train_features.csv.gz')
    experiment_name = f'{args.model}_{args.history}'
    out = ROOT / 'experiments' / 'v2' / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'model': args.model, 'history': args.history, 'seed': 42,
        'parameters': {'threads': args.threads, 'lgb_estimators': args.lgb_estimators, 'cat_iterations': args.cat_iterations},
        'evaluation': 'train dates strictly before validation; fitted category vocabulary and historical stats from train only',
        'holdouts': [],
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    for split in (['long', 'short'] if args.split == 'both' else [args.split]):
        start = SPLITS[split]
        subset = train[train['RENT_APPROVAL_DATE'] < start].copy()
        valid = train[(train['RENT_APPROVAL_DATE'] >= start) & (train['RENT_APPROVAL_DATE'] <= SPLIT_END)].copy()
        if len(subset) == 0 or len(valid) == 0 or subset['RENT_APPROVAL_DATE'].max() >= valid['RENT_APPROVAL_DATE'].min():
            raise AssertionError('Invalid chronological train/validation boundaries')
        print(f'[{experiment_name} {split}] train {len(subset):,}, valid {len(valid):,}', flush=True)
        xt, cats = make_xy(subset, subset, args.history)
        xv, _ = make_xy(subset, valid, args.history)
        pred = fit_predict(args.model, xt, subset['MONTHLY_RENT'].to_numpy(dtype=float), xv, cats, args)
        actual = valid['MONTHLY_RENT'].to_numpy(dtype=float)
        metrics = score(actual, pred)
        metrics.update({'split': split, 'train_end': str(subset['RENT_APPROVAL_DATE'].max()),
                        'valid_start': start, 'valid_end': SPLIT_END,
                        'n_train': len(subset), 'n_valid': len(valid), 'n_features': xt.shape[1]})
        result['holdouts'].append(metrics)
        print(f'  RMSE={metrics["rmse"]:.2f}; MAE={metrics["mae"]:.2f}; bias={metrics["mean_error_pred_minus_actual"]:+.2f}', flush=True)
        grouped = (pd.DataFrame({'month': valid['RENT_APPROVAL_DATE'].to_numpy(), 'y': actual, 'pred': pred})
                   .groupby('month', sort=True))
        monthly = grouped.apply(lambda g: pd.Series({
            'n': len(g), 'rmse': float(np.sqrt(np.mean((g['y'] - g['pred']) ** 2))),
            'mae': float(np.mean(np.abs(g['y'] - g['pred']))),
            'actual_mean': float(g['y'].mean()), 'prediction_mean': float(g['pred'].mean()),
        }), include_groups=False).reset_index()
        monthly.to_csv(out / f'validation_{split}_monthly.csv', index=False, float_format='%.4f')
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Saved metrics:', out / 'metrics.json', flush=True)
    if not args.submit:
        return
    if test is None or len(test) != 50000:
        raise ValueError('Test must have 50,000 rows')
    print(f'[{experiment_name}] fitting full train and predicting Kaggle test...', flush=True)
    xall, cats = make_xy(train, train, args.history)
    xtest, _ = make_xy(train, test, args.history)
    preds = fit_predict(args.model, xall, train['MONTHLY_RENT'].to_numpy(dtype=float), xtest, cats, args)
    if not np.isfinite(preds).all():
        raise AssertionError('Nonfinite prediction')
    reference = pd.read_csv(ROOT / 'examples' / 'submission_reference.csv')
    if len(reference) != len(test) or not reference['Id'].is_unique or reference['Id'].isna().any():
        raise ValueError('Reference submission IDs missing/duplicated/mismatched')
    dest = ROOT / 'submissions' / 'v2' / experiment_name / 'submission.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Id': reference['Id'], 'Predicted': preds}).to_csv(dest, index=False, float_format='%.6f')
    print(f'Saved Kaggle submission to {dest}; submit manually and record leaderboard RMSE.', flush=True)


if __name__ == '__main__':
    main()
