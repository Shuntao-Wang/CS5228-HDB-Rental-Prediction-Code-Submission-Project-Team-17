"""V4 comparative rent feature ablation on strictly chronological validation.

Run from the repository root:
    python scripts/test_v4_comparables.py
    python scripts/run_v4_experiments.py --mode comparables --split both
    python scripts/run_v4_experiments.py --mode extended --split both
    python scripts/run_v4_experiments.py --mode comparables --submit --submit-only

V1/V2/V3 files are not modified.  Historical features use only RENT labels
from earlier MONTHS of the supplied reference training split; never holdout
or query labels. Default CatBoost hyperparameters are deliberately V2's.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

from run_v2_experiments import ROOT, DATA, SPLITS, SPLIT_END, _month_asof, make_xy, fit_predict, score

# Controlled ablation stages: the first two groups only for 'comparables';
# extended introduces three additional high-cardinality, potentially sparse groups.
GROUP_CONFIG = {
    'town_type_area': (['TOWN', 'FLAT_TYPE', '_AREA_BAND'], 'HIST_TOWN_TYPE_MEAN'),
    'street_type': (['TOWN', 'STREET', 'FLAT_TYPE'], 'HIST_TOWN_TYPE_MEAN'),
    'street_type_area': (['TOWN', 'STREET', 'FLAT_TYPE', '_AREA_BAND'], 'HIST_TOWN_TYPE_MEAN'),
    'block_type_area': (['BLOCK_KEY', 'FLAT_TYPE', '_AREA_BAND'], 'HIST_BLOCK_TYPE_MEAN'),
    'type_area': (['FLAT_TYPE', '_AREA_BAND'], 'HIST_GLOBAL_MEAN'),
}


def _get_band(df: pd.DataFrame, bin_width: float) -> pd.Series:
    area = pd.to_numeric(df['FLOOR_AREA_SQM'], errors='coerce')
    if area.isna().any() or not np.isfinite(area.to_numpy()).all():
        raise ValueError('Non-finite FLOOR_AREA_SQM')
    if (area <= 0).any():
        raise ValueError('Nonpositive FLOOR_AREA_SQM')
    return np.floor(area / bin_width).astype('int32')


def make_comp_features(reference: pd.DataFrame, query: pd.DataFrame,
                       mode: str, smoothing: float = 25.0,
                       bin_width: float = 10.0) -> pd.DataFrame:
    """Historical comparable-house means/counts, strictly excluding query month.

    Only the reference *training fold* label column is read. All groups contain
    physical descriptors, not targets. A row from month M uses only reference
    rows with reference month < M (including when reference == query).
    """
    if mode not in {'comparables', 'extended'}:
        raise ValueError('mode must be comparables or extended')
    if smoothing <= 0 or bin_width <= 0:
        raise ValueError('smoothing and bin_width must be >0')
    if reference.empty or not np.isfinite(reference['MONTHLY_RENT'].to_numpy(dtype=float)).all():
        raise ValueError('Reference empty or invalid targets')
    needed = {'TOWN','FLAT_TYPE','STREET','BLOCK_KEY','FLOOR_AREA_SQM','MONTH_INDEX'}
    if needed - set(reference) or needed - set(query):
        raise ValueError(f'Missing columns: {needed - set(reference) | needed - set(query)}')
    # Derive the same deterministic area bins for reference and query.
    ref = reference.copy()
    qry = query.copy()
    ref['_AREA_BAND'] = _get_band(ref, bin_width)
    qry['_AREA_BAND'] = _get_band(qry, bin_width)
    # Use ONLY the same fold's prior means as shrinkage fallback.
    # Import here to avoid recomputing the existing categorical feature matrix.
    from run_v2_experiments import make_past_rent_features
    base = make_past_rent_features(reference, query, smoothing=25.0)
    levels = ['town_type_area', 'street_type']
    if mode == 'extended':
        levels += ['street_type_area', 'block_type_area', 'type_area']
    features = pd.DataFrame(index=query.index)
    for name in levels:
        keys, parent = GROUP_CONFIG[name]
        matched = _month_asof(ref, qry, keys)
        cnt = matched['_n'].fillna(0).to_numpy(dtype=float)
        summ = matched['_s'].fillna(0).to_numpy(dtype=float)
        fallback = base[parent].to_numpy(dtype=float)
        mean = np.divide(summ + smoothing * np.nan_to_num(fallback, nan=0.0),
                         cnt + smoothing, out=np.full(len(qry), np.nan),
                         where=(cnt+smoothing > 0) & np.isfinite(fallback))
        prefix = f'COMP_{name.upper()}'
        features[f'{prefix}_MEAN'] = mean
        features[f'{prefix}_LOG_COUNT'] = np.log1p(cnt)
    if not features.index.equals(query.index):
        raise AssertionError('Feature index changed')
    return features


def build_xy(reference: pd.DataFrame, query: pd.DataFrame, mode: str,
             smoothing: float = 25.0, bin_width: float = 10.0):
    if mode not in {'prior','comparables','extended'}:
        raise ValueError(mode)
    x, cats = make_xy(reference, query, 'prior')
    if mode != 'prior':
        extra = make_comp_features(reference, query, mode, smoothing, bin_width)
        for c in extra.columns:
            x[c] = extra[c].to_numpy(dtype=float)
    return x, cats


def per_month_report(valid: pd.DataFrame, pred: np.ndarray, reference: pd.DataFrame):
    data = pd.DataFrame({'month':valid['RENT_APPROVAL_DATE'].to_numpy(),
                         'month_index':valid['MONTH_INDEX'].to_numpy(),
                         'actual':valid['MONTHLY_RENT'].to_numpy(dtype=float),
                         'pred':np.asarray(pred,dtype=float)})
    data['error'] = data['pred'] - data['actual']
    g = data.groupby(['month','month_index'],sort=True).agg(
        n=('error','size'), mse=('error',lambda z: float(np.mean(z*z))),
        mae=('error',lambda z: float(np.mean(np.abs(z)))),
        bias=('error','mean'), actual_mean=('actual','mean'), pred_mean=('pred','mean'),
    ).reset_index()
    g['rmse'] = np.sqrt(g.pop('mse'))
    g['months_after_train_end'] = g['month_index'] - int(reference['MONTH_INDEX'].max())
    return g[['month','months_after_train_end','n','rmse','mae','bias','actual_mean','pred_mean']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',default='comparables',choices=['prior','comparables','extended'])
    parser.add_argument('--split',default='both',choices=['long','short','both'])
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--submit-only',action='store_true',help='Requires --submit; skips validation')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--cat-iterations',type=int,default=300)
    parser.add_argument('--smoothing',type=float,default=25.0)
    parser.add_argument('--area-bin-width',type=float,default=10.0)
    args=parser.parse_args()
    if args.submit_only and not args.submit:
        parser.error('--submit-only requires --submit')
    if args.cat_iterations<1 or args.threads<1 or args.smoothing<=0 or args.area_bin_width<=0:
        parser.error('Invalid numeric argument')
    train=pd.read_csv(DATA/'train_features.csv.gz')
    if len(train)!=150000 or 'MONTHLY_RENT' not in train:
        raise ValueError('Expected original V1 train_features (150000 labeled rows)')
    version=f'catboost_{args.mode}'
    out=ROOT/'experiments'/'v4'/version
    out.mkdir(parents=True,exist_ok=True)
    record={'version':'V4','model':'catboost','feature_mode':args.mode,
            'params':{'cat_iterations':args.cat_iterations,'threads':args.threads,
                      'smoothing':args.smoothing,'area_bin_width':args.area_bin_width,
                      'depth':6,'learning_rate':0.06,'l2_leaf_reg':5,'seed':42},
            'validation_policy':'train labels strictly earlier than validation start; all history query rows exclude their own month',
            'generated_at_utc':datetime.now(timezone.utc).isoformat(),'holdouts':[]}
    if not args.submit_only:
        for split in (['long','short'] if args.split=='both' else [args.split]):
            subset=train[train.RENT_APPROVAL_DATE<SPLITS[split]].copy()
            valid=train[(train.RENT_APPROVAL_DATE>=SPLITS[split]) & (train.RENT_APPROVAL_DATE<=SPLIT_END)].copy()
            if int(subset.MONTH_INDEX.max()) >= int(valid.MONTH_INDEX.min()):
                raise AssertionError('Chronological partition violated')
            print(f'[{version} {split}] train={len(subset):,} validate={len(valid):,}',flush=True)
            xtr,cats=build_xy(subset,subset,args.mode,args.smoothing,args.area_bin_width)
            xva,_=build_xy(subset,valid,args.mode,args.smoothing,args.area_bin_width)
            assert xtr.columns.tolist()==xva.columns.tolist()
            pred=fit_predict('catboost',xtr,subset.MONTHLY_RENT.to_numpy(dtype=float),xva,cats,args)
            m=score(valid.MONTHLY_RENT.to_numpy(dtype=float),pred)
            m.update({'split':split,'train_end':str(subset.RENT_APPROVAL_DATE.max()),
                      'valid_start':str(valid.RENT_APPROVAL_DATE.min()),'valid_end':str(valid.RENT_APPROVAL_DATE.max()),
                      'n_train':len(subset),'n_valid':len(valid),'n_features':xtr.shape[1]})
            record['holdouts'].append(m)
            per_month_report(valid,pred,subset).to_csv(out/f'validation_{split}_monthly.csv',index=False,float_format='%.6f')
            print(f'  RMSE={m["rmse"]:.3f} MAE={m["mae"]:.3f} bias={m["mean_error_pred_minus_actual"]:+.3f}; {xtr.shape[1]} features',flush=True)
        # Preserve complete previous metrics only if exact same experiment config.
        metrics_file=out/'metrics.json'
        if metrics_file.exists():
            old=json.loads(metrics_file.read_text(encoding='utf-8'))
            if old.get('params')==record['params'] and old.get('feature_mode')==record['feature_mode']:
                merged={m['split']:m for m in old.get('holdouts',[])}
                merged.update({m['split']:m for m in record['holdouts']})
                record['holdouts']=[merged[k] for k in ['long','short'] if k in merged]
        metrics_file.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print('Saved',metrics_file,flush=True)
    if args.submit:
        test=pd.read_csv(DATA/'test_features.csv.gz')
        xtr,cats=build_xy(train,train,args.mode,args.smoothing,args.area_bin_width)
        xte,_=build_xy(train,test,args.mode,args.smoothing,args.area_bin_width)
        assert xtr.columns.tolist()==xte.columns.tolist()
        print(f'[{version} submit] training full 150,000 rows; predicting 50,000 test rows',flush=True)
        pred=fit_predict('catboost',xtr,train.MONTHLY_RENT.to_numpy(dtype=float),xte,cats,args)
        if len(pred)!=50000 or not np.isfinite(pred).all():
            raise AssertionError('Bad test predictions')
        # Submitted Id is position matching original test row order, established
        # by the original Kaggle reference submission, NOT a model feature.
        sample=ROOT/'examples'/'submission_reference.csv'
        if sample.exists():
            ids=pd.read_csv(sample)['Id'].to_numpy()
            if len(ids)!=len(pred) or len(np.unique(ids))!=len(ids):
                raise AssertionError('Reference submission IDs invalid')
        else:
            ids=np.arange(len(pred))
            print('Warning: examples/submission_reference.csv missing; using verified 0..49999 ordering',flush=True)
        dest=ROOT/'submissions'/'v4'/version/'submission.csv'
        dest.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'Id':ids,'Predicted':pred}).to_csv(dest,index=False,float_format='%.8f')
        print('Saved',dest,flush=True)

if __name__=='__main__':
    main()
