"""Paired V5 vs V8 ablation, with optional candidate Kaggle CSV.

Examples:
 python scripts/run_v8_experiments.py --variant shape --model cat_mid --seeds 1
 python scripts/run_v8_experiments.py --variant premium --model cat_mid --seeds 1
 python scripts/run_v8_experiments.py --variant trend --model cat_mid --seeds 1
 python scripts/run_v8_experiments.py --variant all --blend --seeds 3 --submit
"""
from __future__ import annotations
import argparse,json,pickle
import numpy as np
import pandas as pd
from run_v2_experiments import ROOT,score
from run_v5_experiments import VARIANTS,fit_predict
from v8_features import SHAPE_CATS
BASE=ROOT/'experiments/v5/feature_cache.pkl'
EXTRAS=ROOT/'experiments/v8'
FAMILIES={'shape':('V8_SHAPE_',),'premium':('V8_PREMIUM_',),
          'trend':('V8_TREND_',),'shape_premium':('V8_SHAPE_','V8_PREMIUM_'),
          'all':('V8_SHAPE_','V8_PREMIUM_','V8_TREND_')}

def attach(base,extra,variant,target):
    cols=[c for c in extra['xtr'].columns if c.startswith(FAMILIES[variant])]
    if not cols:raise AssertionError('No V8 features selected')
    result=base.copy()
    for key in ('xtr',target):
        if len(base[key])!=len(extra[key]):raise AssertionError('V8 row count mismatch')
        if set(base[key].columns)&set(cols):raise AssertionError('Feature collision')
        result[key]=pd.concat([base[key].reset_index(drop=True),
                               extra[key][cols].reset_index(drop=True)],axis=1)
    assert result['xtr'].columns.tolist()==result[target].columns.tolist()
    result['cats']=list(base['cats'])+[c for c in SHAPE_CATS if c in cols]
    return result

def predict(cache,models,seeds,threads,target='xva'):
    p=np.zeros(len(cache[target]))
    for model in models:
        lib,params=VARIANTS[model]
        for j in range(seeds):
            p+=fit_predict(lib,params,cache['xtr'],cache['ytr'],[cache[target]],
                           cache['cats'],42+j,threads)[0]
    return p/(len(models)*seeds)

def read_extras(split):
    path=EXTRAS/f'extras_{split}.pkl'
    if not path.exists():raise FileNotFoundError(f'{path} missing; run v8_prepare_features.py')
    with path.open('rb') as f:return pickle.load(f)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant',choices=sorted(FAMILIES),required=True)
    ap.add_argument('--model',choices=['cat_reg','cat_mid','lgb_reg'],default='cat_mid')
    ap.add_argument('--blend',action='store_true')
    ap.add_argument('--seeds',type=int,default=1)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--submit',action='store_true')
    ap.add_argument('--round-step',type=int,choices=[0,50,100],default=0)
    a=ap.parse_args()
    if a.seeds<1 or a.threads<1:ap.error('seeds and threads must be positive')
    if not BASE.is_file():raise FileNotFoundError('Run v5_cache_features.py first')
    models=['cat_reg','cat_mid','lgb_reg'] if a.blend else [a.model]
    name=f'{a.variant}_{"blend3" if a.blend else a.model}_s{a.seeds}'
    out=ROOT/'experiments/v8/runs'/name;out.mkdir(parents=True,exist_ok=True)
    with BASE.open('rb') as f:cache=pickle.load(f)
    results=[]
    for split in ('long','short'):
        b=cache[split];v=attach(b,read_extras(split),a.variant,'xva')
        print(f'[{split}] fitting V5 control',flush=True)
        p0=predict(b,models,a.seeds,a.threads)
        print(f'[{split}] fitting V8 {a.variant}',flush=True)
        p1=predict(v,models,a.seeds,a.threads)
        y=b['yva'];r0=score(y,p0)['rmse'];r1=score(y,p1)['rmse']
        rounds={str(step):score(y,p1 if step==0 else np.round(p1/step)*step)['rmse']
                for step in (0,50,100)}
        results.append({'split':split,'rmse_v5':r0,'rmse_v8':r1,
                        'delta_v8_minus_v5':r1-r0,'v8_rounding_rmse':rounds})
        pd.DataFrame({'month':b['dva'],'actual':y,'pred_v5':p0,'pred_v8':p1}).to_csv(
            out/f'valid_{split}.csv',index=False)
        print(f'{split}: V5 {r0:.5f}, V8 {r1:.5f}, delta {r1-r0:+.5f}, '
              f'round50 {rounds["50"]:.5f}, round100 {rounds["100"]:.5f}',flush=True)
    metrics={'variant':a.variant,'models':models,'seeds':a.seeds,'results':results,
             'note':'Local paired time validation, not Kaggle score'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    if a.submit:
        v=attach(cache['full'],read_extras('full'),a.variant,'xte')
        p=predict(v,models,a.seeds,a.threads,'xte')
        if a.round_step:p=np.round(p/a.round_step)*a.round_step
        if len(p)!=50000 or not np.isfinite(p).all():raise AssertionError('Invalid test predictions')
        ids=pd.read_csv(ROOT/'examples/submission_reference.csv')['Id']
        if len(ids)!=len(p) or not ids.is_unique:raise AssertionError('Invalid submission IDs')
        dest=ROOT/'submissions/v8'/name/'submission.csv'
        dest.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'Id':ids.to_numpy(),'Predicted':p}).to_csv(dest,index=False,float_format='%.8f')
        print('Candidate saved (NOT uploaded):',dest,flush=True)
if __name__=='__main__':main()
