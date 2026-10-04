#!/usr/bin/env python
from pathlib import Path
import argparse, sys, pandas as pd, numpy as np, pickle
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbc_bep.io import load_yaml
from gbc_bep.features import fit_55d_features

ap=argparse.ArgumentParser()
ap.add_argument('--input',default='data/processed/features_with_rc.parquet',help='Table containing timestamp, target, 47 operational columns, tau_h/kappa_w_k/gamma_m2')
a=ap.parse_args(); dcfg=load_yaml(ROOT/'configs/data.yaml'); gcfg=load_yaml(ROOT/'configs/gbc.yaml')
features=dcfg['operational_features']
if len(features)!=47: raise SystemExit('Set the exact 47 operational feature names in configs/data.yaml.')
df=pd.read_parquet(ROOT/a.input)
for c in ['tau_h','kappa_w_k','gamma_m2']:
    if c not in df: raise SystemExit(f'Missing RC descriptor column: {c}')
ts=pd.to_datetime(df[dcfg['timestamp_column']]); train=(ts>=dcfg['splits']['train'][0])&(ts<=dcfg['splits']['train'][1]); other=~train
res=fit_55d_features(df.loc[train,features].to_numpy(),df.loc[train,['tau_h','kappa_w_k','gamma_m2']].to_numpy(),df.loc[train,dcfg['target_column']].to_numpy(),df.loc[other,features].to_numpy(),df.loc[other,['tau_h','kappa_w_k','gamma_m2']].to_numpy(),gcfg['purity_threshold'],gcfg['min_ball_size'],gcfg['nearest_balls'],gcfg['split_random_state'])
cols=[f'feat_{i:02d}' for i in range(55)]
out=pd.DataFrame(index=df.index,columns=cols,dtype=float); out.loc[train,cols]=res['train']; out.loc[other,cols]=res['other']
final=df[[dcfg['timestamp_column'],dcfg['target_column']]].join(out)
final.to_parquet(ROOT/'data/processed/model_55d.parquet',index=False)
with open(ROOT/'data/processed/feature_objects.pkl','wb') as f: pickle.dump({'scaler':res['scaler'],'gbc':res['gbc']},f)
print('Wrote data/processed/model_55d.parquet and feature_objects.pkl')
