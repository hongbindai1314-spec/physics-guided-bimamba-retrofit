#!/usr/bin/env python
from pathlib import Path
import argparse, sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbc_bep.io import load_yaml, read_table, write_table
from gbc_bep.preprocessing import causal_short_gap_fill, historical_knn_fill, trailing_iqr_replace

ap=argparse.ArgumentParser(); ap.add_argument('--input',default=None); a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/data.yaml'); pcfg=load_yaml(ROOT/'configs/preprocessing.yaml')
path=a.input or cfg['raw_bms_path']; df=read_table(ROOT/path if not Path(path).is_absolute() else path)
features=cfg['operational_features']
if len(features)!=47:
    raise SystemExit('Set the exact 47 operational feature columns in configs/data.yaml before running the pipeline.')
X=df[features].copy()
for c in features: X[c]=causal_short_gap_fill(X[c],pcfg['short_gap_max_steps'])
X=historical_knn_fill(X,pcfg['knn_k']); X=trailing_iqr_replace(X,pcfg['outlier_window_steps'],pcfg['outlier_iqr_multiplier'],pcfg['outlier_replacement_history'])
out=df[[cfg['timestamp_column'],cfg['target_column']]].copy(); out[features]=X
write_table(out,ROOT/cfg['processed_path']); print('Wrote',ROOT/cfg['processed_path'])
