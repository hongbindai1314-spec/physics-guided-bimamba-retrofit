#!/usr/bin/env python
from pathlib import Path
import argparse, json, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbc_bep.bootstrap import metric_intervals, paired_block_bootstrap_pvalue

ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--target',required=True); ap.add_argument('--pred',nargs='+',required=True)
ap.add_argument('--B',type=int,default=10000); ap.add_argument('--block-length',type=int,default=672)
a=ap.parse_args(); df=pd.read_csv(a.input); y=df[a.target].to_numpy(); preds={c:df[c].to_numpy() for c in a.pred}
ci=metric_intervals(y,preds,B=a.B,block_length=a.block_length)
out={'confidence_intervals':ci,'paired_tests':{}}
if len(a.pred)>1:
    proposed=a.pred[0]
    for other in a.pred[1:]:
        out['paired_tests'][other]={
          'MAE':paired_block_bootstrap_pvalue(y,preds[other],preds[proposed],'absolute',a.B,a.block_length),
          'MAPE':paired_block_bootstrap_pvalue(y,preds[other],preds[proposed],'ape',a.B,a.block_length),
          'MSE':paired_block_bootstrap_pvalue(y,preds[other],preds[proposed],'squared',a.B,a.block_length)}
Path(ROOT/'results/bootstrap').mkdir(parents=True,exist_ok=True)
Path(ROOT/'results/bootstrap/block_bootstrap.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
