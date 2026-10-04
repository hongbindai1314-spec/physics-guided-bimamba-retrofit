#!/usr/bin/env python
from pathlib import Path
import json, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbc_bep.validation import add_pointwise_errors, stratified_summary, ad_summary, global_statistics

inp=ROOT/'data/validation/energyplus_60_validation.csv'
out=ROOT/'results/validation'; out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(inp)
add_pointwise_errors(df).to_csv(out/'pointwise_with_errors.csv',index=False)
stratified_summary(df).to_csv(out/'stratified_summary.csv',index=False)
ad_summary(df).to_csv(out/'applicability_domain_summary.csv',index=False)
stats=global_statistics(df)
(out/'global_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(stratified_summary(df).to_string(index=False))
print('\nApplicability domain:')
print(ad_summary(df).to_string(index=False))
print('\nGlobal statistics:')
print(json.dumps(stats,indent=2))
