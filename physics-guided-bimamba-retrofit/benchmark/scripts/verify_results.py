#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd,numpy as np
from pipeline.core import PhysicsMLP,metrics,CANDIDATES,LOW,HIGH,fronts

df=pd.read_csv(ROOT/'data/features_55d.csv.gz');df=df[df.split=='test']
X=df[[f'feat_{i:02d}' for i in range(55)]].to_numpy()
model=PhysicsMLP.load(ROOT/'models/physics_mlp.npz');p=model.predict(X)
saved=pd.read_csv(ROOT/'results/test_predictions.csv.gz')
error=float(np.max(abs(saved.physics_MLP.to_numpy()-p)))
assert error<1e-5,(error,'checkpoint does not reproduce stored predictions')
pareto=pd.read_csv(ROOT/'results/pareto_front.csv')
C=pareto[CANDIDATES].to_numpy();F=pareto.iloc[:,12:].to_numpy()
assert np.all(C>=LOW-1e-7) and np.all(C<=HIGH+1e-7)
assert len(fronts(F)[0])==len(F)
checked=0
for line in (ROOT/'RESULTS_SHA256.txt').read_text().splitlines():
    digest,path=line.split('  ',1)
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    checked+=1
report=dict(checkpoint_prediction_max_abs_diff=error,sha256_files_verified=checked,pareto_n=len(F),
             model_architecture='NumPy physics-guided multilayer perceptron',test_metrics=metrics(saved.y_true.to_numpy(),p))
print(json.dumps(report,indent=2))
