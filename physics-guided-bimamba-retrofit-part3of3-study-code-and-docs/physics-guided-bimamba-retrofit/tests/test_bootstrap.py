import numpy as np
from gbc_bep.bootstrap import circular_block_indices, metric_intervals

def test_circular_block_indices_length():
    rng=np.random.default_rng(1); idx=circular_block_indices(100,12,rng)
    assert len(idx)==100 and idx.min()>=0 and idx.max()<100

def test_metric_intervals_runs():
    rng=np.random.default_rng(2); y=np.arange(200,dtype=float)+10; p=y+rng.normal(0,1,200)
    out=metric_intervals(y,{'m':p},B=20,block_length=24,seed=3)
    assert 'RMSE' in out['m'] and out['m']['RMSE']['upper']>=out['m']['RMSE']['lower']
