from __future__ import annotations
import numpy as np


def relative_interval_width(samples, q=(0.05,0.95)):
    x=np.asarray(samples,float); lo,hi=np.quantile(x,q); mean=x.mean()
    return float((hi-lo)/mean*100.0)

def downside_p95(samples): return float(np.quantile(np.asarray(samples,float),0.95))

def sample_rc(mean,cov,n=500,seed=42): return np.random.default_rng(seed).multivariate_normal(mean,cov,size=n)

def sample_occupancy_poisson(rate,n=500,seed=42): return np.random.default_rng(seed).poisson(rate,size=(n,)+np.shape(rate))
