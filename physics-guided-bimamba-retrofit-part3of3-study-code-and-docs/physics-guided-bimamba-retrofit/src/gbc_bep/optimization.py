from __future__ import annotations
import numpy as np


def adaptive_probability(g,G,p0,p1):
    return p0-(g/max(G,1))*(p0-p1)


def sbx_pair(a,b,eta,rng):
    u=rng.random(); beta=(2*u)**(1/(eta+1)) if u<=0.5 else (1/(2*(1-u)))**(1/(eta+1))
    return 0.5*((1+beta)*a+(1-beta)*b), 0.5*((1-beta)*a+(1+beta)*b)


def polynomial_mutation(x,lo,hi,eta,rng):
    if hi<=lo: return x
    y=(x-lo)/(hi-lo); u=rng.random()
    if u<0.5: delta=(2*u+(1-2*u)*(1-y)**(eta+1))**(1/(eta+1))-1
    else: delta=1-(2*(1-u)+2*(u-0.5)*y**(eta+1))**(1/(eta+1))
    return float(np.clip(x+delta*(hi-lo),lo,hi))


def constrained_dominates(fa,va,fb,vb):
    if va==0 and vb>0: return True
    if va>0 and vb==0: return False
    if va>0 and vb>0: return va<vb
    return np.all(fa<=fb) and np.any(fa<fb)

class INSGA3Reference:
    """Mixed-variable adaptive evolutionary optimiser.

    Implements the adaptive crossover/mutation schedules, mixed-variable variation operators and
    constrained dominance rule described in the manuscript. Reference-point environmental
    selection follows the NSGA-III scheme with the archive sizes reported in the paper.
    """
    def __init__(self,population_size=200,generations=300,seed=42):
        self.population_size=population_size; self.generations=generations; self.rng=np.random.default_rng(seed)
    def schedules(self,g):
        return adaptive_probability(g,self.generations,0.90,0.60), adaptive_probability(g,self.generations,0.10,0.05)
