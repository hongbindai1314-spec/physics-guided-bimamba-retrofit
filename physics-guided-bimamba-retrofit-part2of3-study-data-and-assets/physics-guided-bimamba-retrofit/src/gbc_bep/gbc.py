from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans

@dataclass
class Ball:
    indices: np.ndarray
    center: np.ndarray
    radius: float
    target_var: float
    covariance: np.ndarray

class GranularBallRegressor:
    """Regression-oriented granular-ball decomposition.

    Implements the target-variance purity criterion and recursive splitting used in the study.
    Splitting is performed with deterministic 2-means, which reproduces the reported ball
    structure and membership assignments.
    """
    def __init__(self, purity_threshold=0.70, min_ball_size=15, nearest_balls=5, random_state=42):
        self.purity_threshold=purity_threshold; self.min_ball_size=min_ball_size
        self.nearest_balls=nearest_balls; self.random_state=random_state
        self.balls=[]; self.global_max_var_=None

    def _make_ball(self,X,y,idx):
        Xi=X[idx]; center=Xi.mean(0); radius=float(np.linalg.norm(Xi-center,axis=1).max(initial=0))
        cov=np.cov(Xi,rowvar=False) if len(Xi)>1 else np.eye(X.shape[1])
        cov=np.atleast_2d(cov)+np.eye(X.shape[1])*1e-6
        return Ball(idx,center,radius,float(np.var(y[idx])),cov)

    def _purity(self,b):
        return 1.0-b.target_var/(self.global_max_var_+1e-12)

    def _split(self,X,idx):
        if len(idx)<2*self.min_ball_size: return None
        lab=KMeans(n_clusters=2,n_init=10,random_state=self.random_state).fit_predict(X[idx])
        a=idx[lab==0]; b=idx[lab==1]
        if min(len(a),len(b))<self.min_ball_size: return None
        return a,b

    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float); self.X_=X; self.y_=y
        root_idx=np.arange(len(X)); self.global_max_var_=max(float(np.var(y)),1e-12)
        queue=[root_idx]; balls=[]
        while queue:
            idx=queue.pop(0); ball=self._make_ball(X,y,idx)
            if self._purity(ball)>=self.purity_threshold or len(idx)<2*self.min_ball_size:
                balls.append(ball); continue
            split=self._split(X,idx)
            if split is None: balls.append(ball)
            else: queue.extend(split)
        self.balls=balls
        self.centers_=np.vstack([b.center for b in balls])
        self.radii_=np.array([max(b.radius,1e-6) for b in balls])
        return self

    def membership(self,X):
        X=np.asarray(X,float)
        d2=((X[:,None,:]-self.centers_[None,:,:])**2).sum(-1)
        k=min(self.nearest_balls,len(self.balls))
        nearest=np.argpartition(d2,k-1,axis=1)[:,:k]
        out=np.zeros((len(X),k),float)
        for i in range(len(X)):
            ids=nearest[i]; bw=self.radii_[ids]
            raw=np.exp(-d2[i,ids]/(2*bw*bw))
            out[i]=raw/(raw.sum()+1e-12)
        # ordered by ascending distance for a stable fixed k-vector
        order=np.argsort(np.take_along_axis(d2,nearest,axis=1),axis=1)
        return np.take_along_axis(out,order,axis=1)
