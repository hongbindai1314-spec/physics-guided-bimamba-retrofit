from __future__ import annotations
import numpy as np


def nearest_mahalanobis(X, centers, covariances):
    X=np.asarray(X,float); centers=np.asarray(centers,float); covs=np.asarray(covariances,float)
    out=np.empty(len(X))
    for i,x in enumerate(X):
        best=np.inf
        for mu,cov in zip(centers,covs):
            inv=np.linalg.pinv(cov)
            d=float(np.sqrt((x-mu)@inv@(x-mu)))
            best=min(best,d)
        out[i]=best
    return out


def fit_threshold(train_distances, percentile=95):
    return float(np.percentile(np.asarray(train_distances,float),percentile))


def annual_coverage(distances, threshold):
    d=np.asarray(distances,float)
    return float(np.mean(d<=threshold)*100.0)


def classify_coverage(coverage_pct, minimum_pct=95.0):
    return "In-domain" if coverage_pct>=minimum_pct else "Out-of-domain"
