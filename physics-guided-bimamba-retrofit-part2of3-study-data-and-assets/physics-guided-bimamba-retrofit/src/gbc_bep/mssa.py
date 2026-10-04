from __future__ import annotations
import numpy as np


def _trajectory(x: np.ndarray, L: int) -> np.ndarray:
    n=len(x); K=n-L+1
    return np.column_stack([x[i:i+L] for i in range(K)])


def _hankelize(mat: np.ndarray) -> np.ndarray:
    L,K=mat.shape; n=L+K-1
    out=np.zeros(n); cnt=np.zeros(n)
    for i in range(L):
        for j in range(K):
            out[i+j]+=mat[i,j]; cnt[i+j]+=1
    return out/cnt


def terminal_mssa(history: np.ndarray, embedding_length=96, retained_components=15) -> np.ndarray:
    """Return only the reconstructed terminal multivariate vector.

    history shape: [W, D]. No sample after the terminal row is accessed.
    """
    X=np.asarray(history,float)
    W,D=X.shape
    if W < embedding_length: raise ValueError("history shorter than embedding length")
    mats=[_trajectory(X[:,j],embedding_length) for j in range(D)]
    stacked=np.vstack(mats)
    U,S,Vt=np.linalg.svd(stacked,full_matrices=False)
    m=min(retained_components,len(S))
    rec=(U[:,:m]*S[:m])@Vt[:m]
    terminals=[]
    for j in range(D):
        block=rec[j*embedding_length:(j+1)*embedding_length]
        terminals.append(_hankelize(block)[-1])
    return np.asarray(terminals)


def causal_terminal_series(X: np.ndarray, support_window=672, embedding_length=96, retained_components=15) -> np.ndarray:
    X=np.asarray(X,float)
    out=np.full_like(X,np.nan,dtype=float)
    for t in range(support_window-1,len(X)):
        out[t]=terminal_mssa(X[t-support_window+1:t+1],embedding_length,retained_components)
    return out
