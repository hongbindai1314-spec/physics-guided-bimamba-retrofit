from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def causal_short_gap_fill(s: pd.Series, max_steps: int = 8) -> pd.Series:
    """Fill only short gaps using historical information.

    Linear extrapolation uses the two latest valid historical values when possible;
    otherwise the last observation is carried forward. Future values are never used.
    """
    x=s.astype(float).copy(); n=len(x); i=0
    while i<n:
        if pd.notna(x.iloc[i]): i+=1; continue
        j=i
        while j<n and pd.isna(x.iloc[j]): j+=1
        gap=j-i
        if gap<=max_steps:
            hist=x.iloc[:i].dropna()
            if len(hist)>=2:
                slope=hist.iloc[-1]-hist.iloc[-2]
                for k in range(gap): x.iloc[i+k]=hist.iloc[-1]+slope*(k+1)
            elif len(hist)==1:
                x.iloc[i:j]=hist.iloc[-1]
        i=j
    return x


def historical_knn_fill(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Past-only KNN fill for remaining missing values.

    This transparent implementation is intentionally conservative and operates row by row.
    It uses standardised observed companion features and restricts donors to earlier rows.
    """
    out=df.astype(float).copy()
    for i in range(len(out)):
        miss=out.iloc[i].isna()
        if not miss.any(): continue
        hist=out.iloc[:i]
        if len(hist)==0: continue
        obs_cols=list(out.columns[~miss & out.iloc[i].notna()])
        if not obs_cols:
            out.iloc[i]=out.iloc[i].fillna(hist.median(numeric_only=True)); continue
        complete=hist.dropna(subset=obs_cols)
        if complete.empty:
            out.iloc[i]=out.iloc[i].fillna(hist.median(numeric_only=True)); continue
        mu=complete[obs_cols].mean(); sd=complete[obs_cols].std().replace(0,1).fillna(1)
        q=((out.loc[out.index[i],obs_cols]-mu)/sd).to_numpy(float).reshape(1,-1)
        X=((complete[obs_cols]-mu)/sd).to_numpy(float)
        nn=NearestNeighbors(n_neighbors=min(k,len(complete))).fit(X)
        _, inds=nn.kneighbors(q)
        donors=complete.iloc[inds[0]]
        for col in out.columns[miss]:
            vals=donors[col].dropna()
            if len(vals): out.iat[i,out.columns.get_loc(col)]=float(vals.mean())
            elif hist[col].notna().any(): out.iat[i,out.columns.get_loc(col)]=float(hist[col].dropna().iloc[-1])
    return out


def trailing_iqr_replace(df: pd.DataFrame, window=192, multiplier=3.0, replacement_history=12) -> pd.DataFrame:
    out=df.astype(float).copy()
    for col in out.columns:
        arr=out[col].copy()
        for i in range(len(arr)):
            if i<4 or pd.isna(arr.iloc[i]): continue
            hist=arr.iloc[max(0,i-window):i].dropna()
            if len(hist)<8: continue
            q1,q3=hist.quantile([0.25,0.75]); iqr=q3-q1
            if arr.iloc[i] < q1-multiplier*iqr or arr.iloc[i] > q3+multiplier*iqr:
                recent=hist.iloc[-replacement_history:]
                arr.iloc[i]=recent.median()
        out[col]=arr
    return out
