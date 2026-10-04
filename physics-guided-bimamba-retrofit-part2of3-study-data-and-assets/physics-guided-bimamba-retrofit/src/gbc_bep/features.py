from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from .gbc import GranularBallRegressor


def fit_55d_features(X_operational_train, rc_train, y_train, X_operational_other=None, rc_other=None,
                     purity_threshold=0.70, min_ball_size=15, nearest_balls=5, random_state=42):
    """Fit training-only scaler/GBC and build the 55-D representation.

    Dimensions are fixed to the case-study statement: 47 operational + 3 RC + 5 GBC weights.
    GBC itself is fitted on the 50-D [operational; RC] training matrix.
    """
    Xo=np.asarray(X_operational_train,float); rc=np.asarray(rc_train,float); y=np.asarray(y_train,float)
    if Xo.shape[1] != 47 or rc.shape[1] != 3:
        raise ValueError(f"expected 47 operational + 3 RC columns, got {Xo.shape[1]} + {rc.shape[1]}")
    base=np.c_[Xo,rc]
    scaler=StandardScaler().fit(base)
    base_s=scaler.transform(base)
    gbc=GranularBallRegressor(purity_threshold,min_ball_size,nearest_balls,random_state).fit(base_s,y)
    W=gbc.membership(base_s)
    X55=np.c_[base_s,W]
    result={"train":X55,"scaler":scaler,"gbc":gbc}
    if X_operational_other is not None:
        bo=np.asarray(X_operational_other,float); ro=np.asarray(rc_other,float)
        if bo.shape[1]!=47 or ro.shape[1]!=3: raise ValueError("other set must also be 47 + 3 columns")
        bs=scaler.transform(np.c_[bo,ro]); result["other"]=np.c_[bs,gbc.membership(bs)]
    return result
