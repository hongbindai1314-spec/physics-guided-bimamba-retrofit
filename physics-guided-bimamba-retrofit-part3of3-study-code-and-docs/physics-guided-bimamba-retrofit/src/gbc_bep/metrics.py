from __future__ import annotations
import numpy as np


def _a(x): return np.asarray(x, dtype=float)

def mae(y, p):
    y, p = _a(y), _a(p)
    return float(np.mean(np.abs(y-p)))

def rmse(y, p):
    y, p = _a(y), _a(p)
    return float(np.sqrt(np.mean((y-p)**2)))

def mape(y, p, eps=1e-12):
    y, p = _a(y), _a(p)
    mask = np.abs(y) > eps
    return float(np.mean(np.abs((p[mask]-y[mask])/y[mask]))*100.0)

def r2(y, p):
    y, p = _a(y), _a(p)
    denom = np.sum((y-y.mean())**2)
    return float(1.0 - np.sum((p-y)**2)/denom) if denom > 0 else float('nan')

def cv_rmse(y, p):
    y = _a(y)
    return 100.0 * rmse(y, p) / float(np.mean(y))

def summarize(y, p):
    return {"MAE": mae(y,p), "RMSE": rmse(y,p), "MAPE_pct": mape(y,p), "R2": r2(y,p), "CV_RMSE_pct": cv_rmse(y,p)}
