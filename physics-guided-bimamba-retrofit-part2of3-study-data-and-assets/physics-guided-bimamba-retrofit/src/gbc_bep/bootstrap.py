from __future__ import annotations
import numpy as np
from .metrics import summarize


def circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Construct one circular moving-block bootstrap index vector of length n."""
    if n <= 0 or block_length <= 0:
        raise ValueError("n and block_length must be positive")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    blocks = [(s + np.arange(block_length)) % n for s in starts]
    return np.concatenate(blocks)[:n]


def metric_intervals(y, predictions: dict[str, np.ndarray], B=10000, block_length=672, alpha=0.05, seed=2026):
    """Dependence-aware percentile CIs using identical blocks for all models."""
    y = np.asarray(y, float)
    preds = {k: np.asarray(v, float) for k,v in predictions.items()}
    n = len(y)
    if any(len(v) != n for v in preds.values()):
        raise ValueError("all predictions must have the same length as y")
    rng = np.random.default_rng(seed)
    metric_names = ["MAE","RMSE","MAPE_pct","R2","CV_RMSE_pct"]
    draws = {m:{q:[] for q in metric_names} for m in preds}
    for _ in range(B):
        idx = circular_block_indices(n, block_length, rng)
        yy = y[idx]
        for model, pp in preds.items():
            s = summarize(yy, pp[idx])
            for q in metric_names: draws[model][q].append(s[q])
    out = {}
    for model, qs in draws.items():
        out[model] = {}
        for q, vals in qs.items():
            lo, hi = np.quantile(vals, [alpha/2, 1-alpha/2])
            out[model][q] = {"estimate": summarize(y,preds[model])[q], "lower":float(lo), "upper":float(hi)}
    return out


def paired_block_bootstrap_pvalue(y, pred_a, pred_b, loss="absolute", B=10000, block_length=672, seed=2026):
    """Two-sided null-centred paired circular block-bootstrap p-value.

    Positive observed difference means pred_a has larger loss than pred_b.
    """
    y=np.asarray(y,float); a=np.asarray(pred_a,float); b=np.asarray(pred_b,float)
    if loss == "absolute":
        da=np.abs(y-a); db=np.abs(y-b)
    elif loss == "squared":
        da=(y-a)**2; db=(y-b)**2
    elif loss == "ape":
        den=np.maximum(np.abs(y),1e-12); da=np.abs(y-a)/den; db=np.abs(y-b)/den
    else:
        raise ValueError("loss must be absolute, squared or ape")
    d=da-db
    observed=float(d.mean())
    centered=d-observed
    rng=np.random.default_rng(seed)
    exceed=0
    for _ in range(B):
        idx=circular_block_indices(len(d),block_length,rng)
        stat=float(centered[idx].mean())
        if abs(stat) >= abs(observed): exceed += 1
    return {"mean_loss_difference":observed,"p_value":(1+exceed)/(B+1)}
