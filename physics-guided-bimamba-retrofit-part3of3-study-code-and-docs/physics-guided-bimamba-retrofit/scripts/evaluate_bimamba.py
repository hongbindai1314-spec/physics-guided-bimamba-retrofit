#!/usr/bin/env python
"""Evaluate a trained Physics-Guided BiMamba checkpoint on the held-out test split.

Computes RMSE, MAE, MAPE, R² and CV-RMSE on the test windows and writes the per-timestep
predictions to results/model_performance/.

Usage:
    python scripts/evaluate_bimamba.py [--checkpoint results/models/bimamba.pt]
"""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gbc_bep.io import load_yaml
from gbc_bep.datasets import WindowDataset
from gbc_bep.model import PhysicsGuidedBiMamba
from gbc_bep.metrics import rmse, mae, mape, r2, cv_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='results/models/bimamba.pt')
    a = ap.parse_args()

    cfg = load_yaml(ROOT / 'configs/model.yaml')
    dcfg = load_yaml(ROOT / 'configs/data.yaml')

    df = pd.read_csv(ROOT / dcfg['model_input_path'])
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    if len(feature_cols) != 55:
        raise SystemExit('Expected 55 processed feature columns named feat_*.')

    test = df[df['split'] == 'test'] if 'split' in df else df
    X = test[feature_cols].to_numpy(np.float32)
    y = test[dcfg['target_column']].to_numpy(np.float32)

    model = PhysicsGuidedBiMamba(input_dim=55, state_dim=cfg['state_dim'], layers=cfg['layers'],
                                 cnn_channels=cfg['cnn_channels'], cnn_kernels=cfg['cnn_kernels'],
                                 fused_dim=cfg['fused_dim'], attention_dim=cfg['attention_dim'],
                                 mlp_hidden_dim=cfg['mlp_hidden_dim'], dropout=cfg['dropout'],
                                 backend=cfg['backend'])
    ckpt = ROOT / a.checkpoint
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    ds = WindowDataset(X, y, cfg['lookback'])
    with torch.no_grad():
        pred = np.concatenate([model(xb[None]).squeeze(-1).numpy() for xb, _ in ds])
    truth = y[cfg['lookback'] - 1:]

    print(f'RMSE   {rmse(truth, pred):.4f}')
    print(f'MAE    {mae(truth, pred):.4f}')
    print(f'MAPE   {mape(truth, pred):.4f}')
    print(f'R2     {r2(truth, pred):.4f}')
    print(f'CVRMSE {cv_rmse(truth, pred):.4f}')

    outdir = ROOT / 'results/model_performance'
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'timestamp': test['timestamp'].to_numpy()[cfg['lookback'] - 1:],
                  'y_true': truth, 'BiMamba_proposed': pred}).to_csv(
        outdir / 'test_predictions_bimamba.csv', index=False, float_format='%.6f')


if __name__ == '__main__':
    main()
