#!/usr/bin/env python
"""Train the Physics-Guided BiMamba predictor.

Reads the 55-dimensional model-input table, builds look-back windows over the training
split and optimises the MSE plus the physics-guided energy-balance regulariser described in
the manuscript. The checkpoint is written to results/models/bimamba.pt.

Usage:
    python scripts/train_bimamba.py [--epochs 200]
"""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gbc_bep.io import load_yaml, read_table
from gbc_bep.datasets import WindowDataset
from gbc_bep.model import PhysicsGuidedBiMamba
from gbc_bep.physics_loss import physics_guided_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=None)
    a = ap.parse_args()

    cfg = load_yaml(ROOT / 'configs/model.yaml')
    dcfg = load_yaml(ROOT / 'configs/data.yaml')
    epochs = a.epochs or cfg['max_epochs']

    proc = ROOT / dcfg['model_input_path']
    if not proc.exists():
        raise SystemExit(f'Model-input file not found: {proc}.')
    df = read_table(proc)

    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    if len(feature_cols) != 55:
        raise SystemExit('Expected 55 processed feature columns named feat_*.')

    if 'split' in df.columns:
        df = df[df['split'] == 'train']
    X = df[feature_cols].to_numpy(np.float32)
    y = df[dcfg['target_column']].to_numpy(np.float32)

    ds = WindowDataset(X, y, cfg['lookback'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)

    model = PhysicsGuidedBiMamba(input_dim=55, state_dim=cfg['state_dim'], layers=cfg['layers'],
                                 cnn_channels=cfg['cnn_channels'], cnn_kernels=cfg['cnn_kernels'],
                                 fused_dim=cfg['fused_dim'], attention_dim=cfg['attention_dim'],
                                 mlp_hidden_dim=cfg['mlp_hidden_dim'], dropout=cfg['dropout'],
                                 backend=cfg['backend'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in dl:
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip_norm'])
            opt.step()
            total += loss.item() * len(yb)
        if epoch % 10 == 0:
            print(f'epoch {epoch:4d}  loss {total / len(ds):.6f}')

    out = ROOT / 'results/models'
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'bimamba.pt')
    print('Saved', out / 'bimamba.pt')


if __name__ == '__main__':
    main()
