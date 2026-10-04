#!/usr/bin/env python
"""Recompute the archived model-comparison metrics from the released prediction table."""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]

pred = pd.read_csv(ROOT / 'results/model_performance/test_predictions_14_models.csv.gz')
y = pred.y_true.to_numpy()
rows = []
for c in pred.columns:
    if c in ('timestamp', 'y_true'):
        continue
    p = pred[c].to_numpy()
    e = p - y
    rows.append([
        c,
        np.sqrt(np.mean(e ** 2)),
        np.mean(np.abs(e)),
        100 * np.mean(np.abs(e) / y),
        1 - np.mean(e ** 2) / np.var(y),
        100 * np.sqrt(np.mean(e ** 2)) / np.mean(y),
    ])

out = pd.DataFrame(rows, columns=['model', 'RMSE', 'MAE', 'MAPE', 'R2', 'CVRMSE'])
print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print('\nBlock-bootstrap summaries:  results/bootstrap/block_bootstrap_10000_summary.csv')
print('Pareto archive:             results/optimization/pareto_200.csv')
print('Uncertainty scenarios:      results/uncertainty/uncertainty_500.csv')
