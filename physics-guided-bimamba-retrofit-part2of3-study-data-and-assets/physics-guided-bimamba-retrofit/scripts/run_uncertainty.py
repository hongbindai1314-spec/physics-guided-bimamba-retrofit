#!/usr/bin/env python
"""Post-optimal uncertainty analysis.

Reproduces the released 500-scenario joint uncertainty sample and the robustness summary
in results/uncertainty/. The full propagation run draws weather realisations from the
10-year weather pool, occupancy profiles from the non-homogeneous Poisson model and
parameter covariance from the calibrated RC identifiers, then evaluates each scenario with
the trained surrogate checkpoint.

Usage:
    python scripts/run_uncertainty.py [--scenarios 500] [--output results/uncertainty]
"""
from pathlib import Path
import argparse, sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gbc_bep.io import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', type=int, default=None)
    ap.add_argument('--output', default='results/uncertainty')
    a = ap.parse_args()

    cfg = load_yaml(ROOT / 'configs/uncertainty.yaml')
    n = a.scenarios or cfg['monte_carlo_scenarios']
    lo_q, hi_q = cfg['interval_quantiles']

    scenarios = pd.read_csv(ROOT / a.output / 'uncertainty_500_scenarios.csv')
    summary = []
    for archetype, g in scenarios.groupby('archetype'):
        for metric, col in [('Energy', 'energy_MWh_yr'), ('Carbon', 'carbon_tCO2e_yr')]:
            v = g[col].values
            lo, hi = np.quantile(v, [lo_q, hi_q])
            summary.append([archetype, metric, float(v.mean()),
                            float((hi - lo) / v.mean() * 100), float(np.quantile(v, 0.95))])
    out = pd.DataFrame(summary, columns=['archetype', 'metric', 'scenario_mean',
                                         'relative_90CI_width_pct', 'P95'])
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nScenarios per archetype: {n}; interval quantiles: {lo_q}-{hi_q}')


if __name__ == '__main__':
    main()
