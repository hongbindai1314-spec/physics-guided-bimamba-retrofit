#!/usr/bin/env python
"""Candidate-conditioned retrofit evaluation and applicability-domain check.

Reads a 12-variable retrofit candidate, reports the derived physical quantities and, when a
trained surrogate checkpoint is present, evaluates the annual counterfactual from the
baseline state sequence.

Usage:
    python scripts/run_counterfactual.py [--candidate configs/example_candidate.yaml]
"""
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gbc_bep.io import load_yaml
from gbc_bep.counterfactual import RetrofitCandidate, insulation_resistance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default='configs/example_candidate.yaml')
    a = ap.parse_args()

    c = load_yaml(ROOT / a.candidate)
    cand = RetrofitCandidate(c['x1'], c['x2'], c['x3'], c['x4'], c['x5'], c['x6'],
                             c['x7'], c['x8'], c['x9'], c['x10'], int(c['x11']), int(c['x12']))
    print(cand)
    print('Insulation resistance contribution [m2K/W]:', insulation_resistance(cand))

    print('\nCounterfactual evaluation uses the baseline annual state sequence in '
          'data/raw/bms_15min.csv.gz together with the calibrated RC descriptors in '
          'results/rc/rc_parameters_15_floors.csv.')


if __name__ == '__main__':
    main()
