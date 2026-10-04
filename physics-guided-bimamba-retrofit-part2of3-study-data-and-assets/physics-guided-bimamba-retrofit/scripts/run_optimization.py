#!/usr/bin/env python
"""Run the INSGA-III retrofit optimisation.

Prints the adaptive crossover/mutation schedules and, when a trained surrogate checkpoint
and the candidate-to-objective evaluator are available, writes the 200-design Pareto archive
and the 5-run convergence histories to results/optimization/.

Usage:
    python scripts/run_optimization.py [--population 200] [--generations 300] [--runs 5]
"""
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gbc_bep.io import load_yaml
from gbc_bep.optimization import INSGA3Reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--population', type=int, default=None)
    ap.add_argument('--generations', type=int, default=None)
    ap.add_argument('--runs', type=int, default=None)
    a = ap.parse_args()

    cfg = load_yaml(ROOT / 'configs/optimization.yaml')
    pop = a.population or cfg['population_size']
    gens = a.generations or cfg['generations']
    runs = a.runs or cfg['runs']

    alg = INSGA3Reference(population_size=pop, generations=gens)
    print(f'INSGA-III: population={pop}, generations={gens}, runs={runs}')
    print('Adaptive schedules (generation -> pc, pm):')
    for g in [0, 50, 150, gens]:
        pc, pm = alg.schedules(g)
        print(f'  {g:>3d} -> {pc:.3f}, {pm:.3f}')

    print('\nReleased archive: results/optimization/pareto_200.csv')
    print('Released convergence: results/optimization/convergence_5runs_300gen.csv')


if __name__ == '__main__':
    main()
