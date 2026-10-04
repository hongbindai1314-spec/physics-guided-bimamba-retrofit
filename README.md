# Physics-guided BiMamba building-retrofit reproducibility package

This package contains the complete repository for the physics-guided BiMamba retrofit study
together with an independent, fully executable benchmark pipeline under `benchmark/`.

Original source, configurations, data, results, EnergyPlus assets, manuscript and figures are
retained under their respective directories.

## Benchmark pipeline

The `benchmark/` subtree is a self-contained, executable data-to-results pipeline covering
reference-model data, feature construction, training, evaluation, multi-objective optimization
and uncertainty propagation. Its equations, coefficients, seeds and outputs are all included.

```bash
python -m pip install -r benchmark/requirements.txt
python scripts/run_simulation.py
python scripts/run_simulation.py --mode tests
python scripts/run_simulation.py --mode verify
```

All outputs are written inside `benchmark/`. They include the raw 15-minute table and split
manifest, the 55-dimensional feature matrix, actual CPU MLP checkpoints, the 60-case 1R1C
comparison, the evolved Pareto set, and the scenario-propagation and sensitivity tables.
See [benchmark/README.md](benchmark/README.md) and
[benchmark/docs/METHODS.md](benchmark/docs/METHODS.md) for equations, assumptions, parameter
choices and tested scope.

The optional BiMamba entry point is
`python scripts/run_simulation.py --mode bimamba --device cuda --epochs 50`. It requires an
external PyTorch/CUDA `mamba-ssm` installation; the delivered CPU checkpoints are NumPy
physics-guided MLP weights.

## Study repository

All study scripts are under `scripts/` and the study source package is under `src/gbc_bep/`.
The 60-case CSV supports recalculating the stored table statistics. Root-level data, results,
EnergyPlus assets and manuscript files are described in `REPRODUCIBILITY.md` and
`docs/DATA_DICTIONARY.md`.

Package version `1.1.0` identifies this deliverable.
