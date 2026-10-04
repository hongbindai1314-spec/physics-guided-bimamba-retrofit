# Reproducibility routes

## Benchmark pipeline

```bash
python -m pip install -r benchmark/requirements.txt
python scripts/run_simulation.py
python scripts/run_simulation.py --mode tests
python scripts/run_simulation.py --mode verify
```

This route regenerates the 15-minute reference table, the training-only feature encoder, the
NumPy physics-guided MLP checkpoints, the baseline/ablation predictions, the paired
block-bootstrap summary, the cross-domain transfer experiments, the 60-case 1R1C comparison, the
evolved Pareto set and the uncertainty/sensitivity outputs. It was run from empty output
directories; the five functional tests and the checkpoint-reload/result-hash checks passed. See
`benchmark/VERIFICATION.json`.

## Study repository

All study implementation and result files are available under `scripts/`, `src/gbc_bep/`,
`data/` and `results/`. `scripts/run_validation_analysis.py` recomputes the numerical
statistics of the 60-case CSV. `scripts/rebuild_dataset.py` regenerates the derived data and
result artefacts from the de-identified 15-minute facility table.

## Environment and versioning

```bash
python benchmark/scripts/verify_results.py
```

This reloads the trained checkpoint, recomputes test predictions, checks parameter bounds and
non-dominance, and verifies every hash listed in `benchmark/RESULTS_SHA256.txt`.

Numerical results depend on the BLAS/LAPACK implementation and library versions; byte-level
identity across platforms is not implied. The exact versions used for the recorded run are
listed in `benchmark/requirements.txt`.
