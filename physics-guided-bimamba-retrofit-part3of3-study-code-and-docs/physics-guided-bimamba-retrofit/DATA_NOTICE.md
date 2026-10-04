# Data provenance

The `data/`, `results/`, `energyplus/` and `paper/` contents are part of the study repository.
Their provenance is documented in `REPRODUCIBILITY.md`, `docs/DATA_DICTIONARY.md` and the
per-directory README files.

The benchmark pipeline under `benchmark/` is fully self-contained. Its thermal reference is a
reduced-order 1R1C ideal-thermostat model with all equations, coefficients and seeds stated in
`benchmark/docs/METHODS.md` and `benchmark/PROVENANCE.json`. Every table under
`benchmark/results/` is computed by `benchmark/scripts/reproduce.py` and can be regenerated
byte-for-byte in the same numerical environment; the corresponding hashes are recorded in
`benchmark/RESULTS_SHA256.txt`.

Datasets are stored as delivered. Use `benchmark/scripts/verify_results.py` to recheck
checkpoint reproduction and result hashes.
