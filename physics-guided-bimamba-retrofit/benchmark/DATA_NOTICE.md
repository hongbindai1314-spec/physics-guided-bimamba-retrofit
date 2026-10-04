# Data origin

All records, features and result tables in this package are produced by the executable
benchmark pipeline shipped alongside them. The thermal reference is a reduced-order 1R1C
ideal-thermostat model; the full set of equations, coefficients, candidate bounds and
execution code are public in this package.

The raw table records `data_origin=1R1C_reference` together with the generator seed, so
every row can be traced back to the stated model and configuration.

## Interpretation

Fixed seeds support workflow reproducibility: rerunning the pipeline in the same numerical
environment reproduces the published tables. They do not by themselves establish agreement
with any external measured building, a whole-building simulation engine, or any published
benchmark. Where such an external comparison is required, the corresponding inputs and run
logs must be supplied as described in `docs/ENERGYPLUS_BOUNDARY.md`.
