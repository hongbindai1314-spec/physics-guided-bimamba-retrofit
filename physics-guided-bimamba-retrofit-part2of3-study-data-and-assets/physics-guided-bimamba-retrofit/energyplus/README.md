# EnergyPlus case-study assets

This directory holds the EnergyPlus inputs used for the 60-case cross-verification and the retrofit scenario generation.

- `model/baseline.idf`: baseline model definition for the case-study building.
- `weather/Shanghai_2023.epw`: weather file used for the annual simulations.
- `schedules/annual_schedules_15min.csv.gz`: occupancy, lighting, plug and setpoint schedules at 15-minute resolution.
- `scenarios/energyplus_60_case_inputs_and_outputs.csv`: the 60-case validation matrix.
- `scenarios/json/`: one machine-readable file per validation case, listing the 12 retrofit variables and the resulting simulator outputs.

The pointwise simulator outputs are also archived in `data/validation/energyplus_60_validation.csv`, which is the file consumed by `scripts/run_validation_analysis.py`.
