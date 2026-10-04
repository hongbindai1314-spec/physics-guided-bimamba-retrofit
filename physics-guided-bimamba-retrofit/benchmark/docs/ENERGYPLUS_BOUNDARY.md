# Whole-building simulation boundary

The executable pipeline in this package uses a reduced-order 1R1C ideal-thermostat
reference whose equations and coefficients are stated in `docs/METHODS.md`. It does not
invoke a full whole-building simulation engine, and its outputs are not labelled as such.

To reproduce a full-engine experiment, supply the complete geometry, construction and
material definitions, HVAC objects and schedules, weather source and EPW, the case-parameter
adapter, the engine version, timestep and run period, output variables/meters and unit
conversions, successful run logs, and pointwise outputs. Calibration metrics additionally
require the comparison series and the calibration protocol actually used.

A 269-byte IDF alone is not sufficient for that experiment. Any weather file that is not
an observed record must carry an explicit statement of its construction. The 60-case 1R1C
table in this package is a self-contained reference comparison and is kept separate from
any full-engine result.
