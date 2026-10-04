# Data dictionary and schema

## Raw BMS table

Expected frequency: 15 minutes. Expected timestamp column: `timestamp`.

The study uses 47 operational variables drawn from thermal, HVAC, energy, weather and occupancy measurements. The variable list is held in `configs/data.yaml`. Typical fields are:

| Concept | Suggested column | Unit |
|---|---|---|
| zone temperature mean | `zone_temp_c` | °C |
| zone relative humidity | `zone_rh_pct` | % |
| zone CO2 mean | `zone_co2_ppm` | ppm |
| outdoor dry-bulb | `outdoor_temp_c` | °C |
| outdoor RH | `outdoor_rh_pct` | % |
| GHI | `ghi_w_m2` | W/m² |
| wind speed | `wind_speed_m_s` | m/s |
| Wi-Fi occupancy | `wifi_count` | persons |
| chiller power | `chiller_power_kw` | kW |
| HVAC target | `hvac_energy_kwh` | kWh/15 min |
| lighting electricity | `lighting_energy_kwh` | kWh/15 min |
| plug electricity | `plug_energy_kwh` | kWh/15 min |

Floor/zone-level fields are retained as separate columns to reach the 47-variable operational matrix used in the study.

## Processed feature tensor

Model-input dimensionality used throughout the study:

- 47 operational variables;
- 3 RC descriptors: `tau_h`, `kappa_w_k`, `gamma_m2`;
- 5 GBC nearest-ball membership weights: `gbc_w1` ... `gbc_w5`;
- total: 55 features.

## 60-case validation CSV

- `x1`: insulation thickness (m)
- `x2`: window U-value (W/m²K)
- `x3`: SHGC
- `x4`: WWR
- `x5`: cooling COP
- `x6`: heating efficiency
- `x7`: cooling setpoint (°C)
- `x8`: heating setpoint (°C)
- `x9`: ventilation rate (L/s/person in the validation table)
- `x10`: lighting power density (W/m²)
- `x11`: VFD installed (0/1)
- `x12`: ERV installed (0/1)
- `BiMamba_Energy`, `EnergyPlus_Energy`: annual EUI (kWh/m²·yr)
- `BiMamba_PPD`, `EnergyPlus_PPD`: work-time PPD (%)
- `BiMamba_Carbon`, `EnergyPlus_Carbon`: annual carbon intensity (kg CO2e/m²·yr)
- `A_x`: annual GBC applicability-domain coverage (%)
