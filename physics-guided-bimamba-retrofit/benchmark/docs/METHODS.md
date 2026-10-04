# Benchmark methods and configuration

## Physical reference

The reference is an explicit Euler 1R1C ideal-thermostat model:

`T_free(t+1) = T_start(t) + dt * [UA*(T_out(t)-T_start(t)) + Q_solar(t) + Q_people(t) + Q_lights(t)] / C`.

Cooling/heating needed to reach the upper/lower setpoint is capped at 120 kW. The post-control temperature becomes the next interval's start state. Electrical HVAC energy is `dt*(Q_cool/COP_cool + Q_heat/COP_heat + fan_power)`. Lighting is a separate electrical term. Both heating and cooling use COP; the heating parameter is **not** a boiler efficiency. There is no latent-load model and no detailed multizone HVAC topology.

Constants: dt=0.25h, floor area=1200 m2, envelope reference area=600 m2, volume=3600 m3, capacitance=95 kWh/K. Wall U-value is `1/(0.5 + insulation_m/0.035)`. Ventilation conductance is `0.33*volume*ACH*(1-0.7*ERV)/1000` kW/K. Person gain is 18 kW at full occupancy. VFD reduces fan power by 35%.

The approximate physics energy estimate uses 93% of UA and 90% of reference internal/solar gains, so it is a deliberately imperfect physics teacher. It uses the reference simulator's start-of-interval indoor state, which is treated as an observed state in the one-step task. All weather, occupancy and candidate controls at the current origin are assumed known; this is not an autonomous multi-step forecast.

## Data and preprocessing

Weather has an annual sinusoid, daily sinusoid and seeded daily perturbations; occupancy has weekday schedules and noise. Training samples cover independently drawn configurations per day. No result is constructed to match a desired RMSE/MAE/MAPE. Current target energy is absent from predictor features: load inputs are lags of 1, 4 and 96 intervals. Indoor state is from the preceding state transition.

Training-only means/stds standardize 47 features. RC descriptors are C/UA, UA and solar aperture. Fixed KMeans centres are fitted on a nine-feature training subspace. Five weights refer to the nearest-centre **ranks**, not globally fixed centre identities. The training 95th-percentile nearest-centre distance defines the support threshold. That numeric threshold is computed, not forced to a preset value.

The generated tables have no missing data and no deliberately injected outliers.

## Executed neural model

The CPU model is a 55-input, 32-hidden-unit tanh MLP with lagged temporal features. It is not a sequence-state-space model. Adam updates all weights using:

`L = mean((pred-y)^2) + 0.05*mean(max(abs(pred-physics)-0.2,0)^2)`.

The common target standard deviation rescales both terms during training; this does not change their relative weight. Predictions are clipped at zero and the clipping derivative is used in training. The no-physics ablation uses the same initial seed and minibatch order with lambda=0. All checkpoints and predicted values derive from optimization. No performance ordering is imposed, including whether the physical term improves prediction accuracy.

The default CPU epochs are fixed at 16; validation is reported but is not used for model selection. The optional BiMamba route uses validation early stopping. Encoder objects remain training-fitted for external domains. Fine-tuning budgets use only the first 2, 7 or 14 days, and every adaptation run tests the same days 22-30. External climates are 30-day January profiles shifted by +5 or -5 C, used as held-out domain-transfer cases rather than as annual climate validation.

## Metrics and bootstrap

RMSE/MAE are in interval kWh. Reported `MAPE_pct` is stabilized MAPE with denominator `max(abs(y),1 kWh)` to avoid small-load divergence; it must not be compared directly with a MAPE using a different denominator. R2 uses the observed test mean.

Circular moving blocks preserve 672 consecutive 15-minute samples, with a final truncated block to restore exact sample size. One set of bootstrap indices is shared by compared errors. Percentile 95% RMSE/MAE intervals are computed from block sums. The paired test compares squared-error improvement over persistence and uses a centered null distribution, with the +1 Monte Carlo correction.

## Candidate comparison and optimization

An eight-day profile comprises two days from January, April, July and October. Episodes are concatenated and simulated with the same initial 21 C and an ongoing state; this is an approximate stitched seasonal profile, not an exact annual weather trajectory. Annualized energy multiplies the sampled sum by 365/8. The 60-case table measures approximate teacher energy against the 1R1C reference.

Four minimized objectives are annualized HVAC+lighting MWh, a 15-year undiscounted illustrative cost, mean temperature degree violation outside 19-26 C, and annualized carbon (`energy_MWh*0.55 + capital_USD*0.00002`). Pricing and emission coefficients are illustrative scenario parameters. Optimization evaluates the physical reference directly.

The optimizer evaluates offspring with mixed continuous/binary parameters, adaptive crossover/mutation schedules, non-dominated sorting and reference-direction niching. The normalization uses min-max scaling. Variation uses blend crossover and Gaussian mutation. Default results use 3 runs, 24 individuals and 16 generations; population size and generation count can be increased with CLI flags.

## Uncertainty and sensitivity

For each selected energy/cost/balanced solution, scenarios perturb outdoor temperature, occupancy and both COP values, then execute the physical simulator again. Occupancy uses either Poisson counts or whole-day resampling. 500 scenarios per solution/model are computed; 100/200/300/500 prefixes support convergence inspection. Scenario values and any ranking differences are retained without forcing convergence or stability.

The MLP dropout table perturbs hidden units after training, over one observation per test day. Rates 0.1/0.2/0.3 and 20/50/100/200 passes are actually sampled. This is a model-perturbation stress test and does not by itself establish prediction-interval coverage.
