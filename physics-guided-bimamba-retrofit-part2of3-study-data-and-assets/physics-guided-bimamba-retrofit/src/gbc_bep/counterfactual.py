from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RetrofitCandidate:
    insulation_m: float; window_u: float; shgc: float; wwr: float; cooling_cop: float
    heating_efficiency: float; cooling_setpoint_c: float; heating_setpoint_c: float
    ventilation: float; lpd_w_m2: float; vfd: int; erv: int


def vfd_power_fraction(plr):
    plr=np.asarray(plr,float)
    return 0.0013+0.1470*plr+0.9506*plr**2-0.0998*plr**3


def insulation_resistance(candidate: RetrofitCandidate, lambda_ins=0.030):
    return candidate.insulation_m/lambda_ins


def apply_counterfactual(base: dict[str,np.ndarray], c: RetrofitCandidate, baseline: RetrofitCandidate | None=None):
    """Apply explicit, auditable candidate transformations to a dictionary of annual sequences.

    This adapter intentionally works on named physical quantities rather than silently mutating an
    opaque 55-D tensor. The caller must then rebuild the 47 operational features, RC descriptors and
    5 GBC memberships using the same feature pipeline as training.
    """
    out={k:np.asarray(v).copy() for k,v in base.items()}
    # Control boundary conditions
    if "cooling_setpoint_c" in out: out["cooling_setpoint_c"][:]=c.cooling_setpoint_c
    if "heating_setpoint_c" in out: out["heating_setpoint_c"][:]=c.heating_setpoint_c
    if "ventilation_rate" in out: out["ventilation_rate"][:]=c.ventilation
    if "lpd_w_m2" in out: out["lpd_w_m2"][:]=c.lpd_w_m2
    if "lighting_energy_kwh" in out and baseline is not None:
        out["lighting_energy_kwh"]*=c.lpd_w_m2/max(baseline.lpd_w_m2,1e-12)
    # Thermal-to-electric conversion hooks if thermal demand is available
    if "cooling_thermal_kwh" in out: out["cooling_electric_kwh"]=out["cooling_thermal_kwh"]/c.cooling_cop
    if "heating_thermal_kwh" in out: out["heating_energy_kwh"]=out["heating_thermal_kwh"]/max(c.heating_efficiency,1e-12)
    # VFD auxiliary electricity
    if c.vfd and "fan_pump_plr" in out and "fan_pump_rated_kw" in out:
        out["fan_pump_kw"]=out["fan_pump_rated_kw"]*vfd_power_fraction(out["fan_pump_plr"])
    # ERV sensible/latent load recovery
    if c.erv:
        if "vent_sensible_load" in out: out["vent_sensible_load"]*=1-0.72
        if "vent_latent_load" in out: out["vent_latent_load"]*=1-0.65
    out["R_ins_m2K_W"]=np.array([insulation_resistance(c)])
    out["window_u_W_m2K"]=np.array([c.window_u])
    out["shgc"]=np.array([c.shgc]); out["wwr"]=np.array([c.wwr])
    return out
