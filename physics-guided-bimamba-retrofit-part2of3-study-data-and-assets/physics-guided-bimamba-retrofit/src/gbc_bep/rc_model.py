from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RCParameters:
    R_zw: float
    R_wm: float
    C_z: float
    C_w: float
    C_m: float
    alpha_sol: float


def descriptors(params: RCParameters, window_area_m2: float) -> dict[str,float]:
    """Compute the three manuscript-defined RC descriptors.

    tau is converted from seconds to hours when R[C] are supplied in K/W and J/K.
    """
    tau_h=params.R_zw*params.C_z/3600.0
    kappa=1.0/(params.R_zw+params.R_wm)
    gamma=params.alpha_sol*window_area_m2
    return {"tau_h":tau_h,"kappa_w_k":kappa,"gamma_m2":gamma}


def note_on_estimation() -> str:
    return (
        "The three-node RC state-space form used in the study is instantiated with the calibrated "
        "per-floor parameters in results/rc/rc_parameters_15_floors.csv. The continuous A(theta), "
        "B(theta) matrices are assembled from R_zw, R_wm, C_z, C_w and C_m, and the parameter "
        "identification uses the ADMM scheme described in the manuscript."
    )
