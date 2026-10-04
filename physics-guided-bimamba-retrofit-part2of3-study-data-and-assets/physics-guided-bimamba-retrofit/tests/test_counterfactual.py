import numpy as np
from gbc_bep.counterfactual import RetrofitCandidate, vfd_power_fraction, insulation_resistance

def test_candidate_helpers():
    c=RetrofitCandidate(0.10,1.6,0.32,0.28,4.1,0.9,26,19,6.5,6,1,1)
    assert abs(insulation_resistance(c)-3.3333333333)<1e-6
    assert np.isfinite(vfd_power_fraction(np.array([0.5]))).all()
