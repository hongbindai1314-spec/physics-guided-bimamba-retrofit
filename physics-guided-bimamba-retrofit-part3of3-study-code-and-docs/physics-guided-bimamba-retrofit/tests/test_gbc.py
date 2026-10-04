import numpy as np
from gbc_bep.gbc import GranularBallRegressor

def test_gbc_membership_shape_and_sum():
    rng=np.random.default_rng(0); X=np.r_[rng.normal(-2,0.3,(60,3)),rng.normal(2,0.3,(60,3))]; y=np.r_[rng.normal(0,0.1,60),rng.normal(5,0.1,60)]
    g=GranularBallRegressor(purity_threshold=0.7,min_ball_size=10,nearest_balls=5).fit(X,y)
    W=g.membership(X[:8]); assert W.shape[0]==8; assert np.allclose(W.sum(1),1)
