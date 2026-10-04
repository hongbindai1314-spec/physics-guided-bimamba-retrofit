from __future__ import annotations
import torch
import torch.nn.functional as F


def physics_guided_loss(pred_kwh, target_kwh, rc_implied_kwh, lambda_phys=0.05, epsilon_kwh=15.0, residual_scale=None):
    data=F.mse_loss(pred_kwh,target_kwh)
    mismatch=torch.relu(torch.abs(pred_kwh-rc_implied_kwh)-epsilon_kwh)
    if residual_scale is not None:
        mismatch=mismatch/(residual_scale+1e-8)
    phys=torch.mean(mismatch*mismatch)
    return data+lambda_phys*phys, {"data_loss":data.detach(),"physics_loss":phys.detach()}
