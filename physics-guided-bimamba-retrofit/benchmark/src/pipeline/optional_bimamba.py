"""Optional CUDA/PyTorch BiMamba backbone for the benchmark.

Requires genuine mamba_ssm.Mamba; there is no silently substituted GRU.
Upstream API: https://github.com/state-spaces/mamba
"""
import torch
from torch import nn
from mamba_ssm import Mamba

class BidirectionalLayer(nn.Module):
    def __init__(self,d_model,d_state,dropout):
        super().__init__()
        self.forward_block=Mamba(d_model=d_model,d_state=d_state,d_conv=4,expand=2)
        self.reverse_block=Mamba(d_model=d_model,d_state=d_state,d_conv=4,expand=2)
        self.norm=nn.LayerNorm(d_model);self.fuse=nn.Linear(2*d_model,d_model);self.drop=nn.Dropout(dropout)
    def forward(self,x):
        z=self.norm(x)
        h=torch.cat([self.forward_block(z),torch.flip(self.reverse_block(torch.flip(z,[1])),[1])],dim=-1)
        return x+self.drop(self.fuse(h))

class BiMamba(nn.Module):
    def __init__(self,d_model=64,d_state=16,layers=2,dropout=.2):
        super().__init__();self.input=nn.Linear(55,d_model)
        self.layers=nn.ModuleList([BidirectionalLayer(d_model,d_state,dropout) for _ in range(layers)])
        self.attention=nn.Linear(d_model,1)
        self.head=nn.Sequential(nn.Linear(d_model,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,1),nn.Softplus())
    def forward(self,x):
        h=self.input(x)
        for layer in self.layers:h=layer(h)
        a=torch.softmax(self.attention(h),dim=1)
        return self.head((h*a).sum(1)).squeeze(-1)

def physical_loss(pred,y,physics,scale,lambda_phys=.05,epsilon=.2):
    data=((pred-y)/scale).square().mean()
    physical=(torch.relu(torch.abs(pred-physics)-epsilon)/scale).square().mean()
    return data+lambda_phys*physical,data,physical
