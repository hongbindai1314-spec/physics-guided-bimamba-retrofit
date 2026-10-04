from __future__ import annotations
import torch
from torch import nn

class TorchFallbackBlock(nn.Module):
    """Dependency-free substitute used only when ``mamba-ssm`` is unavailable; not numerically equivalent to Mamba."""
    def __init__(self,d_model):
        super().__init__(); self.gru=nn.GRU(d_model,d_model,batch_first=True)
    def forward(self,x): return self.gru(x)[0]

class SelectiveBlock(nn.Module):
    def __init__(self,d_model,backend="mamba_ssm"):
        super().__init__(); self.backend=backend
        if backend=="mamba_ssm":
            try:
                from mamba_ssm import Mamba
            except Exception as e:
                raise ImportError("Install a compatible `mamba-ssm` build, or set backend='torch_fallback' for unit tests on machines without it") from e
            self.block=Mamba(d_model=d_model,d_state=16,d_conv=4,expand=2)
        elif backend=="torch_fallback": self.block=TorchFallbackBlock(d_model)
        else: raise ValueError("backend must be mamba_ssm or torch_fallback")
    def forward(self,x): return self.block(x)

class BiMambaLayer(nn.Module):
    def __init__(self,d_model,backend="mamba_ssm",dropout=0.3):
        super().__init__(); self.fwd=SelectiveBlock(d_model,backend); self.bwd=SelectiveBlock(d_model,backend)
        self.a=nn.Linear(2*d_model,d_model); self.g=nn.Linear(2*d_model,d_model); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        f=self.fwd(x); b=torch.flip(self.bwd(torch.flip(x,dims=[1])),dims=[1])
        h=torch.cat([f,b],dim=-1)
        return self.drop(self.a(h)*torch.sigmoid(self.g(h)))

class PhysicsGuidedBiMamba(nn.Module):
    def __init__(self,input_dim=55,state_dim=128,layers=4,cnn_channels=64,cnn_kernels=(3,5,7),fused_dim=128,attention_dim=64,mlp_hidden_dim=128,dropout=0.3,backend="mamba_ssm"):
        super().__init__(); self.input_dim=input_dim
        self.branches=nn.ModuleList([nn.Conv1d(input_dim,cnn_channels,k,padding=k//2) for k in cnn_kernels])
        self.fuse=nn.Linear(cnn_channels*len(cnn_kernels),fused_dim)
        self.layers=nn.ModuleList([BiMambaLayer(fused_dim,backend,dropout) for _ in range(layers)])
        self.attn=nn.Sequential(nn.Linear(fused_dim,attention_dim),nn.Tanh(),nn.Linear(attention_dim,1))
        self.head=nn.Sequential(nn.Linear(fused_dim,mlp_hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(mlp_hidden_dim,1),nn.Softplus())
    def forward(self,x):
        # x [B,T,F]
        z=x.transpose(1,2)
        conv=torch.cat([torch.relu(c(z)) for c in self.branches],dim=1).transpose(1,2)
        h=self.fuse(conv)
        for layer in self.layers: h=layer(h)
        a=torch.softmax(self.attn(h).squeeze(-1),dim=1)
        ctx=torch.sum(h*a.unsqueeze(-1),dim=1)
        return self.head(ctx).squeeze(-1)
