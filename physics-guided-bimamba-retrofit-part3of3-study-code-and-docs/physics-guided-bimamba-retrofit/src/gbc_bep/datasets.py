from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self,X,y,lookback=96,rc_implied=None):
        self.X=np.asarray(X,np.float32); self.y=np.asarray(y,np.float32); self.lookback=lookback
        self.rc=None if rc_implied is None else np.asarray(rc_implied,np.float32)
        if len(self.X)!=len(self.y): raise ValueError("X/y length mismatch")
    def __len__(self): return max(0,len(self.y)-self.lookback)
    def __getitem__(self,i):
        t=i+self.lookback-1
        x=torch.from_numpy(self.X[t-self.lookback+1:t+1])
        y=torch.tensor(self.y[t+1],dtype=torch.float32)
        if self.rc is None: return x,y
        return x,y,torch.tensor(self.rc[t+1],dtype=torch.float32)
