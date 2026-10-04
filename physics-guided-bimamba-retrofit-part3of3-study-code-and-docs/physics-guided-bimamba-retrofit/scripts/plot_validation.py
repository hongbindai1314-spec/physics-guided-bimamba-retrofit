#!/usr/bin/env python
from pathlib import Path
import sys, pandas as pd, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/validation/energyplus_60_validation.csv')
out=ROOT/'results/validation'; out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(df.EnergyPlus_Energy,df.BiMamba_Energy)
lo=min(df.EnergyPlus_Energy.min(),df.BiMamba_Energy.min()); hi=max(df.EnergyPlus_Energy.max(),df.BiMamba_Energy.max()); ax.plot([lo,hi],[lo,hi],'--')
ax.set_xlabel('EnergyPlus annual EUI (kWh/m²·yr)'); ax.set_ylabel('BiMamba annual EUI (kWh/m²·yr)'); ax.set_title('60-case surrogate–EnergyPlus cross-verification')
fig.tight_layout(); fig.savefig(out/'energyplus_energy_scatter.png',dpi=200); print(out/'energyplus_energy_scatter.png')
