from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from .metrics import mape, mae, rmse, r2


def add_pointwise_errors(df):
    d=df.copy()
    d["Energy_APE_pct"]=(d.BiMamba_Energy-d.EnergyPlus_Energy).abs()/d.EnergyPlus_Energy.abs()*100
    d["PPD_AE_pp"]=(d.BiMamba_PPD-d.EnergyPlus_PPD).abs()
    d["Carbon_APE_pct"]=(d.BiMamba_Carbon-d.EnergyPlus_Carbon).abs()/d.EnergyPlus_Carbon.abs()*100
    d["AD_status"]=np.where(d.A_x>=95,"In-domain","Out-of-domain")
    return d


def stratified_summary(df):
    d=add_pointwise_errors(df)
    g=d.groupby("Stratum",sort=False).agg(
        n=("ID","count"), Energy_MAPE_pct=("Energy_APE_pct","mean"), PPD_MAE_pp=("PPD_AE_pp","mean"),
        Carbon_MAPE_pct=("Carbon_APE_pct","mean"), In_domain=("AD_status",lambda x:int((x=="In-domain").sum()))
    ).reset_index()
    overall=pd.DataFrame([{
        "Stratum":"Overall","n":len(d),"Energy_MAPE_pct":d.Energy_APE_pct.mean(),"PPD_MAE_pp":d.PPD_AE_pp.mean(),
        "Carbon_MAPE_pct":d.Carbon_APE_pct.mean(),"In_domain":int((d.AD_status=="In-domain").sum())
    }])
    return pd.concat([g,overall],ignore_index=True)


def ad_summary(df):
    d=add_pointwise_errors(df)
    g=d.groupby("AD_status",sort=False).agg(n=("ID","count"),Energy_MAPE_pct=("Energy_APE_pct","mean"),PPD_MAE_pp=("PPD_AE_pp","mean"),Carbon_MAPE_pct=("Carbon_APE_pct","mean")).reset_index()
    overall=pd.DataFrame([{"AD_status":"All designs","n":len(d),"Energy_MAPE_pct":d.Energy_APE_pct.mean(),"PPD_MAE_pp":d.PPD_AE_pp.mean(),"Carbon_MAPE_pct":d.Carbon_APE_pct.mean()}])
    return pd.concat([g,overall],ignore_index=True)


def global_statistics(df):
    return {
        "energy": {"MAPE_pct":mape(df.EnergyPlus_Energy,df.BiMamba_Energy),"RMSE":rmse(df.EnergyPlus_Energy,df.BiMamba_Energy),"R2":r2(df.EnergyPlus_Energy,df.BiMamba_Energy)},
        "ppd": {"MAE_pp":mae(df.EnergyPlus_PPD,df.BiMamba_PPD),"RMSE_pp":rmse(df.EnergyPlus_PPD,df.BiMamba_PPD),"R2":r2(df.EnergyPlus_PPD,df.BiMamba_PPD)},
        "carbon": {"MAPE_pct":mape(df.EnergyPlus_Carbon,df.BiMamba_Carbon),"RMSE":rmse(df.EnergyPlus_Carbon,df.BiMamba_Carbon),"R2":r2(df.EnergyPlus_Carbon,df.BiMamba_Carbon)},
        "rank_energy": {"Spearman_rho":float(spearmanr(df.EnergyPlus_Energy,df.BiMamba_Energy).statistic),"Kendall_tau":float(kendalltau(df.EnergyPlus_Energy,df.BiMamba_Energy).statistic)},
        "AD": {"in_domain":int((df.A_x>=95).sum()),"out_of_domain":int((df.A_x<95).sum())}
    }
