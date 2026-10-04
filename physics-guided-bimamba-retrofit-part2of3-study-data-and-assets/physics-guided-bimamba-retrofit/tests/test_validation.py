from pathlib import Path
import pandas as pd
from gbc_bep.validation import stratified_summary, ad_summary, global_statistics
ROOT=Path(__file__).resolve().parents[1]

def test_validation_closes():
    df=pd.read_csv(ROOT/'data/validation/energyplus_60_validation.csv')
    s=stratified_summary(df); o=s[s.Stratum=='Overall'].iloc[0]
    assert len(df)==60
    assert abs(o.Energy_MAPE_pct-2.3140916076)<1e-6
    assert abs(o.PPD_MAE_pp-0.43)<1e-12
    assert abs(o.Carbon_MAPE_pct-2.3187395904)<1e-6
    assert int(o.In_domain)==57
    a=ad_summary(df)
    assert int(a.loc[a.AD_status=='Out-of-domain','n'].iloc[0])==3
    g=global_statistics(df)
    assert g['AD']=={'in_domain':57,'out_of_domain':3}
