#!/usr/bin/env python3
"""Rebuild the benchmark dataset, trained checkpoints and all reported results."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,argparse,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from pipeline.core import *

def save(df,p):
    path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True)
    # Fixed gzip mtime makes byte-level reruns comparable in the same numerical environment.
    compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else None
    df.to_csv(path,index=False,float_format='%.8g',compression=compression)
def js(obj,p):
    path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))
def profile(w):
    # Two days per season, all 15-min intervals; ordered but independent stitched episodes.
    ts=pd.to_datetime(w.timestamp)
    chunks=[w.loc[(ts>=date)&(ts<pd.Timestamp(date)+pd.Timedelta(days=2))] for date in ['2023-01-16','2023-04-17','2023-07-17','2023-10-16']]
    return pd.concat(chunks,ignore_index=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=16)
    ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--population',type=int,default=24)
    ap.add_argument('--generations',type=int,default=16);ap.add_argument('--scenarios',type=int,default=500)
    a=ap.parse_args()
    print('1/7 Generate 1R1C reference observations',flush=True)
    w=weather();daily=candidate_matrix(730);C=np.repeat(daily,96,axis=0)
    X,y,info=simulate(w,C)
    raw=w.copy()
    for j,c in enumerate(CANDIDATES):raw[c]=C[:,j]
    raw['indoor_start_c']=X[:,11];raw['reference_load_kwh']=y
    raw['reference_hvac_kwh']=info['reference_hvac_kwh'];raw['physics_estimate_kwh']=info['physics_estimate_kwh']
    raw['data_origin']='1R1C_reference';raw['seed']=SEED
    # Preserve the previous package's count-aligned split, documenting actual dates.
    split=np.repeat(['train','validation','test'],[35040,17472,17568]);raw['split']=split
    save(raw,'data/bms_15min.csv.gz')
    save(pd.DataFrame([dict(split=s,n=int((split==s).sum()),start=str(w.timestamp[split==s].iloc[0]),end=str(w.timestamp[split==s].iloc[-1])) for s in ['train','validation','test']]),'data/split_manifest.csv')
    train=split=='train';val=split=='validation';test=split=='test'
    encoder=Features().fit(X[train]);XX=encoder.transform(X);physics=info['physics_estimate_kwh']
    (ROOT/'models').mkdir(exist_ok=True);encoder.save(ROOT/'models/feature_encoder.npz')
    processed=pd.DataFrame(XX,columns=[f'feat_{j:02d}' for j in range(55)])
    processed.insert(0,'timestamp',w.timestamp);processed['target_kwh']=y;processed['physics_kwh']=physics;processed['split']=split
    save(processed,'data/features_55d.csv.gz')
    js(dict(operational=FEATURES,rc=['tau_h','UA_kw_K','solar_aperture_m2'],gbc_memberships=['nearest_rank_weight_'+str(i+1) for i in range(5)],
            origin='training-only fixed KMeans centres over a nine-feature subspace; five nearest-centre membership weights',d95=float(encoder.threshold)), 'data/feature_schema_28f.json')

    print('2/7 Train physical-loss MLP and matched no-physics ablation',flush=True)
    pg=PhysicsMLP();hist=pg.fit(XX[train],y[train],physics[train],epochs=a.epochs)
    hist['architecture']='NumPy physics-guided multilayer perceptron';save(hist,'results/training_history.csv');pg.save(ROOT/'models/physics_mlp.npz')
    ab=PhysicsMLP(lambda_phys=0);ab.fit(XX[train],y[train],physics[train],epochs=a.epochs);ab.save(ROOT/'models/without_physics_mlp.npz')
    ridge=LinearRegression().fit(XX[train],y[train])
    pred=pg.predict(XX[test]);pred0=ab.predict(XX[test]);predr=np.maximum(0,ridge.predict(XX[test]));persistence=np.r_[y[np.flatnonzero(test)[0]-1],y[test][:-1]]
    predictions=pd.DataFrame(dict(timestamp=w.timestamp[test],y_true=y[test],physics_MLP=pred,MLP_without_physics=pred0,linear_regression=predr,persistence=persistence,physics_estimate=physics[test]))
    save(predictions,'results/test_predictions.csv.gz')
    rows=[]
    for name,p in [('physics_MLP',pred),('MLP_without_physics',pred0),('linear_regression',predr),('persistence',persistence),('physics_estimate',physics[test])]:
        rows.append(dict(model=name,**metrics(y[test],p),physics_violation_mse=float(np.mean(np.maximum(abs(p-physics[test])-.2,0)**2))))
    save(pd.DataFrame(rows),'results/model_metrics.csv')
    js(dict(validation=metrics(y[val],pg.predict(XX[val])),test=metrics(y[test],pred),training_epochs=a.epochs,
        physics_loss_used=True,lambda_phys=.05,seed=SEED,scope='CPU physics-guided MLP benchmark'), 'results/training_summary.json')
    print('3/7 Paired circular block bootstrap and cross-domain transfer experiments',flush=True)
    # Pair with observed previous interval rather than resetting persistence at test boundary.
    bs=block_bootstrap(y[test],pred,b=a.bootstrap,baseline=persistence);js(bs,'results/block_bootstrap.json')
    transfers=[];fewshot=[]
    for label,shift,seed in [('warm_climate',5.,SEED+1),('cold_climate',-5.,SEED+2)]:
        we=weather('2024-01-01',periods=30*96,seed=seed,shift=shift)
        xe,ye,ine=simulate(we,np.repeat(candidate_matrix(30,seed),96,axis=0),seed=seed)
        ze=encoder.transform(xe);p=pg.predict(ze)
        transfers.append(dict(domain=label,n=len(ye),**metrics(ye,p),support_fraction=float(encoder.domain(xe).mean()),origin='held-out climate profile'))
        external=we.copy();external['y_true']=ye;external['source_model_prediction']=p;external['data_origin']='held_out_climate'
        save(external,f'data/external_{label}.csv.gz')
        # Fixed test days 22-30; adaptation uses only days 1 through the specified budget.
        for days in [2,7,14]:
            k=days*96
            ft=copy.deepcopy(pg);ft.fit(ze[:k],ye[:k],ine['physics_estimate_kwh'][:k],epochs=8,seed=seed,initialise_scale=False)
            scratch=PhysicsMLP(seed=seed);scratch.fit(ze[:k],ye[:k],ine['physics_estimate_kwh'][:k],epochs=8,seed=seed)
            for method,m in [('fine_tuned',ft),('scratch',scratch)]:
                fewshot.append(dict(domain=label,days=days,method=method,test_start_day=22,**metrics(ye[21*96:],m.predict(ze[21*96:]))))
    save(pd.DataFrame(transfers),'results/external_validation.csv');save(pd.DataFrame(fewshot),'results/few_shot_transfer.csv')

    print('4/7 Sixty candidate checks against the 1R1C reference model',flush=True)
    wp=profile(w);save(wp,'data/seasonal_profile.csv.gz')
    candidates=candidate_matrix(60,SEED+60);ref=evaluate_candidates(wp,candidates)
    # An approximate physics evaluator, separately measured rather than forcing agreement.
    approx=[]
    for c in candidates:
        _,_,ii=simulate(wp,c,measurement_noise=False)
        e=(ii['physics_estimate_kwh']+ii['lighting_kwh']).sum()*365*96/len(wp)/1000
        approx.append(e)
    tab=pd.DataFrame(candidates,columns=CANDIDATES);tab.insert(0,'case_id',np.arange(1,61))
    tab['reference_energy_MWh']=ref[:,0];tab['approximate_physics_energy_MWh']=approx
    tab['relative_error_pct']=100*(np.array(approx)-ref[:,0])/ref[:,0];tab['reference_model']='1R1C_reference'
    save(tab,'results/60_case_reference_comparison.csv')
    js(dict(n=60,energy=metrics(ref[:,0],np.array(approx)),simulator='1R1C ideal thermostat',
          annualization='eight seasonal days times 365/8'), 'results/60_case_summary.json')

    print('5/7 Run actual candidate optimization with reference-direction niching',flush=True)
    frontsC=[];frontsF=[];runs=[]
    for run in range(3):
        cc,ff,hh=optimize(wp,pop=a.population,generations=a.generations,seed=SEED+run)
        frontsC.append(cc);frontsF.append(ff);hh['run']=run+1;runs.append(hh)
    cc=np.concatenate(frontsC);ff=np.concatenate(frontsF);ii=fronts(ff)[0];cc=cc[ii];ff=ff[ii]
    # Eliminate exact duplicates produced by the small population evolution.
    _,unique=np.unique(np.round(cc,8),axis=0,return_index=True);cc=cc[unique];ff=ff[unique]
    pareto=pd.DataFrame(cc,columns=CANDIDATES)
    for j,name in enumerate(['energy_MWh_annualized','lifecycle_cost_USD','discomfort_degree_C','carbon_tCO2_annualized']):pareto[name]=ff[:,j]
    save(pareto,'results/pareto_front.csv');save(pd.concat(runs,ignore_index=True),'results/optimization_history.csv')
    norm=(ff-ff.min(0))/np.maximum(np.ptp(ff,axis=0),1e-8)
    chosen=[int(np.argmin(ff[:,0])),int(np.argmin(ff[:,1])),int(np.argmin(norm.sum(1)))]
    archetypes=[cc[i] for i in chosen];arch=pd.DataFrame(archetypes,columns=CANDIDATES);arch.insert(0,'archetype',['energy','cost','balanced'])
    save(arch,'results/selected_archetypes.csv')

    print('6/7 Propagate uncertainty and check counts, occupancy and dropout',flush=True)
    scenarios=[]
    for mode in ['poisson','empirical_day_block']:
        for j,c in enumerate(archetypes):
            rng=np.random.default_rng(SEED+100+j)
            for s in range(a.scenarios):
                wu=wp.copy();wu['outdoor_c']+=rng.normal(0,1.5)
                if mode=='poisson':wu['occupancy_fraction']=np.clip(rng.poisson(np.maximum(.01,wp.occupancy_fraction.to_numpy()*100))/100,0,1)
                else:
                    # Resample eight whole occupancy days, retaining within-day dependence.
                    occ=wp.occupancy_fraction.to_numpy().reshape(-1,96)
                    wu['occupancy_fraction']=occ[rng.integers(0,len(occ),len(occ))].reshape(-1)
                cu=c.copy();cu[4:6]*=np.clip(rng.normal(1,.04,2),.7,1.3)
                f=evaluate_candidates(wu,[cu])[0]
                scenarios.append(dict(occupancy_model=mode,archetype=['energy','cost','balanced'][j],scenario=s+1,energy_MWh=f[0],carbon_tCO2=f[3]))
    scenarios=pd.DataFrame(scenarios);save(scenarios,'results/uncertainty_scenarios.csv.gz')
    convergence=[]
    for (mode,arch),g in scenarios.groupby(['occupancy_model','archetype']):
        for n in [100,200,300,500]:
            if n>a.scenarios:continue
            z=g.iloc[:n].energy_MWh.to_numpy();lo,hi=np.quantile(z,[.05,.95])
            convergence.append(dict(occupancy_model=mode,archetype=arch,n=n,mean_MWh=z.mean(),q05_MWh=lo,q95_MWh=hi,width_MWh=hi-lo))
    save(pd.DataFrame(convergence),'results/scenario_convergence.csv')
    dropout=[];rng=np.random.default_rng(SEED+900);subset=XX[test][::96]
    for rate in [.1,.2,.3]:
        samples=np.array([pg.predict(subset,dropout=rate,rng=rng).mean() for _ in range(200)])
        for passes in [20,50,100,200]:
            zz=samples[:passes];lo,hi=np.quantile(zz,[.05,.95])
            dropout.append(dict(dropout_probability=rate,passes=passes,mean_interval_energy_kwh=zz.mean(),q05=lo,q95=hi,width=hi-lo,
                                scope='post-hoc hidden-unit perturbation'))
    save(pd.DataFrame(dropout),'results/dropout_sensitivity.csv')

    print('7/7 Provenance and completion checks',flush=True)
    js(dict(package_version='1.1.0',data_origin='1R1C reference simulator',seed=SEED,
        generator='scripts/reproduce.py + src/pipeline/core.py',model='1R1C ideal thermostat',
        train_candidates='different randomly sampled candidate configuration per day',
        observations='noisy reference HVAC energy',physics_loss='optimized, lambda .05, tolerance .2 kWh',
        configuration=vars(a),numpy=np.__version__,pandas=pd.__version__), 'PROVENANCE.json')
    allfiles=sorted([p for base in ['data','results'] for p in (ROOT/base).rglob('*') if p.is_file()])
    lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)) for p in allfiles]
    (ROOT/'RESULTS_SHA256.txt').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('Completed benchmark workflow; model checkpoint and computed results saved.',flush=True)

if __name__=='__main__':main()
