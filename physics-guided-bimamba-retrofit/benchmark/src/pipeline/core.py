"""Executable building-retrofit benchmark pipeline. Units: kW, kWh, degrees C, hours, USD.

The reference model is a reduced-order 1R1C ideal-thermostat formulation of the
zoned thermal balance. Outputs are computed from the stated equations and
coefficients; no result is fitted to any manuscript performance target.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

CANDIDATES = ['insulation_m','window_u','shgc','wwr','cooling_cop',
              'heating_cop','cooling_setpoint_c','heating_setpoint_c',
              'ventilation_ach','lpd_w_m2','vfd','erv']
LOW = np.array([.02,1.0,.2,.2,2.5,2.0,23.,18.,.2,4.,0.,0.])
HIGH = np.array([.18,3.5,.75,.55,6.,5.,27.,22.,1.0,12.,1.,1.])
BASE = np.array([.06,2.6,.5,.35,3.5,3.,25.,20.,.6,9.,0.,0.])
SEED = 20261004
AREA = 1200.
DT = .25
FEATURES = [
 'outdoor_c','rh_pct','ghi_w_m2','wind_m_s','occupancy_fraction',
 'hour_sin','hour_cos','day_sin','day_cos','weekday','occupied_schedule',
 'indoor_start_c','hvac_lag1_kwh','hvac_lag4_kwh','hvac_lag96_kwh',
 *CANDIDATES,
 'occ_sq','outdoor_occ','solar_occ',
 'delta_cool','delta_heat','positive_cooling','positive_heating',
 'wall_u','window_area_m2','ua_kw_k','capacity_kwh_k',
 'solar_gain_kw','people_gain_kw','light_gain_kw','vent_ua_kw_k',
 'lighting_kwh','outdoor_rh','solar_shgc','occupied_lpd','time_since_midnight_h']
assert len(FEATURES) == 47

def weather(start='2022-01-01', periods=70080, seed=SEED, shift=0.):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=periods, freq='15min')
    h = t.hour.to_numpy()+t.minute.to_numpy()/60
    d = t.dayofyear.to_numpy()
    day_noise = np.repeat(rng.normal(0,2.,int(np.ceil(periods/96))),96)[:periods]
    tout = 18+shift+11*np.sin(2*np.pi*(d-110)/365)+3*np.sin(2*np.pi*(h-10)/24)+day_noise
    solar = np.maximum(0,np.sin(np.pi*(h-6)/12))*700*(.7+.3*rng.random(periods))
    schedule = ((t.dayofweek.to_numpy()<5)&(h>=8)&(h<18)).astype(float)
    occ = np.clip(.08+.8*schedule+rng.normal(0,.025,periods),0,1)
    return pd.DataFrame(dict(timestamp=t,outdoor_c=tout,rh_pct=np.clip(65-.7*(tout-18)+rng.normal(0,5,periods),15,95),
        ghi_w_m2=solar,wind_m_s=np.maximum(.1,rng.normal(2,1,periods)),occupancy_fraction=occ))

def candidate_matrix(n, seed=SEED):
    rng=np.random.default_rng(seed)
    c=LOW+(HIGH-LOW)*rng.random((n,12));c[:,10:]=rng.integers(0,2,(n,2))
    return c

def properties(c):
    c=np.asarray(c);ins,u,shgc,wwr,cop,hcop,csp,hsp,vent,lpd,vfd,erv=c.T
    wall_u=1/(.5+ins/.035)
    window_area=600*wwr
    ventua=.33*3600*vent*(1-.7*erv)/1000
    ua=(wall_u*(600-window_area)+u*window_area)/1000+ventua
    capacity=np.full_like(ua,95.)
    return wall_u,window_area,ventua,ua,capacity

def simulate(w, c=BASE, seed=SEED, measurement_noise=True):
    """Euler 1R1C balance with ideal thermostat and a 120 kW HVAC limit.
    Weather/occupancy are declared exogenous scenario inputs at each forecast origin.
    Candidate parameters can vary daily in this multi-configuration benchmark.
    """
    n=len(w); c=np.broadcast_to(c,(n,12)).copy();tout=w.outdoor_c.to_numpy()
    wall,win,ventua,ua,cap=properties(c)
    occ=w.occupancy_fraction.to_numpy();solar=w.ghi_w_m2.to_numpy()*win*c[:,2]/1000
    people=occ*18;lights=c[:,9]*AREA/1000*(.15+.85*occ)
    gains=solar+people+lights
    indoor=np.empty(n);end=np.empty(n);cool=np.zeros(n);heat=np.zeros(n);comfort=np.zeros(n)
    prev=21.
    for i in range(n):
        indoor[i]=prev
        free=prev+DT*(ua[i]*(tout[i]-prev)+gains[i])/cap[i]
        cool[i]=min(120.,max(0.,(free-c[i,6])*cap[i]/DT))
        heat[i]=min(120.,max(0.,(c[i,7]-free)*cap[i]/DT))
        prev=free+DT*(heat[i]-cool[i])/cap[i];end[i]=prev
        comfort[i]=max(0.,prev-26.)+max(0.,19.-prev)
    fan=(3+5*occ)*np.where(c[:,10]>.5,.65,1.)
    hvac=DT*(cool/c[:,4]+heat/c[:,5]+fan)
    y=hvac.copy()
    if measurement_noise:
        y=np.maximum(0,y+np.random.default_rng(seed).normal(0,.12,n))
    h=pd.to_datetime(w.timestamp).dt.hour.to_numpy()+pd.to_datetime(w.timestamp).dt.minute.to_numpy()/60
    day=pd.to_datetime(w.timestamp).dt.dayofyear.to_numpy()
    weekday=(pd.to_datetime(w.timestamp).dt.dayofweek.to_numpy()<5).astype(float)
    lag=lambda k: np.r_[np.zeros(min(k,n)),y[:-k]] if n>k else np.zeros(n)
    X=np.column_stack([
      w.outdoor_c,w.rh_pct,w.ghi_w_m2,w.wind_m_s,occ,
      np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(2*np.pi*day/365),np.cos(2*np.pi*day/365),weekday,((h>=8)&(h<18))*weekday,
      indoor,lag(1),lag(4),lag(96),c,
      occ**2,w.outdoor_c*occ,w.ghi_w_m2/1000*occ,
      w.outdoor_c-c[:,6],c[:,7]-w.outdoor_c,np.maximum(0,w.outdoor_c-c[:,6]),np.maximum(0,c[:,7]-w.outdoor_c),
      wall,win,ua,cap,solar,people,lights,ventua,DT*lights,w.outdoor_c*w.rh_pct/100,w.ghi_w_m2/1000*c[:,2],occ*c[:,9],h])
    # Physics estimate is intentionally imperfect; it uses only current exogenous variables and indoor-start state.
    q=ua*.93*(w.outdoor_c.to_numpy()-indoor)+.9*gains
    free=indoor+DT*q/cap
    pc=np.minimum(120.,np.maximum(0,(free-c[:,6])*cap/DT));ph=np.minimum(120.,np.maximum(0,(c[:,7]-free)*cap/DT))
    physics=DT*(pc/c[:,4]+ph/c[:,5]+fan)
    info=dict(reference_hvac_kwh=hvac,lighting_kwh=DT*lights,comfort_degree_c=comfort,
              reference_indoor_end_c=end,physics_estimate_kwh=physics)
    return X.astype(np.float64),y,info

class Features:
    """Training-only scaling, RC descriptor scaling, GBC centers and applicability threshold."""
    def fit(self,X):
        self.mean=X.mean(0);self.std=np.maximum(X.std(0),1e-6)
        z=(X-self.mean)/self.std
        self.indices=np.array([0,2,4,11,27,34,35,36,40])
        self.centers=KMeans(n_clusters=8,random_state=SEED,n_init=5).fit(z[::4,self.indices]).cluster_centers_
        self.rc_mean=self.rc(X).mean(0);self.rc_std=np.maximum(self.rc(X).std(0),1e-6)
        self.threshold=np.quantile(cdist(z[:,self.indices],self.centers).min(1),.95)
        return self
    def rc(self,X):
        # tau=C/UA, total envelope+ventilation UA, solar aperture area*SHGC.
        return np.column_stack([X[:,37]/np.maximum(X[:,36],.01),X[:,36],X[:,35]*X[:,17]])
    def transform(self,X):
        z=(X-self.mean)/self.std
        dist=cdist(z[:,self.indices],self.centers)
        idx=np.argsort(dist,axis=1)[:,:5];d=np.take_along_axis(dist,idx,axis=1)
        weights=np.exp(-d);weights/=weights.sum(1,keepdims=True)
        # Five nearest-center membership weights over the fitted centre set.
        out=np.column_stack([z,(self.rc(X)-self.rc_mean)/self.rc_std,weights])
        return np.clip(out,-12,12)
    def domain(self,X):
        z=(X-self.mean)/self.std
        return cdist(z[:,self.indices],self.centers).min(1)<=self.threshold
    def save(self,p):
        np.savez(p,mean=self.mean,std=self.std,indices=self.indices,centers=self.centers,
                 rc_mean=self.rc_mean,rc_std=self.rc_std,threshold=self.threshold)
    @classmethod
    def load(cls,p):
        model=cls()
        with np.load(p,allow_pickle=False) as z:
            for key in z.files:setattr(model,key,z[key])
        return model

class PhysicsMLP:
    """NumPy temporal-feature physics-guided MLP.
    L=mean((pred-y)^2)+lambda*mean(relu(abs(pred-physics)-epsilon)^2).
    """
    def __init__(self,seed=SEED,hidden=32,lambda_phys=.05,epsilon=.2):
        rng=np.random.default_rng(seed);self.w=rng.normal(0,.07,(55,hidden));self.b=np.zeros(hidden)
        self.v=rng.normal(0,.07,hidden);self.a=0.;self.lam=lambda_phys;self.eps=epsilon
    def predict(self,X,dropout=0.,rng=None):
        h=np.tanh(X@self.w+self.b)
        if dropout:
            h=h*((rng or np.random.default_rng(SEED)).random(h.shape)>=dropout)/(1-dropout)
        return np.maximum(0,(h@self.v+self.a)*self.scale+self.offset)
    def fit(self,X,y,physics,epochs=16,seed=SEED,initialise_scale=True):
        if initialise_scale:self.scale=max(y.std(),1.);self.offset=y.mean()
        yn=(y-self.offset)/self.scale;pn=(physics-self.offset)/self.scale
        rng=np.random.default_rng(seed);params=[self.w,self.b,self.v,np.array([self.a])]
        m=[np.zeros_like(p) for p in params];v=[np.zeros_like(p) for p in params];step=0;history=[]
        for epoch in range(epochs):
            order=rng.permutation(len(y))
            for start in range(0,len(y),256):
                ii=order[start:start+256];xb=X[ii];h=np.tanh(xb@params[0]+params[1]);rawp=h@params[2]+params[3][0]
                p=np.maximum(rawp,-self.offset/self.scale)
                r=p-pn[ii];over=np.maximum(abs(r)-self.eps/self.scale,0)
                g=2*((p-yn[ii])+self.lam*np.sign(r)*over)/len(ii)
                g*=rawp>-self.offset/self.scale
                dh=g[:,None]*params[2]*(1-h*h)
                grads=[xb.T@dh,dh.sum(0),h.T@g,np.array([g.sum()])]
                step+=1
                for j,gr in enumerate(grads):
                    gr=np.clip(gr,-10,10);m[j]=.9*m[j]+.1*gr;v[j]=.999*v[j]+.001*gr*gr
                    params[j]-=.002*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            self.a=float(params[3][0]);p=self.predict(X);r=p-physics
            history.append(dict(epoch=epoch+1,data_loss=float(np.mean((p-y)**2)),
                physics_loss=float(np.mean(np.maximum(abs(r)-self.eps,0)**2)),lambda_phys=self.lam))
        self.a=float(params[3][0]);return pd.DataFrame(history)
    def save(self,p):
        np.savez(p,w=self.w,b=self.b,v=self.v,a=self.a,scale=self.scale,offset=self.offset,
                 lambda_phys=self.lam,epsilon=self.eps,architecture='NumPy physics-guided multilayer perceptron')
    @classmethod
    def load(cls,p):
        model=cls()
        with np.load(p,allow_pickle=False) as z:
            model.w=z['w'];model.b=z['b'];model.v=z['v'];model.a=float(z['a'])
            model.scale=float(z['scale']);model.offset=float(z['offset'])
            model.lam=float(z['lambda_phys']);model.eps=float(z['epsilon'])
        return model

def metrics(y,p):
    e=p-y;return dict(RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(np.mean(abs(e))),
        MAPE_pct=float(np.mean(abs(e)/np.maximum(abs(y),1.))*100),
        R2=float(1-np.sum(e*e)/max(np.sum((y-y.mean())**2),1e-12)))

def block_bootstrap(y,p,b=10000,length=672,seed=SEED,baseline=None):
    """Circular blocks with shared paired samples; centered-null bootstrap p-value."""
    n=len(y);length=min(length,n);rng=np.random.default_rng(seed)
    blocks,rem=divmod(n,length);starts=rng.integers(0,n,(b,blocks+bool(rem)))
    def means(v):
        cs=np.r_[0,np.cumsum(np.r_[v,v])]
        sums=(cs[starts[:,:blocks]+length]-cs[starts[:,:blocks]]).sum(1)
        if rem:sums+=cs[starts[:,-1]+rem]-cs[starts[:,-1]]
        return sums/n
    err=p-y;rmse=np.sqrt(means(err**2));mae=means(abs(err))
    persistence=np.r_[y[0],y[:-1]] if baseline is None else baseline
    delta=(persistence-y)**2-err**2
    ds=means(delta);observed=delta.mean()
    pv=(1+np.sum(abs(ds-observed)>=abs(observed)))/(b+1)
    return dict(n=n,block_length=length,replicates=b,RMSE_ci95=np.quantile(rmse,[.025,.975]).tolist(),
        MAE_ci95=np.quantile(mae,[.025,.975]).tolist(),mean_squared_error_gain_vs_persistence=float(observed),
        centered_null_two_sided_p=float(pv),unit='15-minute interval observations')

def dominates(a,b):return np.all(a<=b) and np.any(a<b)
def fronts(F):
    # O(N^2) sorting is sufficient for this small reproducibility benchmark.
    wins=[[] for _ in F];counts=np.zeros(len(F),int)
    for i in range(len(F)):
        for j in range(i+1,len(F)):
            if dominates(F[i],F[j]):wins[i].append(j);counts[j]+=1
            elif dominates(F[j],F[i]):wins[j].append(i);counts[i]+=1
    layer=np.flatnonzero(counts==0).tolist();out=[]
    while layer:
        out.append(layer);nxt=[]
        for i in layer:
            for j in wins[i]:
                counts[j]-=1
                if counts[j]==0:nxt.append(j)
        layer=nxt
    return out

def reference_points(h=3):
    return np.array([[a,b,c,h-a-b-c] for a in range(h+1) for b in range(h-a+1) for c in range(h-a-b+1)],float)/h

def select(F,n,rng):
    """Reference-direction niching selection with explicit min-max normalization.

    Non-dominated sorting is followed by association to a set of uniform reference
    directions, and the least-populated direction is filled first.
    """
    fs=fronts(F);chosen=[];last=[]
    for layer in fs:
        if len(chosen)+len(layer)<=n:chosen+=layer
        else:last=layer.copy();break
    if not last:return np.array(chosen[:n],int)
    norm=(F-F.min(0))/np.maximum(np.ptp(F,axis=0),1e-8)
    refs=reference_points();refs/=np.linalg.norm(refs,axis=1,keepdims=True)
    distance=np.sqrt(np.maximum(0,np.sum(norm**2,axis=1)[:,None]-(norm@refs.T)**2))
    assoc=distance.argmin(1);rho=np.bincount(assoc[chosen],minlength=len(refs))
    while len(chosen)<n:
        active=np.unique(assoc[last]);r=int(rng.choice(active[rho[active]==rho[active].min()]))
        candidates=[j for j in last if assoc[j]==r]
        j=min(candidates,key=lambda j:distance[j,r]) if rho[r]==0 else int(rng.choice(candidates))
        chosen.append(j);last.remove(j);rho[r]+=1
    return np.array(chosen,int)

def evaluate_candidates(w,C):
    """Physical reference evaluator on a disclosed representative seasonal profile.
    Objective energy is annualized from the sampled seasonal episodes.
    """
    rows=[]
    factor=365*96/len(w)
    for c in C:
        _,_,info=simulate(w,c,measurement_noise=False)
        energy=(info['reference_hvac_kwh']+info['lighting_kwh']).sum()*factor/1000
        capital=AREA*(c[0]*300+max(0,3.5-c[1])*18)+max(0,c[4]-2.5)*6500+c[10]*8000+c[11]*14000
        discomfort=float(info['comfort_degree_c'].mean())
        carbon=energy*.55+capital*.00002
        rows.append([energy,capital+energy*1000*.12*15,discomfort,carbon])
    return np.array(rows)

def optimize(w,pop=24,generations=16,seed=SEED):
    rng=np.random.default_rng(seed);C=candidate_matrix(pop,seed);F=evaluate_candidates(w,C);history=[]
    for g in range(generations):
        pc=.9-.3*g/max(1,generations-1);pm=.1-.05*g/max(1,generations-1)
        a=C[rng.integers(0,pop,pop)];b=C[rng.integers(0,pop,pop)]
        blend=rng.uniform(-.15,1.15,(pop,12));child=np.where(rng.random((pop,1))<pc,blend*a+(1-blend)*b,a)
        mutation=rng.random((pop,12))<pm
        child+=mutation*rng.normal(0,.10,(pop,12))*(HIGH-LOW)
        child=np.clip(child,LOW,HIGH);child[:,10:]=np.round(child[:,10:])
        FC=evaluate_candidates(w,child);allc=np.r_[C,child];allf=np.r_[F,FC]
        idx=select(allf,pop,rng);C=allc[idx];F=allf[idx]
        history.append(dict(generation=g+1,pc=pc,pm=pm,nondominated_count=len(fronts(F)[0]),
             min_energy_MWh=float(F[:,0].min()),min_lifecycle_cost_USD=float(F[:,1].min())))
    front=np.array(fronts(F)[0]);return C[front],F[front],pd.DataFrame(history)
