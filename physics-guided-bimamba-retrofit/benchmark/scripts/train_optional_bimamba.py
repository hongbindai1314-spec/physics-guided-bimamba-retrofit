#!/usr/bin/env python3
"""Optional genuine BiMamba training. Requires PyTorch and a CUDA mamba-ssm build."""
import argparse,sys,json,copy,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
try:
    import torch
    from torch.utils.data import Dataset,DataLoader
    from pipeline.optional_bimamba import BiMamba,physical_loss
except ImportError as e:
    raise SystemExit('Optional route requires PyTorch and a compatible mamba-ssm/CUDA installation. '+str(e))
from pipeline.core import SEED,metrics

class Windows(Dataset):
    def __init__(self,X,y,p,indices,lookback):self.X=X;self.y=y;self.p=p;self.indices=indices;self.lookback=lookback
    def __len__(self):return len(self.indices)
    def __getitem__(self,k):
        i=self.indices[k]
        # The current exogenous controls are assumed known; observed load channels are lagged.
        return self.X[i-self.lookback+1:i+1],self.y[i],self.p[i],i

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--lookback',type=int,default=96)
    ap.add_argument('--batch-size',type=int,default=128);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise SystemExit('CUDA unavailable; optional BiMamba route not run.')
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    df=pd.read_csv(ROOT/'data/features_55d.csv.gz')
    X=torch.from_numpy(df[[f'feat_{i:02d}' for i in range(55)]].to_numpy(np.float32))
    y=torch.from_numpy(df.target_kwh.to_numpy(np.float32));p=torch.from_numpy(df.physics_kwh.to_numpy(np.float32))
    valid=np.arange(len(df))>=a.lookback-1
    ids={s:np.flatnonzero((df.split.to_numpy()==s)&valid) for s in ['train','validation','test']}
    loaders={s:DataLoader(Windows(X,y,p,ids[s],a.lookback),batch_size=a.batch_size,shuffle=(s=='train')) for s in ids}
    config=dict(d_model=64,d_state=16,layers=2,dropout=.2)
    model=BiMamba(**config).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    scale=float(y[ids['train']].std().clamp(min=1.));best=float('inf');state=None;bad=0;history=[]
    for epoch in range(a.epochs):
        model.train();data_total=phys_total=0.;count=0
        for xb,yb,pb,_ in loaders['train']:
            xb,yb,pb=xb.to(a.device),yb.to(a.device),pb.to(a.device)
            pred=model(xb)*scale;loss,data,physical=physical_loss(pred,yb,pb,scale)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            data_total+=float(data.detach())*len(yb);phys_total+=float(physical.detach())*len(yb);count+=len(yb)
        model.eval();valsum=0.;valn=0
        with torch.no_grad():
            for xb,yb,pb,_ in loaders['validation']:
                pred=model(xb.to(a.device))*scale;valsum+=float(((pred-yb.to(a.device))**2).sum());valn+=len(yb)
        vm=valsum/valn;history.append(dict(epoch=epoch+1,data_loss_scaled=data_total/count,physics_loss_scaled=phys_total/count,validation_MSE=vm))
        print(history[-1],flush=True)
        if vm<best:best=vm;state=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if bad>=10:break
    model.load_state_dict(state);model.eval();rows=[]
    with torch.no_grad():
        for xb,yb,pb,index in loaders['test']:
            pred=(model(xb.to(a.device))*scale).cpu().numpy()
            for i,truth,pp in zip(index.numpy(),yb.numpy(),pred):rows.append([df.timestamp.iloc[int(i)],truth,pp])
    (ROOT/'models').mkdir(exist_ok=True)
    torch.save(dict(state_dict={k:v.cpu() for k,v in state.items()},architecture=config,scale=scale,lookback=a.lookback,
        lambda_phys=.05,epsilon_kwh=.2,origin='independently trained BiMamba'),ROOT/'models/optional_bimamba.pt')
    predictions=pd.DataFrame(rows,columns=['timestamp','y_true','optional_bimamba'])
    predictions.to_csv(ROOT/'results/optional_bimamba_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(ROOT/'results/optional_bimamba_training.csv',index=False)
    (ROOT/'results/optional_bimamba_metrics.json').write_text(json.dumps(metrics(predictions.y_true.to_numpy(),predictions.optional_bimamba.to_numpy()),indent=2))

if __name__=='__main__':main()
