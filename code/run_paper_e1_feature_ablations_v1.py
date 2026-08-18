#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, random, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEEDS=(42,7,123)


def load(path: Path):
    try: return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(path,map_location='cpu')

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def option_zscores(s):
    return (s-s.mean(1,keepdim=True))/s.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)

def top2_margin(s):
    t=torch.topk(s,k=2,dim=1).values
    return t[:,0]-t[:,1]

@dataclass(frozen=True)
class Spec:
    name:str
    base:str='biomed'
    biomed_pair:bool=True
    clip_pair:bool=True
    biomed_z:bool=True
    clip_z:bool=True
    biomed_margin:bool=True
    clip_margin:bool=True
    hidden_dim:int=32
    dropout:float=0.1
    linear:bool=False

SPECS=[
 Spec('E1_full'),
 Spec('E1_no_margin',biomed_margin=False,clip_margin=False),
 Spec('E1_no_zscore',biomed_z=False,clip_z=False),
 Spec('E1_pairs_only',biomed_z=False,clip_z=False,biomed_margin=False,clip_margin=False),
 Spec('E1_biomed_only',clip_pair=False,clip_z=False,clip_margin=False),
 Spec('E1_clip_only',base='clip',biomed_pair=False,biomed_z=False,biomed_margin=False),
 Spec('E1_scores_only',biomed_pair=False,clip_pair=False),
 Spec('E1_linear',linear=True,hidden_dim=0,dropout=0.0),
]

class Normalizer:
    def __init__(self,bp,cp,bs,cs):
        self.bp_mean=bp.reshape(-1,bp.shape[-1]).mean(0)
        self.bp_std=bp.reshape(-1,bp.shape[-1]).std(0,unbiased=False).clamp_min(1e-6)
        self.cp_mean=cp.reshape(-1,cp.shape[-1]).mean(0)
        self.cp_std=cp.reshape(-1,cp.shape[-1]).std(0,unbiased=False).clamp_min(1e-6)
        bm=top2_margin(bs); cm=top2_margin(cs)
        self.bm_mean=bm.mean(); self.bm_std=bm.std(unbiased=False).clamp_min(1e-6)
        self.cm_mean=cm.mean(); self.cm_std=cm.std(unbiased=False).clamp_min(1e-6)

def build_features(spec:Spec,bp,cp,bs,cs,norm:Normalizer):
    parts=[]
    if spec.biomed_pair: parts.append((bp-norm.bp_mean)/norm.bp_std)
    if spec.clip_pair: parts.append((cp-norm.cp_mean)/norm.cp_std)
    if spec.biomed_z: parts.append(option_zscores(bs).unsqueeze(-1))
    if spec.clip_z: parts.append(option_zscores(cs).unsqueeze(-1))
    if spec.biomed_margin:
        x=((top2_margin(bs)-norm.bm_mean)/norm.bm_std).view(-1,1,1).expand(-1,4,1); parts.append(x)
    if spec.clip_margin:
        x=((top2_margin(cs)-norm.cm_mean)/norm.cm_std).view(-1,1,1).expand(-1,4,1); parts.append(x)
    if not parts: raise ValueError('empty features')
    return torch.cat(parts,dim=-1).contiguous()

class Residual(nn.Module):
    def __init__(self,input_dim,spec:Spec,temp=20.0):
        super().__init__(); self.temp=temp
        if spec.linear:
            self.net=nn.Linear(input_dim,1)
            nn.init.zeros_(self.net.weight); nn.init.zeros_(self.net.bias)
        else:
            self.net=nn.Sequential(nn.Linear(input_dim,spec.hidden_dim),nn.GELU(),nn.Dropout(spec.dropout),nn.Linear(spec.hidden_dim,1))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,base): return self.temp*(base+self.net(x).squeeze(-1))

def eval_model(model,x,base,gold,batch=256,device=torch.device('cpu')):
    model.eval(); out=[]; loss=0.0
    with torch.no_grad():
        for s in range(0,len(gold),batch):
            e=min(s+batch,len(gold))
            xb=x[s:e].to(device, non_blocking=True); bb=base[s:e].to(device, non_blocking=True); gb=gold[s:e].to(device, non_blocking=True)
            z=model(xb,bb); out.append(z.detach().cpu()); loss+=nn.functional.cross_entropy(z,gb,reduction='sum').item()
    logits=torch.cat(out); pred=logits.argmax(1); return float(pred.eq(gold).float().mean()),loss/len(gold),logits

def train_seed(spec,seed,tx,tbase,tgold,vx,vbase,vgold,device,epochs=100,patience=15,batch=256,lr=1e-3,wd=1e-2):
    set_seed(seed); model=Residual(tx.shape[-1],spec).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.arange(len(tgold))),batch_size=batch,shuffle=True,generator=gen,num_workers=0)
    best_acc=-1.; best_loss=1e9; best_epoch=0; best_state=None; stale=0; hist=[]
    for epoch in range(1,epochs+1):
        model.train(); total=correct=0; loss_sum=0.
        for (idx,) in loader:
            xb=tx[idx].to(device, non_blocking=True); bb=tbase[idx].to(device, non_blocking=True); gb=tgold[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True); z=model(xb,bb); loss=nn.functional.cross_entropy(z,gb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            total+=len(idx); loss_sum+=loss.item()*len(idx); correct+=z.argmax(1).eq(gb).sum().item()
        va,vl,_=eval_model(model,vx,vbase,vgold,batch,device)
        hist.append({'epoch':epoch,'train_accuracy':correct/total,'train_loss':loss_sum/total,'val_accuracy':va,'val_loss':vl})
        improved=va>best_acc+1e-12 or (abs(va-best_acc)<=1e-12 and vl<best_loss-1e-9)
        if improved:
            best_acc,best_loss,best_epoch=va,vl,epoch; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(best_state)
    return model,{'seed':seed,'best_epoch':best_epoch,'best_val_accuracy':best_acc,'best_val_loss':best_loss,'epochs_run':len(hist),'history':hist}

def exact_mcnemar(fixed,broken):
    n=fixed+broken
    if n==0:return 1.0
    k=min(fixed,broken)
    p=2*sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,p)

def paired(gold,base_pred,new_pred):
    bc=base_pred.eq(gold); nc=new_pred.eq(gold)
    fixed=int((~bc & nc).sum()); broken=int((bc & ~nc).sum())
    return {'both_correct':int((bc&nc).sum()),'both_wrong':int((~bc&~nc).sum()),'fixed':fixed,'broken':broken,'net':fixed-broken,'mcnemar_exact_p':exact_mcnemar(fixed,broken)}

def audit_permutation(model_logits_fn, bp,cp,bs,cs,gold,seed=20260728):
    orig=model_logits_fn(bp,cp,bs,cs).argmax(1)
    gen=torch.Generator().manual_seed(seed); perms=torch.stack([torch.randperm(4,generator=gen) for _ in range(len(gold))])
    rows=torch.arange(len(gold)).unsqueeze(1)
    pbp=bp[rows,perms]; pcp=cp[rows,perms]; pbs=bs[rows,perms]; pcs=cs[rows,perms]
    pp=model_logits_fn(pbp,pcp,pbs,pcs).argmax(1)
    # map permuted index back to original option index
    mapped=perms[torch.arange(len(gold)),pp]
    return int(mapped.ne(orig).sum())

def main():
    ap=argparse.ArgumentParser(description='Paper E1 feature ablations using cached frozen CLIP/BiomedCLIP features.')
    ap.add_argument('--root',type=Path,required=True,help='Directory containing joint_train12000_dev1000_multiseed_v1 and joint_holdout290_ensemble_v1.')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--only',nargs='*',help='Subset of experiment IDs, e.g. E1_full E1_no_margin.')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    args=ap.parse_args()
    if args.device=='auto': device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device=='cuda':
        if not torch.cuda.is_available(): raise SystemExit('CUDA requested but unavailable.')
        device=torch.device('cuda')
    else: device=torch.device('cpu')
    torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    print(f'device={device}',flush=True)
    tr=args.root/'joint_train12000_dev1000_multiseed_v1'; ho=args.root/'joint_holdout290_ensemble_v1'
    tb=load(tr/'train_biomed_features.pt'); tc=load(tr/'train_clip_features.pt'); vb=load(tr/'val_biomed_features.pt'); vc=load(tr/'val_clip_features.pt'); hb=load(ho/'holdout_biomed_features.pt'); hc=load(ho/'holdout_clip_features.pt')
    assert torch.equal(tb['gold_indices'],tc['gold_indices']) and torch.equal(vb['gold_indices'],vc['gold_indices']) and torch.equal(hb['gold_indices'],hc['gold_indices'])
    norm=Normalizer(tb['pair_features'].float(),tc['pair_features'].float(),tb['base_scores'].float(),tc['base_scores'].float())
    args.out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; summaries={}
    specs=[s for s in SPECS if not args.only or s.name in args.only]
    for spec in specs:
        started=time.time(); print('START',spec.name,flush=True)
        tx=build_features(spec,tb['pair_features'].float(),tc['pair_features'].float(),tb['base_scores'].float(),tc['base_scores'].float(),norm)
        vx=build_features(spec,vb['pair_features'].float(),vc['pair_features'].float(),vb['base_scores'].float(),vc['base_scores'].float(),norm)
        hx=build_features(spec,hb['pair_features'].float(),hc['pair_features'].float(),hb['base_scores'].float(),hc['base_scores'].float(),norm)
        tbase=(tb if spec.base=='biomed' else tc)['base_scores'].float(); vbase=(vb if spec.base=='biomed' else vc)['base_scores'].float(); hbase=(hb if spec.base=='biomed' else hc)['base_scores'].float()
        tgold=tb['gold_indices'].long(); vgold=vb['gold_indices'].long(); hgold=hb['gold_indices'].long()
        base_v_pred=vbase.argmax(1); base_h_pred=hbase.argmax(1); base_v_acc=float(base_v_pred.eq(vgold).float().mean()); base_h_acc=float(base_h_pred.eq(hgold).float().mean())
        models=[]; seed_results=[]; v_logits=[]; h_logits=[]
        for seed in SEEDS:
            model,sr=train_seed(spec,seed,tx,tbase,tgold,vx,vbase,vgold,device,epochs=args.epochs,patience=args.patience,batch=args.batch_size); models.append(model); seed_results.append(sr)
            _,_,vl=eval_model(model,vx,vbase,vgold,args.batch_size,device); _,_,hl=eval_model(model,hx,hbase,hgold,args.batch_size,device); v_logits.append(vl); h_logits.append(hl)
            all_rows.append({'experiment_id':spec.name,'split':'dev1000','seed':seed,'accuracy':float(vl.argmax(1).eq(vgold).float().mean()),'base_accuracy':base_v_acc,'delta':float(vl.argmax(1).eq(vgold).float().mean())-base_v_acc,'input_dim':tx.shape[-1],'trainable_parameters':sum(p.numel() for p in model.parameters())})
            all_rows.append({'experiment_id':spec.name,'split':'holdout290','seed':seed,'accuracy':float(hl.argmax(1).eq(hgold).float().mean()),'base_accuracy':base_h_acc,'delta':float(hl.argmax(1).eq(hgold).float().mean())-base_h_acc,'input_dim':tx.shape[-1],'trainable_parameters':sum(p.numel() for p in model.parameters())})
        ev=torch.stack(v_logits).mean(0); eh=torch.stack(h_logits).mean(0); evp=ev.argmax(1); ehp=eh.argmax(1)
        evacc=float(evp.eq(vgold).float().mean()); ehacc=float(ehp.eq(hgold).float().mean())
        all_rows.append({'experiment_id':spec.name,'split':'dev1000','seed':'ensemble','accuracy':evacc,'base_accuracy':base_v_acc,'delta':evacc-base_v_acc,'input_dim':tx.shape[-1],'trainable_parameters':sum(p.numel() for p in models[0].parameters())})
        all_rows.append({'experiment_id':spec.name,'split':'holdout290','seed':'ensemble','accuracy':ehacc,'base_accuracy':base_h_acc,'delta':ehacc-base_h_acc,'input_dim':tx.shape[-1],'trainable_parameters':sum(p.numel() for p in models[0].parameters())})
        def logits_fn(bp,cp,bs,cs):
            x=build_features(spec,bp,cp,bs,cs,norm); base=bs if spec.base=='biomed' else cs
            return torch.stack([eval_model(m,x,base,hgold,args.batch_size,device)[2] for m in models]).mean(0)
        mism=audit_permutation(logits_fn,hb['pair_features'].float(),hc['pair_features'].float(),hb['base_scores'].float(),hc['base_scores'].float(),hgold)
        summary={'spec':asdict(spec),'input_dim':int(tx.shape[-1]),'trainable_parameters':sum(p.numel() for p in models[0].parameters()),'seed_results':seed_results,'dev':{'base_accuracy':base_v_acc,'ensemble_accuracy':evacc,'delta':evacc-base_v_acc,'paired':paired(vgold,base_v_pred,evp)},'holdout':{'base_accuracy':base_h_acc,'ensemble_accuracy':ehacc,'delta':ehacc-base_h_acc,'paired':paired(hgold,base_h_pred,ehp)},'permutation_mismatches_holdout':mism,'runtime_seconds':time.time()-started}
        summaries[spec.name]=summary
        expdir=args.out/spec.name; expdir.mkdir(exist_ok=True)
        torch.save({'spec':asdict(spec),'state_dicts':[m.state_dict() for m in models]},expdir/'ensemble_states.pt')
        summary['environment']={'torch_version':torch.__version__,'device':str(device),'cuda_device':torch.cuda.get_device_name(0) if device.type=='cuda' else None}
        with open(expdir/'summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
        with open(expdir/'ensemble_predictions_holdout.csv','w',encoding='utf-8',newline='') as f:
            fields=['row_index','gold_index','base_pred','new_pred','base_correct','new_correct']
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for i,(g,b,n) in enumerate(zip(hgold.tolist(),base_h_pred.tolist(),ehp.tolist())):
                w.writerow({'row_index':i,'gold_index':g,'base_pred':b,'new_pred':n,'base_correct':int(b==g),'new_correct':int(n==g)})
        print('DONE',spec.name,'dev',evacc,'holdout',ehacc,'mism',mism,'sec',summary['runtime_seconds'],flush=True)
    with open(args.out/'all_summaries.json','w',encoding='utf-8') as f:json.dump(summaries,f,ensure_ascii=False,indent=2)
    with open(args.out/'ablation_results.csv','w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)

if __name__=='__main__': main()
