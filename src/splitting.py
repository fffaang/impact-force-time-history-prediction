"""Exact role quotas, intact specimens, response-bin balance, seeded tie-breaking."""
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from common import ROOT,load_data,save_csv,save_json,group_folds,digest


def allocate(frame,quotas,seed,names,force_first=()):
    groups=sorted(frame.specimen_group.unique());ng=len(groups);ns=len(names)
    roles=list(quotas)
    primary=frame[frame.role=='primary'].sort_values(['Fmax','curve_id'])
    bins={c:int(i*4/len(primary)) for i,c in enumerate(primary.curve_id)}
    role_counts=np.array([[((frame.specimen_group==g)&(frame.role==r)).sum() for g in groups] for r in roles])
    bin_counts=np.array([[sum(bins.get(c,-1)==q for c in frame[frame.specimen_group==g].curve_id) for g in groups] for q in range(4)])
    nx=ng*ns;nd=4*ns;nv=nx+nd
    matrix=[];lower=[];upper=[]
    for g in range(ng):
        r=np.zeros(nv);r[g*ns:(g+1)*ns]=1;matrix.append(r);lower.append(1);upper.append(1)
    for k,role in enumerate(roles):
        for s in range(ns):
            r=np.zeros(nv);r[s:nx:ns]=role_counts[k];matrix.append(r);lower.append(quotas[role][s]);upper.append(quotas[role][s])
    for q in range(4):
        for s in range(ns):
            target=bin_counts[q].sum()*quotas['primary'][s]/len(primary)
            r=np.zeros(nv);r[s:nx:ns]=bin_counts[q];r[nx+q*ns+s]=-1
            matrix.append(r);lower.append(-np.inf);upper.append(target)
            r=r.copy();r[s:nx:ns]*=-1
            matrix.append(r);lower.append(-np.inf);upper.append(-target)
    lb=np.zeros(nv);ub=np.r_[np.ones(nx),np.repeat(np.inf,nd)]
    for group in force_first:lb[groups.index(group)*ns]=1
    integrality=np.r_[np.ones(nx),np.zeros(nd)]
    constraint=LinearConstraint(np.array(matrix),np.array(lower),np.array(upper))
    cost=np.r_[np.zeros(nx),np.ones(nd)]
    first=milp(cost,integrality=integrality,bounds=Bounds(lb,ub),constraints=constraint,options={'time_limit':120,'mip_rel_gap':0})
    if not first.success:raise RuntimeError('No optimal quota/bin allocation: '+first.message)
    r=np.r_[np.zeros(nx),np.ones(nd)]
    matrix.append(r);lower.append(-np.inf);upper.append(float(first.fun)+1e-7)
    cost=np.r_[np.random.default_rng(seed).random(nx),np.zeros(nd)]
    second=milp(cost,integrality=integrality,bounds=Bounds(lb,ub),
        constraints=LinearConstraint(np.array(matrix),np.array(lower),np.array(upper)),options={'time_limit':120,'mip_rel_gap':0})
    if not second.success:raise RuntimeError('Seeded allocation incomplete: '+second.message)
    chosen=np.rint(second.x[:nx]).reshape(ng,ns).argmax(1)
    assignment=dict(zip(groups,[names[i] for i in chosen]))
    return frame.specimen_group.map(assignment),dict(bin_absolute_deviation=float(first.fun),seed=seed,groups=ng)


def create_splits(config):
    if (ROOT/'FROZEN.json').exists():raise RuntimeError('Frozen splits cannot be regenerated')
    frame,_=load_data();eligible=frame[frame.role!='quarantine'].copy()
    summaries=[]
    for seed in config['split_seeds']:
        assigned,info=allocate(eligible,config['role_quotas'],seed,['train','val','test'])
        out=frame[['curve_id','specimen_group','role']].copy();out['split']='quarantine';out.loc[eligible.index,'split']=assigned
        train=frame.loc[out.split=='train'].copy()
        q={r:[n,int((train.role==r).sum())-n] for r,n in config['nested_quotas'].items()}
        own=train[train.curve_id.str.startswith('PRESENT_')].specimen_group.unique()
        nested,ni=allocate(train,q,seed+1000,['A','rest'],own)
        out['nested_A']=False;out.loc[train.index,'nested_A']=nested.eq('A')
        assert out.split.value_counts().to_dict()=={'train':158,'val':20,'test':20,'quarantine':1}
        assert out.nested_A.sum()==131 and out.groupby('specimen_group').split.nunique().max()==1
        for role,quota in config['role_quotas'].items():
            assert [(out.role.eq(role)&out.split.eq(s)).sum() for s in ('train','val','test')]==quota
        save_csv(out,ROOT/f'data/splits/{seed}.csv')
        folds=[]
        for arm,subset in [('A',frame.loc[out.nested_A]),('B_C',train)]:
            for j,(a,b) in enumerate(group_folds(subset,config['seeds'][0])):
                folds.extend(dict(arm=arm,fold=j,curve_id=c) for c in subset.iloc[b].curve_id)
        save_csv(pd.DataFrame(folds),ROOT/f'data/splits/{seed}_inner.csv')
        summaries.append(dict(split_seed=seed,**info,nested=ni,
            present_assignments=out[out.curve_id.str.startswith('PRESENT_')].to_dict('records'),
            split_hash=digest(out.to_dict('records'))))
    save_json(summaries,ROOT/'audit/split_allocation.json')
    return summaries
