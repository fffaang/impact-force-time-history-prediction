from __future__ import annotations
import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import (ExtraTreesRegressor,RandomForestRegressor,
    GradientBoostingRegressor,HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge,HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from common import ROOT,TARGETS,RAW,digest,save_csv,save_json,save_model,group_folds,file_hash
from features import feature_matrix,preprocessing


def specifications(config,target):
    transforms=['direct','logit'] if target in ('Astar','tp_ratio') else ['direct','log']
    if target == 'I': transforms += ['momentum_residual']
    if target == 'tp_ratio': transforms += ['time_ratio']
    specs=[]
    for family in config['families']:
        choices=[dict(family=family,features=f,transform=t,level=l)
            for f,t,l in itertools.product(config['feature_sets'],transforms,[0,1])]
        assert len(choices)<=config['max_configs_per_family_target']
        specs.extend(choices)
    return specs


def estimator(spec,seed,threads=1):
    f,l=spec['family'],spec['level']
    if f=='Ridge': return Ridge(alpha=[1.,30.][l])
    if f=='Huber': return HuberRegressor(alpha=[.01,1.][l],epsilon=[1.35,1.75][l],max_iter=1000)
    if f=='SVR': return SVR(C=[3.,30.][l],gamma=['scale',.03][l],epsilon=.05)
    if f=='KernelRidge': return KernelRidge(alpha=[.1,1.][l],kernel='rbf',gamma=[.01,.05][l])
    if f=='ExtraTrees': return ExtraTreesRegressor(n_estimators=160,min_samples_leaf=[1,3][l],max_features=[1.,.7][l],random_state=seed,n_jobs=threads)
    if f=='RandomForest': return RandomForestRegressor(n_estimators=160,min_samples_leaf=[1,3][l],max_features=[1.,.7][l],random_state=seed,n_jobs=threads)
    if f=='GradientBoosting': return GradientBoostingRegressor(n_estimators=160,max_depth=[2,3][l],learning_rate=.04,min_samples_leaf=3,random_state=seed)
    if f=='HistGradientBoosting': return HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=[7,15][l],min_samples_leaf=6,l2_regularization=[1.,10.][l],early_stopping=False,random_state=seed)
    if f=='XGBoost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200,max_depth=[2,4][l],learning_rate=.04,
            min_child_weight=3,reg_lambda=[5.,15.][l],objective='reg:squarederror',
            tree_method='hist',n_jobs=threads,random_state=seed)
    if f=='CatBoost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=200,depth=[3,5][l],learning_rate=.04,
            l2_leaf_reg=[5.,15.][l],loss_function='RMSE',thread_count=threads,
            random_seed=seed,verbose=False,allow_writing_files=False)
    raise ValueError(f)


class Anchor:
    def __init__(self,spec,target,seed): self.spec,self.target,self.seed=spec,target,seed
    def fit(self,frame):
        assert (frame.split=='train').all() and frame.fit_eligible.all()
        self.train_ids=frame.curve_id.tolist()
        self.pre=preprocessing()
        x=self.pre.fit_transform(feature_matrix(frame,self.spec['features']))
        y=frame[self.target].to_numpy(float); mode=self.spec['transform']
        if mode=='log': y=np.log(np.maximum(y,1e-10))
        elif mode=='logit': y=logit(np.clip(y,1e-6,1-1e-6))
        elif mode=='momentum_residual': y=np.log(y/(frame.m.to_numpy()*frame.v.to_numpy()))
        elif mode=='time_ratio':
            y=np.log(np.maximum(y*frame.Td.to_numpy(),1e-10))
            self.duration=Anchor(self.spec|{'transform':'log'},'Td',self.seed).fit(frame)
        self.mean=float(y.mean()); self.scale=max(float(y.std()),1e-10)
        self.model=estimator(self.spec,self.seed)
        self.model.fit(x,(y-self.mean)/self.scale)
        return self
    def predict(self,frame):
        y=self.model.predict(self.pre.transform(feature_matrix(frame,self.spec['features'])))*self.scale+self.mean
        mode=self.spec['transform']
        if mode in ('log','momentum_residual','time_ratio'): y=np.exp(np.clip(y,-25,30))
        if mode=='logit': y=expit(y)
        if mode=='momentum_residual': y*=frame.m.to_numpy()*frame.v.to_numpy()
        if mode=='time_ratio': y/=self.duration.predict(frame)
        return np.clip(y,1e-6,1-1e-6) if self.target in ('Astar','tp_ratio') else np.maximum(y,1e-8)


class Ensemble:
    def __init__(self,models,weights,train_ids,specs):
        self.models,self.weights,self.train_ids,self.specs=models,np.asarray(weights),train_ids,specs
    def predict(self,frame,seed_index=None):
        preds=[]
        for members in self.models:
            preds.append(np.mean([m.predict(frame) for m in members],axis=0) if seed_index is None
                         else members[seed_index].predict(frame))
        return np.column_stack(preds)@self.weights


def folds(frame,seed):
    return group_folds(frame,seed)


def fit_candidate(frame,target,number,spec,splits,seeds,path):
    if path.exists():return joblib.load(path)
    predictions=np.zeros((len(frame),len(seeds)));messages=[]
    for a,b in splits:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for j,seed in enumerate(seeds):
                model=Anchor(spec,target,seed).fit(frame.iloc[a])
                predictions[b,j]=model.predict(frame.iloc[b])
            messages.extend(str(w.message) for w in caught)
    pred=predictions.mean(axis=1);true=frame[target].to_numpy()
    item=dict(spec=spec,number=number,predictions=predictions,
        R2=float(r2_score(true,pred)),MdRE=float(np.median(abs(pred-true)/np.maximum(abs(true),1e-10))*100),
        warnings=sorted(set(messages)))
    save_model(item,path);return item


class Selector:
    def __init__(self,config,budget,cache='search'):
        self.config,self.budget=config,budget
        self.root=ROOT/'results'/cache; self.root.mkdir(parents=True,exist_ok=True)
    def select(self,frame):
        assert (frame.split=='train').all() and frame.fit_eligible.all()
        frame=frame.reset_index(drop=True)
        key=digest(dict(config=self.config,code={p.name:file_hash(p) for p in (ROOT/'src').glob('*.py')},
            values=pd.util.hash_pandas_object(frame[RAW+TARGETS+['curve_id','specimen_group']],index=False).tolist()))[:24]
        out=self.root/key; out.mkdir(exist_ok=True)
        final=out/'selected.joblib'
        if final.exists(): return joblib.load(final)
        splits=folds(frame,self.config['seeds'][0])
        save_json([dict(fold=f,train_ids=frame.iloc[a].curve_id.tolist(),heldout_ids=frame.iloc[b].curve_id.tolist())
            for f,(a,b) in enumerate(splits)],out/'selection_lineage.json')
        selected={}; rankings=[]
        for target in TARGETS:
            candidates=[]
            specs=specifications(self.config,target)
            workers=int(self.config.get('cpu_workers',1));self.budget.check()
            if workers>1:
                with joblib.parallel_config(backend='loky',n_jobs=workers,
                        inner_max_num_threads=int(self.config.get('worker_threads',1))):
                    stream=joblib.Parallel(return_as='generator_unordered',pre_dispatch=workers)(
                        joblib.delayed(fit_candidate)(frame,target,n,s,splits,self.config['seeds'],out/f'{target}_{n:03d}.joblib')
                        for n,s in enumerate(specs))
                    for item in stream:
                        self.budget.check();candidates.append(item)
            else:
                for number,spec in enumerate(specs):
                    self.budget.check()
                    candidates.append(fit_candidate(frame,target,number,spec,splits,self.config['seeds'],out/f'{target}_{number:03d}.joblib'))
            candidates.sort(key=lambda r:(-r['R2'],r['MdRE'],r['number']))
            # At most three different algorithm families; weights use only training OOF.
            best=[]
            for c in candidates:
                if c['spec']['family'] not in [b['spec']['family'] for b in best]: best.append(c)
                if len(best)==3: break
            p=np.column_stack([c['predictions'].mean(1) for c in best]); y=frame[target].to_numpy()
            scale=max(float(np.var(y)),1e-12)
            opt=minimize(lambda w:float(np.mean((p@w-y)**2)/scale),np.repeat(1/len(best),len(best)),
                method='SLSQP',bounds=[(0.,1.)]*len(best),constraints={'type':'eq','fun':lambda w:sum(w)-1},
                options={'maxiter':200,'ftol':1e-12})
            weights=opt.x if opt.success else np.r_[1.,np.zeros(len(best)-1)]
            if r2_score(y,p@weights)<best[0]['R2']-1e-10: weights=np.r_[1.,np.zeros(len(best)-1)]
            keep=weights>1e-8; weights=weights[keep]/weights[keep].sum(); best=[c for c,k in zip(best,keep) if k]
            models=[[Anchor(c['spec'],target,s).fit(frame) for s in self.config['seeds']] for c in best]
            selected[target]=Ensemble(models,weights,frame.curve_id.tolist(),[c['spec'] for c in best])
            ranks=sorted([next(c for c in candidates if c['spec']['family']==f) for f in self.config['families']],key=lambda c:(-c['R2'],c['MdRE']))
            rankings.extend(dict(target=target,rank=i+1,family=c['spec']['family'],inner_R2=c['R2'],
                inner_MdRE_percent=c['MdRE'],spec=json.dumps(c['spec']),n=len(frame)) for i,c in enumerate(ranks))
            print(f'SELECTED n={len(frame)} target={target} best_inner_R2={candidates[0]["R2"]:.6f}',flush=True)
            save_csv(pd.DataFrame(rankings),out/'family_rankings.csv')
        result=dict(models=selected,train_ids=frame.curve_id.tolist(),key=key,rankings=rankings)
        save_model(result,final); return result
    def crossfit(self,frame):
        out=np.zeros((len(frame),5)); provenance=[]
        for fold,(a,b) in enumerate(folds(frame,self.config['seeds'][0])):
            fit=self.select(frame.iloc[a].reset_index(drop=True))
            assert not set(frame.iloc[b].curve_id)&set(fit['train_ids'])
            assert not set(frame.iloc[b].specimen_group)&set(frame.iloc[a].specimen_group)
            out[b]=predict_anchors(fit,frame.iloc[b])
            provenance.append(dict(fold=fold,selection_key=fit['key'],fit_ids=fit['train_ids'],heldout_ids=frame.iloc[b].curve_id.tolist()))
        return out,provenance


def predict_anchors(fitted,frame,seed_index=None):
    return np.column_stack([fitted['models'][t].predict(frame,seed_index) for t in TARGETS])
