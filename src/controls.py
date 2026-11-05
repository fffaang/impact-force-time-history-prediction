"""Archived architecture, with fresh training-only preprocessing and OOF weights."""
from __future__ import annotations
import copy
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,ElasticNet
from common import ROOT,RAW,TARGETS,group_folds,digest,file_hash,save_model,save_json
from features import feature_matrix,preprocessing

OLD=['Fmax','Td','Impulse','area','tp_ratio']


def legacy():
    sys.path.insert(0,str(ROOT/'legacy'))
    import vwea_core as core
    import vwea_analog_shapes as analog
    return core,analog


def catalog():
    path=ROOT/'config/control_catalog.joblib'
    if path.exists(): return joblib.load(path)
    if (ROOT/'FROZEN.json').exists(): raise RuntimeError('Missing frozen control catalog')
    core,analog=legacy();old=joblib.load(ROOT/'legacy/retained_bundle.joblib')
    targets={};report=[]
    for target,ot in zip(TARGETS,OLD):
        components=[]
        for name,pipe in old['anchors'][ot].models:
            saved=pipe.steps[-1][1];reg=clone(saved)
            # Preserve the archived selected hyperparameter, not fitted coefficients.
            if type(saved).__name__=='RidgeCV': reg=Ridge(alpha=float(saved.alpha_))
            if type(saved).__name__=='ElasticNetCV':
                reg=ElasticNet(alpha=float(saved.alpha_),l1_ratio=float(saved.l1_ratio_),max_iter=50000)
            components.append((name,reg))
            report.append(dict(target=target,component=name,params=reg.get_params(),
                parameter_origin='Archived configuration; no fitted coefficients or weights reused'))
        targets[target]=dict(kind=old['anchors'][ot].kind,components=components)
    shapes={}
    for name,c in old['candidates'].items():
        if c['kind']=='peakshift_v5':
            shapes[name]=dict(kind=c['kind'],pca=clone(c['base']['pca']),
                reg=clone(c['base']['model'].steps[-1][1]))
        else:
            m=c['model']
            if c['kind']=='local_knn_v5':
                shapes[name]=dict(kind=c['kind'],k=m.k,temperature=m.temperature,power=m.power)
            elif c['kind']=='peakwarp_knn_v5':
                shapes[name]=dict(kind=c['kind'],k=m.k,temperature=m.temperature,power=m.power)
            else:
                shapes[name]=dict(kind=c['kind'],reg=clone(m.reg),ncoef=m.ncoef,peak_warp=m.peak_warp)
    result=dict(anchors=targets,shapes=shapes,tau=old['tau'])
    save_model(result,path);save_json(report,ROOT/'audit/control_configuration.json')
    return result


def set_seed(reg,seed,smoke=False):
    reg=clone(reg);params=reg.get_params(deep=True);change={}
    for key in params:
        if key.endswith('random_state'):change[key]=seed
        if key.endswith('n_jobs'):change[key]=1
        if smoke and key.endswith(('n_estimators','max_iter')):change[key]=min(params[key] or 12,12)
    return reg.set_params(**change)


class ControlAnchor:
    def __init__(self,target,kind,reg,seed,smoke=False):
        self.target,self.kind,self.reg,self.seed,self.smoke=target,kind,reg,seed,smoke
    def fit(self,frame):
        assert frame.fit_eligible.all() and frame.split.eq('train').all()
        core,_=legacy();self.train_ids=frame.curve_id.tolist()
        self.model=Pipeline([('pre',preprocessing()),('reg',set_seed(self.reg,self.seed,self.smoke))])
        self.model.fit(feature_matrix(frame,'physics'),core.y_transform(frame[self.target].to_numpy(),self.kind))
        return self
    def predict(self,frame):
        core,_=legacy();raw=self.model.predict(feature_matrix(frame,'physics'))
        y=core.y_inverse(raw,self.kind)
        return np.clip(y,1e-6,1-1e-6) if self.target in TARGETS[3:] else np.maximum(y,1e-8)


def _component(frame,target,kind,reg,seeds,splits,smoke):
    p=np.zeros((len(frame),len(seeds)))
    for a,b in splits:
        for j,s in enumerate(seeds):p[b,j]=ControlAnchor(target,kind,reg,s,smoke).fit(frame.iloc[a]).predict(frame.iloc[b])
    models=[ControlAnchor(target,kind,reg,s,smoke).fit(frame) for s in seeds]
    return p,models


class ControlSelector:
    def __init__(self,config,budget):self.config,self.budget=config,budget
    def select(self,frame,force=False):
        from anchors import Ensemble
        core,_=legacy();frame=frame.reset_index(drop=True)
        key=digest(dict(config=self.config,code=file_hash(__file__),catalog=file_hash(ROOT/'config/control_catalog.joblib'),
            values=pd.util.hash_pandas_object(frame[RAW+TARGETS+['curve_id','specimen_group']],index=False).tolist()))[:24]
        path=ROOT/f'results/control_cache/{key}/selected.joblib'
        if path.exists() and not force:return joblib.load(path)
        models={};lineage=[];splits=group_folds(frame,self.config['seeds'][0])
        for a,b in splits:lineage.append(dict(fit_ids=frame.iloc[a].curve_id.tolist(),heldout_ids=frame.iloc[b].curve_id.tolist()))
        for target,spec in catalog()['anchors'].items():
            self.budget.check();components=spec['components']
            with joblib.parallel_config(backend='loky',n_jobs=self.config.get('cpu_workers',1),inner_max_num_threads=2):
                fitted=joblib.Parallel()(joblib.delayed(_component)(frame,target,spec['kind'],reg,self.config['seeds'],splits,self.config.get('smoke',False)) for _,reg in components)
            p=np.column_stack([x[0].mean(1) for x in fitted]);weights=core.opt_weights(p,frame[target].to_numpy())
            models[target]=Ensemble([x[1] for x in fitted],weights,frame.curve_id.tolist(),[name for name,_ in components])
        result=dict(models=models,train_ids=frame.curve_id.tolist(),key=key,lineage=lineage)
        save_model(result,path);return result
    def crossfit(self,frame):
        from anchors import predict_anchors
        out=np.zeros((len(frame),5));lineage=[]
        for j,(a,b) in enumerate(group_folds(frame,self.config['seeds'][0])):
            model=self.select(frame.iloc[a]);out[b]=predict_anchors(model,frame.iloc[b])
            lineage.append(dict(fold=j,fit_ids=model['train_ids'],heldout_ids=frame.iloc[b].curve_id.tolist(),selection_key=model['key']))
        return out,lineage


def condition_frame(anchors):
    a=pd.DataFrame(anchors,columns=['hat_'+t for t in OLD])
    a['hat_area_from_impulse']=(a.hat_Impulse/(a.hat_Fmax*a.hat_Td+1e-9)).clip(.02,.95)
    for col in list(a):a['log1p_'+col]=np.log1p(np.maximum(a[col],0))
    return a


def augmented(frame,anchors):
    x=pd.DataFrame(feature_matrix(frame,'physics'))
    x.columns=['engineering_'+str(i) for i in range(x.shape[1])]
    return pd.concat([x,condition_frame(anchors)],axis=1)


class ControlShapes:
    def __init__(self,config):self.config=config
    def fit(self,frame,anchor_oof,curves):
        core,analog=legacy();self.train_ids=frame.curve_id.tolist();self.tau=catalog()['tau']
        x=augmented(frame,anchor_oof)
        y=np.array([np.interp(self.tau*row.Td,*curves[row.curve_id])/row.Fmax for row in frame.itertuples()])
        self.members={}
        for name,spec in catalog()['shapes'].items():
            members=[]
            for seed in self.config['seeds']:
                kind=spec['kind'];pre=preprocessing()
                if kind=='local_knn_v5':model=analog.LocalKNNCurveRegressor(pre,spec['k'],spec['temperature'],spec['power']).fit(x,y)
                elif kind=='peakwarp_knn_v5':model=analog.PeakWarpLocalKNNRegressor(pre,self.tau,spec['k'],spec['temperature'],spec['power']).fit(x,y)
                elif kind=='dct_v5':
                    model=analog.DCTCurveRegressor(pre,set_seed(spec['reg'],seed,self.config.get('smoke',False)),spec['ncoef'],spec['peak_warp'],self.tau).fit(x,y)
                else:
                    pca=clone(spec['pca']);pca.set_params(n_components=min(pca.n_components,len(frame)-1))
                    scores=pca.fit_transform(y)
                    reg=Pipeline([('pre',pre),('reg',set_seed(spec['reg'],seed,self.config.get('smoke',False)))]).fit(x,scores)
                    model=dict(pca=pca,reg=reg)
                members.append(model)
            self.members[name]=members
        return self
    def parts(self,frame,anchors,seed_index=None):
        core,analog=legacy();x=augmented(frame,anchors);area=x.hat_area_from_impulse.to_numpy();out={}
        for name,members in self.members.items():
            chosen=members if seed_index is None else [members[seed_index]]
            predictions=[]
            for model in chosen:
                if isinstance(model,dict):
                    p=model['pca'].inverse_transform(model['reg'].predict(x))
                    p=analog._peak_shift(p,self.tau,anchors[:,4])
                else:p=model.predict(x)
                predictions.append(core.post(p,self.tau,area,do_smooth=False))
            out[name]=np.mean(predictions,axis=0)
        return out


class ControlModel:
    def __init__(self,anchors,shapes,names,weights):
        self.anchors,self.shapes,self.names,self.weights=anchors,shapes,names,weights
        self.train_ids=anchors['train_ids']
    def predict(self,frame,seed_index=None):
        from anchors import predict_anchors
        core,_=legacy();a=predict_anchors(self.anchors,frame,seed_index)
        parts=self.shapes.parts(frame,a,seed_index);area=condition_frame(a).hat_area_from_impulse.to_numpy()
        y=core.post(sum(w*parts[n] for n,w in zip(self.names,self.weights)),self.shapes.tau,area,do_smooth=False)
        curves=[(self.shapes.tau*aa[1],yy*aa[0]) for aa,yy in zip(a,y)]
        return a,curves,{'area_target':area}
