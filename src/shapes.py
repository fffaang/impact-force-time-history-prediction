from __future__ import annotations
import itertools
import numpy as np
from sklearn.decomposition import PCA,NMF
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from common import anchors_from_curve
from features import shape_inputs,preprocessing


def aligned(curve,u):
    a=anchors_from_curve(curve); phase=np.clip(a[4],1e-6,1-1e-6)
    tau=np.where(u<=.5,2*u*phase,phase+2*(u-.5)*(1-phase))
    return np.interp(tau*a[1],curve[0],curve[1])/a[0]


def shape_specs(config):
    specs=[]
    for features in config['feature_sets']:
        for k in (3,7): specs.append(dict(kind='Analog',k=k,features=features))
        for kind,reg,k in itertools.product(('PCA','NMF'),('Ridge','ExtraTrees'),config['shape_dimensions']):
            specs.append(dict(kind=kind,reg=reg,k=k,features=features))
        for width,k,epochs in itertools.product(config['shape_widths'],config['shape_dimensions'],config['neural_epochs']):
            specs.append(dict(kind='Decoder',width=width,k=k,epochs=epochs,features=features))
    return [s|{'area':area} for s in specs for area in ('impulse','independent_A')]


def canonical_to_physical(raw,u,anchors,area_mode='impulse'):
    F,T,I,A,phase=np.asarray(anchors,float)
    phase=float(np.clip(phase,1e-6,1-1e-6))
    requested=I/(F*T) if area_mode=='impulse' else A
    target=float(np.clip(requested,1e-6,1-1e-6))
    delta=min(target/20,(1-target)/20,phase/4,(1-phase)/4,.0002)
    tau=np.unique(np.r_[np.linspace(0,1,201),delta,phase-delta,phase,phase+delta,1-delta])
    location=np.where(tau<=phase,.5*tau/phase,.5+.5*(tau-phase)/(1-phase))
    before=np.interp(location,u,raw)
    base=np.maximum(before,0); base/=max(base.max(),1e-12)
    peak=int(np.argmin(abs(tau-phase)))
    lower=np.zeros(len(tau)); upper=np.full(len(tau),1-1e-10)
    upper[[0,-1]]=0; lower[peak]=upper[peak]=1
    w=np.r_[np.diff(tau)[0]/2,(tau[2:]-tau[:-2])/2,np.diff(tau)[-1]/2]
    assert w@lower<=target<=w@upper
    lo,hi=-2.,2.
    for _ in range(55):
        mid=(lo+hi)/2
        if w@np.clip(base+mid,lower,upper)<target: lo=mid
        else: hi=mid
    y=np.clip(base+(lo+hi)/2,lower,upper)
    before_curve=(tau*T,base*F); after_curve=(tau*T,y*F)
    return after_curve,dict(area_mode=area_mode,requested_area=requested,used_area=target,
        area_clip=abs(target-requested),projection_mean_abs=float(w@abs(y-base)),
        independent_A_gap=float(target-A),independent_I_gap=float(F*T*target-I)),before_curve


class Shape:
    def __init__(self,spec,config): self.spec,self.config=spec,config
    def fit(self,frame,anchor_oof,curves):
        assert (frame.split=='train').all() and frame.fit_eligible.all()
        self.train_ids=frame.curve_id.tolist(); self.u=np.linspace(0,1,self.config['shape_points'])
        y=np.array([aligned(curves[cid],self.u) for cid in frame.curve_id])
        self.pre=preprocessing()
        x=self.pre.fit_transform(shape_inputs(frame,anchor_oof,self.spec['features']))
        kind=self.spec['kind']; self.models=[]
        if kind=='Analog': self.x,self.y=x,y; return self
        if kind=='Decoder':
            from neural import train_decoder
            for seed in self.config['seeds']:
                self.models.append(train_decoder(x,y,anchor_oof,[curves[c] for c in frame.curve_id],self.spec,self.config,seed))
            return self
        k=min(self.spec['k'],len(frame)-1)
        self.basis=PCA(n_components=k,svd_solver='full') if kind=='PCA' else NMF(n_components=k,
            init='nndsvda',max_iter=1500,tol=1e-4,random_state=self.config['seeds'][0])
        scores=self.basis.fit_transform(y)
        for seed in self.config['seeds']:
            reg=Ridge(alpha=10.) if self.spec['reg']=='Ridge' else ExtraTreesRegressor(n_estimators=160,min_samples_leaf=2,random_state=seed,n_jobs=1)
            reg.fit(x,scores); self.models.append(reg)
        return self
    def predict(self,frame,anchors,seed_index=None):
        x=self.pre.transform(shape_inputs(frame,anchors,self.spec['features']))
        if self.spec['kind']=='Analog':
            out=[]
            for q in x:
                distances=np.linalg.norm(self.x-q,axis=1)
                ix=np.argsort(distances,kind='stable')[:self.spec['k']]
                weights=np.exp(-(distances[ix]-distances[ix].min())/max(np.median(distances[ix]),1e-8))
                out.append(np.average(self.y[ix],axis=0,weights=weights))
            return np.asarray(out)
        models=self.models if seed_index is None else [self.models[seed_index]]
        if self.spec['kind']=='Decoder':
            from neural import predict_decoder
            return np.mean([predict_decoder(m,x) for m in models],axis=0)
        scores=np.mean([m.predict(x) for m in models],axis=0)
        if self.spec['kind']=='NMF': scores=np.maximum(scores,0)
        return self.basis.inverse_transform(scores)
