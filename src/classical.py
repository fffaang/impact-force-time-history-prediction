import itertools
import numpy as np
import torch
from shapes import Shape
from joint_math import consistent_anchors,reconstruct


def specifications(config):
    result=[]
    for structure,features in itertools.product(config['structures'],config['feature_sets']):
        for k in (3,7):result.append(dict(structure=structure,features=features,kind='Analog',k=k))
        for kind,reg,k in itertools.product(('PCA','NMF'),('Ridge','ExtraTrees'),config['dimensions']):
            result.append(dict(structure=structure,features=features,kind=kind,reg=reg,k=k))
    return result


class ClassicalModel:
    def __init__(self,spec,config):self.spec,self.config=spec,config
    def fit(self,frame,anchor_oof,curves):
        aa,_=consistent_anchors(torch.tensor(anchor_oof,dtype=torch.float64),self.spec['structure'])
        self.shape=Shape(self.spec,self.config|{'shape_points':self.config['canonical_points']}).fit(frame,aa.numpy(),curves)
        self.train_ids=frame.curve_id.tolist();return self
    def predict(self,frame,base_anchors,seed_index=None,projection='power',oracle_anchors=None,oracle_raw=None):
        aa,delta=consistent_anchors(torch.tensor(base_anchors,dtype=torch.float64),self.spec['structure'])
        raw=self.shape.predict(frame,aa.numpy(),seed_index)
        raw=np.maximum(raw,1e-6);raw/=np.maximum(raw.max(1,keepdims=True),1e-8)
        raw[:,[0,-1]]=0;raw[:,raw.shape[1]//2]=1
        if oracle_raw is not None:raw=oracle_raw
        if oracle_anchors is not None:aa=torch.tensor(oracle_anchors,dtype=torch.float64)
        with torch.no_grad():t,f,info=reconstruct(torch.tensor(raw,dtype=torch.float64),aa,projection)
        curves=[]
        for ti,fi in zip(t.numpy(),f.numpy()):
            unique,ix=np.unique(ti,return_index=True);curves.append((unique,fi[ix]))
        return aa.numpy(),curves,{k:v.numpy() for k,v in info.items()}|{'canonical_raw':raw,'consistency_projection_log_delta':delta.numpy()}


class OptimizedModel:
    def __init__(self,anchors,shape,selection):
        self.anchors,self.shape,self.selection=anchors,shape,selection;self.train_ids=anchors['train_ids']
    def predict(self,frame,seed_index=None):
        from anchors import predict_anchors
        return self.shape.predict(frame,predict_anchors(self.anchors,frame,seed_index),seed_index=seed_index)
