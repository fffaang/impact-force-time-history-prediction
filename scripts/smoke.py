"""Small training-only execution tests; these are never reported as model accuracy."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import partition,TARGETS,group_folds,save_json,save_model
from joint_model import Representation,JointModel,train_network,specifications
from anchors import Selector,predict_anchors
from experiment import anchor_config,optimize_control_weights
from classical import ClassicalModel,specifications as classic_specs
from controls import catalog,ControlSelector,ControlShapes,ControlModel


class SmokeBudget:
    def check(self):pass


def run(config):
    torch.set_num_threads(2);frame,curves=partition(config['split_seeds'][0])
    # IDs are drawn exclusively from the new main TRAIN partition.
    train=frame[frame.fit_eligible].iloc[:30].reset_index(drop=True)
    a,b=group_folds(train,config['seeds'][0])[0];fit,held=train.iloc[a],train.iloc[b]
    cfg=config|dict(smoke=True,seeds=[config['seeds'][0]],cpu_workers=1,maximum_epochs=6,
        early_stopping_check_every=1,early_stopping_patience=6,device=config.get('device','cpu'),
        cuda_graph=config.get('device')=='cuda')
    budget=SmokeBudget();selector=Selector(anchor_config(cfg),budget,'smoke_anchor_cache')
    ao,lineage=selector.crossfit(fit);anchors=selector.select(fit);base=predict_anchors(anchors,held)
    model=ClassicalModel(classic_specs(cfg)[0],cfg).fit(fit,ao,curves);aa,p,extra=model.predict(held,base)
    assert np.isfinite(aa).all() and all(np.min(y)>=0 for _,y in p)
    spec=specifications(cfg)[0];rep=Representation(spec,cfg).fit(fit,ao,curves)
    arrays=rep.arrays(fit,ao,curves,True);stop=rep.arrays(held,base,curves)
    net,profile=train_network(rep,arrays,cfg,cfg['seeds'][0],stop_arrays=stop,budget=budget)
    joint=JointModel(rep,[net],[profile]);ja,jp,jx=joint.predict(held,base)
    assert np.isfinite(ja).all() and np.max(abs(jx['area_error']))<1e-8
    catalog();control=ControlSelector(cfg,budget);co,_=control.crossfit(fit);ca=control.select(fit)
    shapes=ControlShapes(cfg).fit(fit,co,curves)
    cb=predict_anchors(ca,held);parts=shapes.parts(held,cb)
    y=np.array([np.interp(shapes.tau*r.Td,*curves[r.curve_id])/r.Fmax for r in held.itertuples()])
    names,weights=optimize_control_weights(parts,y,shapes.tau)
    baseline=ControlModel(ca,shapes,names,weights);ba,bp,bx=baseline.predict(held)
    assert np.isfinite(ba).all() and all(np.min(y)>=0 for _,y in bp)
    save_model(dict(classic=model,joint=joint,control=baseline),ROOT/'working/smoke_models.joblib')
    save_json(dict(ok=True,training_only=True,fit_ids=fit.curve_id.tolist(),held_ids=held.curve_id.tolist(),
        formal_validation_or_test_used=False,device=cfg['device'],neural_profile=profile,
        classical_prediction_shape=list(aa.shape),control_prediction_shape=list(ba.shape)),ROOT/'audit/smoke.json')
    print('SMOKE PASS: classical, neural, control, nested anchor predictions; TRAIN records only',flush=True)


def capacity_diagnostic(config,budget):
    frame,curves=partition(config['split_seeds'][0]);train=frame[frame.fit_eligible].iloc[:12].copy()
    cfg=config|dict(maximum_epochs=min(100,config['maximum_epochs']),early_stopping_check_every=10)
    spec=specifications(cfg)[0]|{'calibration':'limited'}
    rows=[]
    for size in (1,len(train)):
        # Basis fitting needs at least two records; score only the chosen fitting subset.
        rep=Representation(spec|{'k':min(4,len(train)-1)},cfg).fit(train,train[TARGETS].to_numpy(),curves)
        subset=train.iloc[:size];arrays=rep.arrays(subset,subset[TARGETS].to_numpy(),curves,False)
        net,profile=train_network(rep,arrays,cfg,cfg['seeds'][0],epochs=cfg['maximum_epochs'],budget=budget)
        fitted=JointModel(rep,[net],[profile]);aa,p,extra=fitted.predict(subset,subset[TARGETS].to_numpy())
        from metrics import metric_rows
        rows.extend(dict(capacity_n=size,diagnostic='measured_training_anchors_NOT_validation_accuracy',**r)
            for r in metric_rows(subset,curves,aa,p,extra))
    import pandas as pd
    from common import save_csv
    save_csv(pd.DataFrame(rows),ROOT/'results/diagnose/capacity.csv')
