"""Training-only selection, an all-split freeze barrier, then locked evaluation."""
from __future__ import annotations
import copy
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from common import (ROOT,TARGETS,RAW,partition,group_folds,save_model,save_json,save_csv,
                    digest,file_hash,physical_metrics,anchors_from_curve)
from anchors import Selector,predict_anchors
from controls import ControlSelector,ControlShapes,ControlModel,catalog
from classical import ClassicalModel,OptimizedModel,specifications as classic_specs
from joint_model import Representation,JointModel,train_network,specifications as joint_specs
from shapes import aligned
from metrics import metric_rows,anchor_summary,rank_candidate,choose


def train_frame(seed,arm):
    frame,curves=partition(seed)
    train=frame[frame.fit_eligible & (frame.nested_A if arm=='A' else True)].reset_index(drop=True)
    return train,curves


def anchor_config(config):
    ac=json.loads((ROOT/'config/anchors.json').read_text())
    ac.update({k:config[k] for k in ('seeds','cpu_workers','worker_threads')})
    if config.get('smoke'):
        ac['families']=['Ridge'];ac['feature_sets']=['raw'];ac['cpu_workers']=1
    return ac


def context_key(frame,config,kind):
    return digest(dict(config=config,kind=kind,ids=frame.curve_id.tolist(),
        code={p.name:file_hash(p) for p in (ROOT/'src').glob('*.py')},
        data=file_hash(ROOT/'data/curves.npz')))[:24]


def contexts(frame,curves,selector,config,kind,early=False):
    root=ROOT/'results/contexts'/context_key(frame,config,kind);out=[]
    for fold,(a,b) in enumerate(group_folds(frame,config['seeds'][0])):
        path=root/f'fold_{fold}.joblib'
        if path.exists():ctx=joblib.load(path)
        else:
            fit=frame.iloc[a].reset_index(drop=True)
            selected=selector.select(fit);ao,lineage=selector.crossfit(fit)
            ctx=dict(train=a,held=b,anchor_oof=ao,anchor_held=predict_anchors(selected,frame.iloc[b]),
                lineage=lineage,selected=selected,fit_ids=fit.curve_id.tolist(),held_ids=frame.iloc[b].curve_id.tolist())
            save_model(ctx,path)
        if early and 'early_train' not in ctx:
            fit=frame.iloc[a].reset_index(drop=True);ea,eb=group_folds(fit,config['seeds'][0])[0]
            ef=fit.iloc[ea];es=fit.iloc[eb];selected=selector.select(ef)
            ao,lineage=selector.crossfit(ef)
            ctx.update(early_train=ea,early_stop=eb,early_anchors=ao,stop_anchors=predict_anchors(selected,es),
                early_lineage=lineage,early_selected=selected)
            save_model(ctx,path)
        assert ctx['fit_ids']==frame.iloc[a].curve_id.tolist()
        assert not set(frame.iloc[a].specimen_group)&set(frame.iloc[b].specimen_group)
        for lin in ctx['lineage']+ctx.get('early_lineage',[]):
            assert not set(lin['fit_ids'])&set(lin['heldout_ids'])
            assert not set(lin['fit_ids'])&set(ctx['held_ids'])
        out.append(ctx)
    return out


def optimize_control_weights(parts,true,tau):
    names=list(parts);p=np.stack([parts[n] for n in names])
    def loss(w):
        y=np.tensordot(w,p,axes=(0,0))
        return float(np.abs(y-true).mean()+.5*np.abs(np.trapezoid(y,tau)-np.trapezoid(true,tau)).mean())
    starts=[np.ones(len(names))/len(names)]+list(np.eye(len(names)));solutions=[]
    for start in starts:
        r=minimize(loss,start,method='SLSQP',bounds=[(0,1)]*len(names),
            constraints={'type':'eq','fun':lambda w:sum(w)-1},options={'maxiter':300,'ftol':1e-10})
        if r.success:solutions.append(r)
    w=min(solutions,key=lambda r:r.fun).x if solutions else starts[0]
    w=np.maximum(w,0);w/=w.sum()
    return names,w


def search_controls(seed,arm,config,budget):
    path=ROOT/f'results/{seed}/{arm}/selection.joblib'
    if path.exists():return
    frame,curves=train_frame(seed,arm);selector=ControlSelector(config,budget)
    ctxs=contexts(frame,curves,selector,config,'control')
    parts={n:np.zeros((len(frame),200)) for n in catalog()['shapes']};ao=np.zeros((len(frame),5))
    for fold,ctx in enumerate(ctxs):
        budget.check();cache=path.parent/f'fold_{fold}.joblib'
        if cache.exists():done=joblib.load(cache)
        else:
            fit,held=frame.iloc[ctx['train']],frame.iloc[ctx['held']]
            shapes=ControlShapes(config).fit(fit,ctx['anchor_oof'],curves)
            done=dict(model=shapes,parts=shapes.parts(held,ctx['anchor_held']),held_ids=held.curve_id.tolist())
            save_model(done,cache)
        for name,p in done['parts'].items():parts[name][ctx['held']]=p
        ao[ctx['held']]=ctx['anchor_held']
    tau=catalog()['tau'];truth=np.array([np.interp(tau*r.Td,*curves[r.curve_id])/r.Fmax for r in frame.itertuples()])
    names,weights=optimize_control_weights(parts,truth,tau)
    selected=selector.select(frame)
    save_model(dict(anchor_selection=selected,anchor_oof=ao,shape_names=names,shape_weights=weights,
        train_ids=frame.curve_id.tolist(),lineage=[c['lineage'] for c in ctxs]),path)
    save_json(dict(complete=True,arm=arm,n_train=len(frame),weights_origin='training-only OOF; no archived weights',
        names=names,weights=weights.tolist()),path.parent/'search_status.json')
    print(f'CONTROL_SEARCH_COMPLETE split={seed} arm={arm} n={len(frame)}',flush=True)


def optimized_contexts(seed,config,budget,early=False):
    train,curves=train_frame(seed,'C');selector=Selector(anchor_config(config),budget,'optimized_anchor_cache')
    ctxs=contexts(train,curves,selector,config,'optimized',early)
    base=np.zeros((len(train),5))
    for ctx in ctxs:base[ctx['held']]=ctx['anchor_held']
    return train,curves,selector,ctxs,base


def run_candidate(seed,number,kind,spec,config,budget):
    final=ROOT/f'results/{seed}/C/candidate_{number:03d}/complete.joblib'
    if final.exists():return joblib.load(final)
    train,curves,selector,ctxs,base=optimized_contexts(seed,config,budget,kind=='joint')
    rows=[];profiles=[]
    for fold,ctx in enumerate(ctxs):
        budget.check();path=final.parent/f'fold_{fold}/complete.joblib'
        if path.exists():done=joblib.load(path)
        else:
            fit,held=train.iloc[ctx['train']].reset_index(drop=True),train.iloc[ctx['held']]
            if kind=='classical':
                model=ClassicalModel(spec,config).fit(fit,ctx['anchor_oof'],curves);history=[]
            else:
                ef,es=fit.iloc[ctx['early_train']],fit.iloc[ctx['early_stop']]
                er=Representation(spec,config).fit(ef,ctx['early_anchors'],curves)
                ep=er.arrays(ef,ctx['early_anchors'],curves,True);stop=er.arrays(es,ctx['stop_anchors'],curves)
                full=Representation(spec,config).fit(fit,ctx['anchor_oof'],curves)
                arrays=full.arrays(fit,ctx['anchor_oof'],curves,True);models=[];history=[]
                for s in config['seeds']:
                    _,hp=train_network(er,ep,config,s,stop_arrays=stop,checkpoint=path.parent/f'early_{s}.pt',budget=budget)
                    network,pr=train_network(full,arrays,config,s,epochs=hp['selected_epochs'],checkpoint=path.parent/f'refit_{s}.pt',budget=budget)
                    models.append(network);history.append(pr|{'early_stopping':hp})
                model=JointModel(full,models,history)
            aa,pred,extra=model.predict(held,ctx['anchor_held'])
            done=dict(rows=metric_rows(held,curves,aa,pred,extra,fold),profiles=history,model=model,
                curves=pred,anchors=aa,extras=extra,held_ids=held.curve_id.tolist())
            save_model(done,path)
        rows.extend(done['rows']);profiles.append(done['profiles'])
    result=rank_candidate(number,kind,spec,rows,anchor_summary(train[TARGETS].to_numpy(),base),config)
    result.update(rows=rows,profiles=profiles);save_model(result,final)
    return result


def search_new(seeds,config,budget):
    candidates=[('classical',s) for s in classic_specs(config)]+[('joint',s) for s in joint_specs(config)]
    if config.get('smoke'):candidates=[candidates[0],candidates[len(classic_specs(config))]]
    for number,(kind,spec) in enumerate(candidates):
        for seed in seeds:
            budget.check();result=run_candidate(seed,number,kind,spec,config,budget)
            update_ranking(seed,config,len(candidates))
            print(f'CANDIDATE split={seed} completed={number+1}/{len(candidates)} median_R2={result["median_R2"]:.5f} guard={result["anchor_protection"]}',flush=True)
    for seed in seeds:update_ranking(seed,config,len(candidates))


def update_ranking(seed,config,planned=152):
    root=ROOT/f'results/{seed}/C';rankings=[]
    for path in sorted(root.glob('candidate_*/complete.joblib')):
        result=joblib.load(path);rankings.append({k:v for k,v in result.items() if k not in ('rows','profiles')})
    save_json(rankings,root/'rankings.json')
    save_csv(pd.DataFrame([{k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in rankings]),root/'rankings.csv')
    selected=choose(rankings,config)
    status=dict(complete=len(rankings)==planned,planned=planned,completed=len(rankings),selected=selected,
        selection_population='training-only grouped OOF',no_eligible_completed_candidate=selected is None)
    save_json(status,root/'search_status.json');return status


def refit_anchor_selection(selected,train):
    fitted=copy.deepcopy(selected)
    for ensemble in fitted['models'].values():
        ensemble.models=[[copy.deepcopy(member).fit(train) for member in members] for members in ensemble.models]
        ensemble.train_ids=train.curve_id.tolist()
    fitted['train_ids']=train.curve_id.tolist();return fitted


def refit_control(seed,arm,config,budget):
    selection=ROOT/f'results/{seed}/{arm}/selection.joblib'
    if not selection.exists():return False
    selected=joblib.load(selection);train,curves=train_frame(seed,arm)
    assert selected['train_ids']==train.curve_id.tolist()
    for version in ('final','independent_refit'):
        path=ROOT/f'models/{seed}/{arm}_{version}.joblib'
        if path.exists():continue
        budget.check();anchors=refit_anchor_selection(selected['anchor_selection'],train)
        shapes=ControlShapes(config).fit(train,selected['anchor_oof'],curves)
        save_model(ControlModel(anchors,shapes,selected['shape_names'],selected['shape_weights']),path)
    return True


def refit_optimized(seed,config,budget):
    status=update_ranking(seed,config,2 if config.get('smoke') else 152)
    selected=status['selected']
    if selected is None:return False
    train,curves,selector,ctxs,ao=optimized_contexts(seed,config,budget)
    anchor_fit=selector.select(train)
    save_csv(pd.DataFrame(anchor_fit['rankings']),ROOT/f'reports/anchor_algorithm_rankings_{seed}.csv')
    complete=joblib.load(ROOT/f'results/{seed}/C/candidate_{selected["number"]:03d}/complete.joblib')
    for version in ('final','independent_refit'):
        path=ROOT/f'models/{seed}/C_{version}.joblib'
        if path.exists():continue
        budget.check();anchors=refit_anchor_selection(anchor_fit,train)
        if selected['kind']=='classical':shape=ClassicalModel(selected['spec'],config).fit(train,ao,curves)
        else:
            rep=Representation(selected['spec'],config).fit(train,ao,curves);arrays=rep.arrays(train,ao,curves,True)
            networks=[];profiles=[]
            for j,s in enumerate(config['seeds']):
                epochs=int(np.median([f[j]['selected_epochs'] for f in complete['profiles']]))
                net,prof=train_network(rep,arrays,config,s,epochs=epochs,
                    checkpoint=ROOT/f'working/{seed}_{version}_{s}.pt',budget=budget)
                networks.append(net);profiles.append(prof)
            shape=JointModel(rep,networks,profiles)
        save_model(OptimizedModel(anchors,shape,selected),path)
    return True


def training_oracle(seed,config,budget):
    status_path=ROOT/f'results/{seed}/C/search_status.json'
    if not status_path.exists():return
    selected=json.loads(status_path.read_text())['selected']
    if selected is None:return
    train,curves,_,ctxs,base=optimized_contexts(seed,config,budget);rows=[]
    for fold,ctx in enumerate(ctxs):
        done=joblib.load(ROOT/f'results/{seed}/C/candidate_{selected["number"]:03d}/fold_{fold}/complete.joblib')
        held=train.iloc[ctx['held']];model=done['model'];actual=held[TARGETS].to_numpy()
        measured=np.array([aligned(curves[c],np.linspace(0,1,config['canonical_points'])) for c in held.curve_id])
        cases={'A_production':(done['anchors'],done['curves'],done['extras']),
            'A_without_area_projection':model.predict(held,ctx['anchor_held'],projection='none'),
            'B_measured_anchors_predicted_shape':model.predict(held,ctx['anchor_held'],oracle_anchors=actual),
            'C_predicted_anchors_measured_shape':model.predict(held,ctx['anchor_held'],oracle_raw=measured),
            'D_measured_anchors_measured_shape':model.predict(held,ctx['anchor_held'],oracle_anchors=actual,oracle_raw=measured),
            'D_exact_PWL_reference':(actual,[curves[c] for c in held.curve_id],{})}
        for case,(aa,pred,extra) in cases.items():
            rows.extend(dict(case=case,**r) for r in metric_rows(held,curves,aa,pred,extra,fold))
    save_csv(pd.DataFrame(rows),ROOT/f'results/{seed}/C/training_oracle.csv')


def seal_models(config,reason):
    # An explicit absence is frozen too. Later searching/refitting is forbidden.
    path=ROOT/'MODEL_SELECTION_FROZEN.json'
    if path.exists():return json.loads(path.read_text())
    outcomes=[];files={}
    for seed in config['split_seeds']:
        for arm in 'ABC':
            paths=[ROOT/f'models/{seed}/{arm}_{v}.joblib' for v in ('final','independent_refit')]
            ready=all(p.exists() for p in paths)
            outcomes.append(dict(split_seed=seed,arm=arm,available=ready))
            if ready:files.update({p.relative_to(ROOT).as_posix():file_hash(p) for p in paths})
    result=dict(epoch=time.time(),reason=reason,outcomes=outcomes,files=files)
    save_json(result,path);return result
