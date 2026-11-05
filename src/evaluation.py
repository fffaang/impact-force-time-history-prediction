from __future__ import annotations
import json
import time
import joblib
import numpy as np
import pandas as pd
from common import ROOT,TARGETS,partition,save_csv,save_json,file_hash
from metrics import metric_rows,summaries,paired_bootstrap


def evaluate(config,budget):
    seal_path=ROOT/'MODEL_SELECTION_FROZEN.json'
    if not seal_path.exists():raise RuntimeError('Freeze decisions for ALL three splits before opening holdouts')
    seal=json.loads(seal_path.read_text())
    for name,sha in seal['files'].items():
        if file_hash(ROOT/name)!=sha:raise RuntimeError('Selected model hash changed: '+name)
    start=ROOT/'results/evaluation_opened.json'
    if not start.exists():save_json(dict(epoch=time.time(),seal_sha256=file_hash(seal_path)),start)
    records=[];stability=[];qa=[];algorithms=[]
    for outcome in seal['outcomes']:
        if not outcome['available']:continue
        seed,arm=outcome['split_seed'],outcome['arm'];budget.check()
        selected_model=joblib.load(ROOT/f'models/{seed}/{arm}_final.joblib')
        structure=selected_model.selection['spec']['structure'] if arm=='C' else 'five_independent_legacy_outputs'
        calibration=selected_model.selection['spec'].get('calibration','none') if arm=='C' else 'none'
        for target,ensemble in selected_model.anchors['models'].items():
            derived=arm=='C' and target==('Astar' if structure=='impulse' else 'I')
            algorithms.append(dict(split_seed=seed,arm=arm,target=target,
                production_role='derived_Astar=I/(Fmax*Td)' if derived and target=='Astar' else 'derived_I=Fmax*Td*Astar' if derived else 'independent_regression',
                base_algorithm_components=json.dumps(ensemble.specs),base_ensemble_weights=json.dumps(ensemble.weights.tolist()),
                structure=structure,calibration=calibration,
                derived_note='Independent target benchmark not directly used for this derived production output' if derived else ''))
        folder=ROOT/f'results/{seed}/evaluation/{arm}';cached=folder/'complete.joblib'
        if cached.exists():done=joblib.load(cached)
        else:
            model=joblib.load(ROOT/f'models/{seed}/{arm}_final.joblib')
            repeat=joblib.load(ROOT/f'models/{seed}/{arm}_independent_refit.joblib')
            frame,curves=partition(seed);frame=frame[frame.role!='quarantine'].copy()
            if arm=='A':frame.loc[frame.split.eq('train')&~frame.nested_A,'split']='train_excluded_A'
            assert frame.split.eq('val').sum()==20 and frame.split.eq('test').sum()==20
            assert not set(model.train_ids)&set(frame[frame.split.isin(['val','test'])].curve_id)
            aa,preds,extra=model.predict(frame);bb,other,_=repeat.predict(frame)
            np.testing.assert_allclose(aa,bb,rtol=2e-7,atol=1e-7)
            max_delta=0.
            for p,q in zip(preds,other):
                np.testing.assert_allclose(p[0],q[0],rtol=2e-7,atol=1e-7)
                np.testing.assert_allclose(p[1],q[1],rtol=2e-7,atol=1e-7)
                max_delta=max(max_delta,float(np.max(abs(p[1]-q[1]))))
            rows=[dict(split_seed=seed,arm=arm,**r) for r in metric_rows(frame,curves,aa,preds,extra)]
            sr=[]
            for j,s in enumerate(config['seeds']):
                a,p,x=model.predict(frame,seed_index=j)
                sr.extend(dict(split_seed=seed,arm=arm,model_seed=s,**r) for r in metric_rows(frame,curves,a,p,x))
            # Numerical predictions must be invariant to deployment query batching.
            probe=frame.iloc[[0,len(frame)//2,len(frame)-1]]
            for k,index in enumerate([0,len(frame)//2,len(frame)-1]):
                one,curve,_=model.predict(probe.iloc[[k]])
                np.testing.assert_allclose(one[0],aa[index],rtol=1e-9,atol=1e-7)
                np.testing.assert_allclose(curve[0][1],preds[index][1],rtol=1e-9,atol=1e-7)
            long=[]
            for row,pred in zip(frame.itertuples(),preds):
                for origin,curve in [('observed',curves[row.curve_id]),('predicted',pred)]:
                    long.extend(dict(curve_id=row.curve_id,split=row.split,series=origin,time_ms=float(t),force_kN=float(f)) for t,f in zip(*curve))
            save_csv(pd.DataFrame(long),folder/'force_histories_long.csv')
            save_csv(pd.DataFrame(rows),folder/'per_curve.csv')
            check=dict(split_seed=seed,arm=arm,records=len(frame),validation=20,test=20,
                independent_refit_pass=True,max_force_difference=max_delta,batch_invariant=True,
                forces_nonnegative=bool(all(r['force_nonnegative'] for r in rows)),
                four_anchor_identity_max_abs=float(np.max(abs(aa[:,2]-aa[:,0]*aa[:,1]*aa[:,3]))) if arm=='C' else None,
                control_identity_note='A/B retain five independent regressors; no exact identity claimed' if arm!='C' else None)
            if arm=='C':np.testing.assert_allclose(aa[:,2],aa[:,0]*aa[:,1]*aa[:,3],rtol=1e-10,atol=1e-8)
            done=dict(records=rows,stability=sr,qa=check)
            from common import save_model
            save_model(done,cached)
        records.extend(done['records']);stability.extend(done['stability']);qa.append(done['qa'])
    if not records:
        save_json(dict(complete=False,reason='No independently refitted model completed within budget'),ROOT/'reports/acceptance.json');return
    df=pd.DataFrame(records);save_csv(df,ROOT/'reports/all_predictions.csv')
    save_csv(pd.DataFrame(stability),ROOT/'reports/three_seed_predictions.csv')
    save_csv(pd.DataFrame(algorithms),ROOT/'reports/production_models.csv')
    anchors,curve=summaries(df)
    save_csv(anchors,ROOT/'reports/anchor_metrics_all_populations.csv');save_csv(curve,ROOT/'reports/curve_metrics_all_populations.csv')
    save_csv(anchors[anchors.population.eq('all')&anchors.split.isin(['val','test'])],ROOT/'reports/Table_5.csv')
    save_csv(curve[curve.population.eq('all')&curve.split.isin(['val','test'])],ROOT/'reports/Table_6b.csv')
    save_csv(df[df.curve_R2<0],ROOT/'reports/negative_R2_cases.csv');save_json(qa,ROOT/'reports/numerical_QA.json')
    bootstrap=[]
    for seed in config['split_seeds']:
        for split in ('val','test'):
            for first,second in [('A','B'),('B','C')]:
                budget.check();a=df[(df.split_seed==seed)&(df.split==split)&(df.arm==first)]
                b=df[(df.split_seed==seed)&(df.split==split)&(df.arm==second)]
                if len(a)!=20 or len(b)!=20:continue
                bootstrap.extend(dict(split_seed=seed,split=split,comparison=second+'-'+first,**r)
                    for r in paired_bootstrap(a,b,config['bootstrap_replicates'],seed))
    save_csv(pd.DataFrame(bootstrap),ROOT/'reports/paired_source_bootstrap.csv')
    acceptance=[]
    for seed in config['split_seeds']:
        for arm in 'ABC':
            am=anchors[(anchors.split_seed==seed)&(anchors.arm==arm)&anchors.population.eq('all')&anchors.split.isin(['val','test'])]
            cm=curve[(curve.split_seed==seed)&(curve.arm==arm)&curve.population.eq('all')&curve.split.isin(['val','test'])]
            if len(am)!=10 or len(cm)!=2:
                acceptance.append(dict(split_seed=seed,arm=arm,status='NOT_COMPLETED'));continue
            anchor_ok=bool((am.R2>=config['target_r2']).all());median_ok=bool((cm.median_curve_R2>=config['target_r2']).all())
            no_negative=bool((cm.negative_R2_count==0).all())
            acceptance.append(dict(split_seed=seed,arm=arm,status='PASS' if anchor_ok and median_ok and no_negative else 'TARGET_NOT_MET',
                ten_anchor_R2_ge_080=anchor_ok,two_curve_medians_ge_080=median_ok,all_curve_R2_nonnegative=no_negative,
                predicted_force_nonnegative=bool(cm.force_nonnegative.all()),anchor_targets_passed=int((am.R2>=config['target_r2']).sum()),
                validation_curve_median_R2=float(cm[cm.split=='val'].median_curve_R2.iloc[0]),
                test_curve_median_R2=float(cm[cm.split=='test'].median_curve_R2.iloc[0]),
                validation_negative_count=int(cm[cm.split=='val'].negative_R2_count.iloc[0]),
                test_negative_count=int(cm[cm.split=='test'].negative_R2_count.iloc[0])))
    save_json(dict(computational_evidence_complete=True,acceptance=acceptance,
        not_blind_or_unseen_source_validation=True,external_validation_cancelled=True,
        preliminary_data_limitations=json.loads((ROOT/'audit/data_audit.json').read_text())['limitations']),ROOT/'reports/acceptance.json')
