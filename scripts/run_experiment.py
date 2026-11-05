from pathlib import Path
import argparse
import json
import os
import sys
import time
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if (ROOT/'.test_dependencies').exists():sys.path.insert(0,str(ROOT/'.test_dependencies'))
import yaml
from common import Budget,fingerprint,save_json,file_hash,load_data,TARGETS,anchors_from_curve


def audit(config):
    import numpy as np
    from controls import catalog
    from classical import specifications as cs
    from joint_model import specifications as js
    frame,curves=load_data();assert len(frame)==199 and frame.role.ne('quarantine').sum()==198
    assert len(cs(config))==56 and len(js(config))==96
    assert config['positive_correction_factor']==1.1 and config['bounded_logit_correction']==.2
    assert config['canonical_points']==201
    for row in frame.itertuples():np.testing.assert_allclose(anchors_from_curve(curves[row.curve_id]),[getattr(row,t) for t in TARGETS],rtol=2e-11)
    catalog();save_json(dict(ok=True,archive=199,eligible=198,classical_candidates=56,joint_candidates=96,
        literature_pointwise_QA_complete=False,external_validation=False),ROOT/'audit/implementation_audit.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/split811_20261102.yml')
    p.add_argument('--stage',choices=['audit','split','diagnose','search','refit','evaluate','export','all'],default='all')
    p.add_argument('--split',default='all');p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--freeze',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();config=yaml.safe_load((ROOT/a.config).read_text())
    if a.device:config['device']=a.device
    if a.smoke:
        from smoke import run
        run(config);return
    if a.stage=='audit':audit(config)
    elif a.stage=='split':
        from splitting import create_splits
        create_splits(config);return
    if a.freeze:
        if a.stage!='audit':raise ValueError('Freeze only after audit and successful tests')
        tests=json.loads((ROOT/'audit/server_preflight.json').read_text())
        assert tests['tests_passed'] and tests['cuda_verified'] and tests['smoke_passed'] and tests['all_anchor_families_verified']
        frozen=ROOT/'FROZEN.json'
        if frozen.exists():assert json.loads(frozen.read_text())==fingerprint()
        else:save_json(fingerprint(),frozen)
        return
    if a.stage=='audit':return
    if not (ROOT/'FROZEN.json').exists():raise RuntimeError('Formal stages require server freeze')
    if json.loads((ROOT/'FROZEN.json').read_text())!=fingerprint():raise RuntimeError('Frozen files changed')
    if config['device']!=yaml.safe_load((ROOT/a.config).read_text())['device']:raise RuntimeError('Cannot change device after freeze')
    sealed_resume=False
    if a.stage in ('all','search','refit') and (ROOT/'MODEL_SELECTION_FROZEN.json').exists():
        if a.stage=='all':sealed_resume=True
        else:raise RuntimeError('Models already sealed; no holdout-driven retraining allowed')
    seeds=config['split_seeds'] if a.split=='all' else [int(a.split)]
    assert all(s in config['split_seeds'] for s in seeds)
    from experiment import search_controls,search_new,refit_control,refit_optimized,training_oracle,seal_models,update_ranking
    from evaluation import evaluate
    stages=(['evaluate','export'] if sealed_resume else ['diagnose','search','refit','evaluate','export']) if a.stage=='all' else [a.stage]
    for stage in stages:
        save_json(dict(stage=stage,status='running',updated_epoch=time.time()),ROOT/'working/progress.json')
        budget=Budget(stage,config['budget_hours'].get(stage,6));errors=[]
        try:
            if stage=='diagnose':
                from smoke import capacity_diagnostic
                capacity_diagnostic(config,budget)
            elif stage=='search':
                for seed in seeds:
                    for arm in 'AB':search_controls(seed,arm,config,budget)
                search_new(seeds,config,budget)
            elif stage=='refit':
                for seed in seeds:
                    for arm in 'AB':refit_control(seed,arm,config,budget)
                for seed in seeds:
                    refit_optimized(seed,config,budget);training_oracle(seed,config,budget)
            elif stage=='evaluate':
                if not (ROOT/'MODEL_SELECTION_FROZEN.json').exists():
                    if a.split!='all':raise RuntimeError('All-split model sealing is required')
                    seal_models(config,'Predeclared search/refit finished or budget exhausted; missing arms remain incomplete')
                evaluate(config,budget)
            elif stage=='export':
                from prepare_workbooks import prepare
                from report_results import report
                prepare();report()
        except TimeoutError as e:
            errors.append(str(e));save_json(dict(stage=stage,reason=str(e),partial_results=True),ROOT/f'results/{stage}_budget_exhausted.json')
            print(f'BUDGET_EXHAUSTED {stage}: {e}',flush=True)
        finally:budget.close()
        save_json(dict(stage=stage,status='budget_exhausted' if errors else 'completed',errors=errors,
            updated_epoch=time.time()),ROOT/'working/progress.json')
    save_json(dict(requested_stage_finished=a.stage,full_workflow_stopped=a.stage=='all',
        delivery_QA_complete=False,updated_epoch=time.time()),ROOT/'working/run_finished.json')


if __name__=='__main__':main()
