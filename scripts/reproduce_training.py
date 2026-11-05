"""Explicit fresh-workspace reproduction; never fit inside the published snapshot."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true',help='Training-only short compatibility test')
    mode.add_argument('--run',action='store_true',help='Explicit full search, refit and evaluation, up to 48 hours')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();destination=a.workdir.resolve()
    if destination==ROOT or ROOT in destination.parents:
        p.error('Use a new directory outside the repository')
    if destination.exists() and any(destination.iterdir()):
        p.error('Fresh reproduction requires an empty directory; archived results are never overwritten')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('src','scripts','config','data','legacy','tests'):
        shutil.copytree(ROOT/name,destination/name,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('requirements.txt','requirements-inference.txt','requirements-base-lock.txt','requirements-server-lock.txt'):
        shutil.copy2(ROOT/name,destination/name)
    for name in ('audit','reports','working','results'):(destination/name).mkdir(exist_ok=True)
    # These are inherited evidence, not scientific results of the fresh run.
    for path in (ROOT/'data/audit').glob('*'):
        if path.is_file():shutil.copy2(path,destination/'audit'/path.name)
    import yaml
    cfg_path=destination/'config/split811_20261102.yml'
    cfg=yaml.safe_load(cfg_path.read_text());cfg['device']=a.device;cfg['cuda_graph']=a.device=='cuda'
    cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def run(*args):subprocess.run([sys.executable,*args],cwd=destination,env=env,check=True)
    run('-m','unittest','discover','-s','tests','-p','test_core.py')
    run('scripts/run_experiment.py','--stage','audit')
    if a.smoke:
        run('scripts/run_experiment.py','--smoke','--device',a.device)
        print('Training-only smoke passed; no validation/test accuracy was estimated.')
        return
    # Record the new portable source freeze, distinct from provenance/original_freeze.json.
    run('-c',"import sys,json;sys.path.insert(0,'src');from common import fingerprint,save_json;save_json(fingerprint(),'FROZEN.json')")
    if a.device=='cuda':
        run('-c',"import torch;assert torch.cuda.is_available();x=torch.ones((32,32),device='cuda');assert float((x@x).mean())==32.;print(torch.cuda.get_device_name())")
    (destination/'working/formal_start.json').write_text(
        json.dumps(dict(epoch=time.time(),total_hours=48,mode='fresh_public_reproduction'))+'\n',encoding='utf-8')
    run('scripts/run_experiment.py','--stage','all','--split','all')
    print('Run stopped. Inspect acceptance.json and any budget-exhaustion records; completion is not guaranteed by process exit.')


if __name__=='__main__':main()
