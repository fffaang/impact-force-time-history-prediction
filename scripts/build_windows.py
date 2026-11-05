"""Build the primary CPU application in a separate Windows staging directory."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'output/windows_build')
    args=p.parse_args()
    if os.name!='nt':p.error('Build on Windows x64 with Python 3.12')
    output=args.output.resolve();source=output/'source'
    if source.exists():p.error('Use a fresh output directory')
    shutil.copytree(ROOT/'app',source,ignore=shutil.ignore_patterns('__pycache__','.runtime','*.pyc'))
    shutil.copy2(ROOT/'models/20261102/C_final.joblib',source/'models/C_final.joblib')
    manifest_path=source/'models/manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.pop('model_path',None)
    for item in manifest['files']:
        if item['path'].startswith('../models/'):item['path']='models/C_final.joblib'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    command=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onedir','--windowed',
             '--name','ImpactCurve','--distpath',str(output/'dist'),'--workpath',str(output/'build'),
             '--specpath',str(output/'build'),'--paths',str(source/'runtime')]
    for directory in ('models','data','static','runtime'):
        command+=['--add-data',str(source/directory)+os.pathsep+directory]
    for name in ('anchors','features','shapes','common','classical'):command+=['--hidden-import',name]
    for name in ('torch','matplotlib','plotly','catboost','xgboost','lightgbm','tkinter','pytest','IPython','numba'):
        command+=['--exclude-module',name]
    command.append(str(source/'app.py'))
    subprocess.run(command,cwd=source,check=True)
    shutil.copytree(ROOT/'third_party/licenses',output/'dist/licenses')
    for name in ('LICENSE','DATA_LICENSE.md','THIRD_PARTY_NOTICES.md'):shutil.copy2(ROOT/name,output/'dist'/name)
    print('Built',output/'dist/ImpactCurve/ImpactCurve.exe')


if __name__=='__main__':main()
