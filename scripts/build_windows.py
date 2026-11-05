"""Build the primary CPU application in a separate Windows staging directory."""
from pathlib import Path
import argparse
import ast
import hashlib
from importlib import metadata
import json
import os
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def verify_runtime():
    lock=json.loads((ROOT/'provenance/windows_runtime.json').read_text(encoding='utf-8'))
    if sys.version_info[:3]!=(3,12,14) or sys.maxsize<=2**32:
        raise RuntimeError('Use the locked CPython 3.12.14 Windows x64 runtime')
    if sys.prefix==sys.base_prefix:
        raise RuntimeError('Build in a fresh virtual environment, without system packages')
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'):
        raise RuntimeError('Unset PYTHONPATH and PYTHONHOME before building')
    base=Path(sys.base_prefix)
    for relative,expected in lock['runtime_file_sha256'].items():
        path=base/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Runtime provenance mismatch: '+relative)
    normalize=lambda value:value.lower().replace('_','-').replace('.','-')
    expected={normalize(w['name']):w['version'] for w in lock['wheels']}
    installed={normalize(d.metadata['Name']):d.version for d in metadata.distributions()}
    if any(installed.get(name)!=version for name,version in expected.items()):
        raise RuntimeError('Install the complete requirements-build-windows.lock')
    if set(installed)-set(expected)-{'pip'}:
        raise RuntimeError('Unexpected packages found; use an isolated build environment')
    return lock


def verify_native_sources(output):
    entries=ast.literal_eval((output/'build/ImpactCurve/COLLECT-00.toc').read_text(encoding='utf-8'))[0]
    base=Path(sys.base_prefix).resolve()
    environment=Path(sys.prefix).resolve()
    system=(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32').resolve()
    system_files=[]
    for name,original,kind in entries:
        if kind!='BINARY':continue
        source=Path(original).resolve()
        if not any(source.is_relative_to(parent) for parent in (base,environment,system)):
            raise RuntimeError('Native dependency outside the isolated environment: '+name)
        if source.is_relative_to(system):
            system_files.append({'file':name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'origin':'Microsoft Windows System32'})
    return system_files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'output/windows_build')
    args=p.parse_args()
    if os.name!='nt':p.error('Build on Windows x64 with Python 3.12')
    lock=verify_runtime()
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
             '--specpath',str(output/'build'),'--paths',str(source/'runtime'),
             '--version-file',str(ROOT/'app/windows_version_info.txt')]
    for directory in ('models','data','static','runtime'):
        command+=['--add-data',str(source/directory)+os.pathsep+directory]
    for name in ('anchors','features','shapes','common','classical'):command+=['--hidden-import',name]
    for name in ('torch','matplotlib','plotly','catboost','xgboost','lightgbm','tkinter','pytest','IPython','numba'):
        command+=['--exclude-module',name]
    command.append(str(source/'app.py'))
    build_env=os.environ.copy()
    system=Path(build_env.get('SystemRoot',r'C:\Windows'))
    build_env['PATH']=os.pathsep.join(map(str,[Path(sys.executable).parent,Path(sys.base_prefix),
        Path(sys.base_prefix)/'DLLs',system/'System32',system]))
    build_env['PYTHONNOUSERSITE']='1'
    subprocess.run(command,cwd=source,env=build_env,check=True)
    system_files=verify_native_sources(output)
    shutil.copytree(ROOT/'third_party/licenses',output/'dist/licenses')
    for name in ('LICENSE','DATA_LICENSE.md','THIRD_PARTY_NOTICES.md'):shutil.copy2(ROOT/name,output/'dist'/name)
    (output/'dist/build_provenance.json').write_text(json.dumps({
        'application_version':manifest['version'],
        'runtime_archive_sha256':lock['sha256'],
        'runtime_archive_url':lock['archive_url'],
        'dependency_lock_sha256':hashlib.sha256((ROOT/'requirements-build-windows.lock').read_bytes()).hexdigest(),
        'system_dependencies':system_files,
        'model_sha256':manifest['source_model_sha256']},indent=2)+'\n',encoding='utf-8')
    print('Built',output/'dist/ImpactCurve/ImpactCurve.exe')


if __name__=='__main__':main()
