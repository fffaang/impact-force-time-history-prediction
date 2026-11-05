"""Restore byte-identical checkpoint aliases after extracting Release model archives."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    items=json.loads((ROOT/'models/release_aliases.json').read_text())
    restored=0;unavailable=[]
    for item in items:
        target=(ROOT/item['path']).resolve();source=(ROOT/item['canonical_path']).resolve()
        if not target.is_relative_to(ROOT/'models') or not source.is_relative_to(ROOT/'models'):
            raise ValueError('Checkpoint path outside models directory')
        if not source.exists():unavailable.append(item['path']);continue
        if sha(source)!=item['sha256']:raise RuntimeError('Source checksum mismatch: '+item['canonical_path'])
        if target.exists():
            if sha(target)!=item['sha256']:raise RuntimeError('Refusing to replace a different model: '+item['path'])
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);restored+=1
    print('Restored',restored,'aliases;',len(unavailable),'checkpoints require their Release archive.')


if __name__=='__main__':main()
