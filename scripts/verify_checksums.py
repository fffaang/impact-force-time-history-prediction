"""Check the content manifest; Git metadata and generated outputs are outside its scope."""
import csv
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify():
    count=0
    with (ROOT/'SHA256SUMS.csv').open(encoding='utf-8',newline='') as f:
        for item in csv.DictReader(f):
            path=ROOT/item['path']
            h=hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=item['sha256']:raise RuntimeError('Checksum mismatch: '+item['path'])
            count+=1
    print('Verified',count,'files')
    return count


if __name__=='__main__':verify()
