from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['Fmax', 'Td', 'I', 'Astar', 'tp_ratio']
RAW = ['v', 'm', 'b', 'h', 'H0', 'd', 'c', 'N', 'fc_reported', 'fyl',
       'dl', 'Asl', 'fyt', 'dt', 'Asv', 's', 'corrosion_pct']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(value, path):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, default=str,allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_csv(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.16g')


def save_model(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); joblib.dump(value, tmp, compress=3)
    os.replace(tmp, path)


def load_data():
    frame = pd.read_csv(ROOT / 'data/archive_199.csv')
    a = np.load(ROOT / 'data/curves.npz')
    assert len(frame) == 199 and frame.curve_id.is_unique
    assert frame.curve_id.tolist() == a['curve_id'].tolist()
    curves = {cid: (a['time_ms'][a['offsets'][i]:a['offsets'][i+1]],
                    a['force_kN'][a['offsets'][i]:a['offsets'][i+1]])
              for i,cid in enumerate(a['curve_id'])}
    return frame, curves


def partition(seed):
    frame, curves = load_data()
    split = pd.read_csv(ROOT/f'data/splits/{seed}.csv').set_index('curve_id').loc[frame.curve_id]
    for name in ('split','nested_A'):
        frame[name] = split[name].to_numpy()
    frame['fit_eligible'] = frame.split.eq('train') & frame.role.ne('quarantine')
    return frame, curves


def group_folds(frame, seed, n=3):
    from sklearn.model_selection import GroupKFold
    groups = sorted(frame.specimen_group.unique())
    if len(groups) < n:
        raise ValueError('Too few independent specimen groups')
    shuffled = np.random.default_rng(seed).permutation(len(groups))
    labels = frame.specimen_group.map(dict(zip(groups,shuffled))).to_numpy()
    out = list(GroupKFold(n_splits=n).split(frame,groups=labels))
    for a,b in out:
        assert not set(frame.iloc[a].specimen_group)&set(frame.iloc[b].specimen_group)
    return out


def pad_curves(ids, curves):
    n = max(len(curves[c][0]) for c in ids)
    ts, fs = [], []
    for c in ids:
        t,f = curves[c]
        ts.append(np.pad(t,(0,n-len(t)),mode='edge'))
        fs.append(np.pad(f,(0,n-len(f)),mode='edge'))
    return np.asarray(ts),np.asarray(fs)


def physical_metrics(tx, fy, px, py):
    """Retain the published exact PWL union-of-supports definition."""
    xx = np.unique(np.r_[tx, px])
    mid = (xx[:-1]+xx[1:])/2
    # One-sided endpoints prevent artificial ramps after nonzero 5%-threshold ends.
    mt = (mid >= tx[0]) & (mid <= tx[-1])
    mp = (mid >= px[0]) & (mid <= px[-1])
    t0,t1 = np.interp(xx[:-1],tx,fy)*mt,np.interp(xx[1:],tx,fy)*mt
    p0,p1 = np.interp(xx[:-1],px,py)*mp,np.interp(xx[1:],px,py)*mp
    dt = np.diff(xx); a,b = p0-t0,p1-t1
    l1 = dt * (np.abs(a)+np.abs(b)) / 2
    cross = a*b < 0
    l1[cross] = dt[cross]*(a[cross]**2+b[cross]**2)/(2*(np.abs(a[cross])+np.abs(b[cross])))
    sse = np.sum(dt*(a*a+a*b+b*b)/3)
    duration = xx[-1]-xx[0]
    integral = np.sum(dt*(t0+t1)/2)
    second = np.sum(dt*(t0*t0+t0*t1+t1*t1)/3)
    sst = second-integral**2/duration
    return dict(curve_R2=float(1-sse/sst) if sst > 0 else np.nan,
                MAE_kN=float(l1.sum()/duration), RMSE_kN=float(np.sqrt(sse/duration)),
                NRMSE=float(np.sqrt(sse/duration)/max(np.max(fy), 1e-12)),
                support_union_ms=float(duration))


def anchors_from_curve(curve):
    t, f = curve
    F = float(np.max(f)); T = float(t[-1]-t[0]); I = float(np.trapezoid(f, t))
    if F<=0 or T<=0:return np.array([F,T,I,np.nan,np.nan])
    return np.array([F, T, I, I/(F*T), (t[int(np.argmax(f))]-t[0])/T])


def fingerprint():
    files = {}
    for folder in ('src', 'scripts', 'config', 'data', 'legacy', 'tests'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    for name in ('requirements.txt','requirements-server-lock.txt','requirements-base-lock.txt',
        'audit/present10_parameter_evidence.csv','audit/present10_contact_windows.csv',
        'audit/specimen_associations.csv','audit/data_audit.json'):
        if (ROOT/name).exists(): files[name]=file_hash(ROOT/name)
    return dict(sha256=digest(files), files=files)


class Budget:
    def __init__(self, stage, hours):
        self.path = ROOT / 'results' / f'budget_{stage}.json'
        self.previous = json.loads(self.path.read_text())['used_seconds'] if self.path.exists() else 0.
        self.start = time.monotonic(); self.limit = hours*3600
        self.last_write = 0.

    def check(self):
        used = self.previous+time.monotonic()-self.start
        formal = ROOT/'working/formal_start.json'
        if formal.exists():
            started = json.loads(formal.read_text())['epoch']
            if time.time()-started >= 48*3600:
                raise TimeoutError('Total 48-hour wall-clock budget exhausted')
        if used-self.last_write > 5 or used >= self.limit:
            save_json(dict(used_seconds=used, limit_seconds=self.limit), self.path)
            self.last_write = used
        if used >= self.limit:
            raise TimeoutError('Stage budget exhausted; partial results retained, not complete.')

    def close(self):
        save_json(dict(used_seconds=self.previous+time.monotonic()-self.start,
                       limit_seconds=self.limit),self.path)
