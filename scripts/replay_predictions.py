"""Re-evaluate unchanged stored C models with the validated CPU forward adapter."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app/runtime'))
# Repeated partitions selected stored neural decoders; the primary app remains NumPy-only.
sys.path.append(str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import classical
from common import RAW, TARGETS, physical_metrics


def replay(seed, output):
    metadata = pd.read_csv(ROOT/'data/archive_199.csv')
    split = pd.read_csv(ROOT/f'data/splits/{seed}.csv').set_index('curve_id')
    frame = metadata[metadata.role!='quarantine'].copy()
    frame['split'] = split.loc[frame.curve_id, 'split'].to_numpy()
    model_path = ROOT/f'models/{seed}/C_final.joblib'
    model = joblib.load(model_path)
    assert set(model.train_ids) == set(frame[frame.split=='train'].curve_id)
    assert not set(model.train_ids) & set(frame[frame.split.isin(['val','test'])].curve_id)
    # Deliberately pass engineering inputs only, not metadata or response targets.
    aa, predictions, extras = model.predict(frame[RAW].copy())
    ref = pd.read_csv(ROOT/f'results/experiment/{seed}/C/per_curve.csv').set_index('curve_id').loc[frame.curve_id]
    np.testing.assert_allclose(aa, ref[[t+'_pred' for t in TARGETS]], rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(aa[:,2], aa[:,0]*aa[:,1]*aa[:,3], rtol=1e-12, atol=1e-8)
    stored = pd.read_csv(ROOT/f'results/experiment/{seed}/C/force_histories_long.csv')
    z = np.load(ROOT/'data/curves.npz')
    original = {c: (z['time_ms'][z['offsets'][i]:z['offsets'][i+1]], z['force_kN'][z['offsets'][i]:z['offsets'][i+1]]) for i,c in enumerate(z['curve_id'])}
    values = []
    maximum_force_error = 0.
    metric_errors = []
    for i, (row, (t,f)) in enumerate(zip(frame.itertuples(), predictions)):
        assert np.isfinite(f).all() and (f>=0).all() and (np.diff(t)>0).all()
        expected = stored[(stored.curve_id==row.curve_id)&(stored.series=='predicted')]
        np.testing.assert_allclose(t, expected.time_ms, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(f, expected.force_kN, rtol=1e-9, atol=1e-8)
        maximum_force_error = max(maximum_force_error, float(np.max(abs(f-expected.force_kN.to_numpy()))))
        m = physical_metrics(*original[row.curve_id], t, f)
        np.testing.assert_allclose(m['curve_R2'], ref.loc[row.curve_id, 'curve_R2'], rtol=1e-8, atol=1e-8)
        metric_errors.append(abs(m['curve_R2']-ref.loc[row.curve_id, 'curve_R2']))
        values.append(dict(curve_id=row.curve_id, split=row.split, **dict(zip(TARGETS, map(float,aa[i]))), **m))
    probe = frame.iloc[[0, len(frame)//2, len(frame)-1]]
    for index, point in zip([0,len(frame)//2,len(frame)-1], range(3)):
        one, hist, _ = model.predict(probe.iloc[[point]][RAW])
        np.testing.assert_allclose(one[0], aa[index], rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(hist[0][1], predictions[index][1], rtol=1e-9, atol=1e-8)
    out = output/str(seed); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(out/'replayed_predictions.csv', index=False)
    check = dict(split_seed=seed, passed=True, records=len(frame), heldout_records=40,
                 force_nonnegative=True, batch_invariant=True, input_response_fields_supplied=False,
                 maximum_anchor_abs_error=float(np.max(abs(aa-ref[[t+'_pred' for t in TARGETS]].to_numpy()))),
                 maximum_force_abs_error_kN=maximum_force_error, maximum_curve_R2_abs_error=float(max(metric_errors)),
                 model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), rtol=1e-9, atol=1e-8)
    (out/'verification.json').write_text(json.dumps(check,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(check), flush=True)
    return check


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['all','20261102','20261103','20261104'], default='all')
    p.add_argument('--output', type=Path, default=ROOT/'output/replay')
    a = p.parse_args()
    results = [replay(s, a.output) for s in ((20261102,20261103,20261104) if a.split=='all' else [int(a.split)])]
    (a.output/'verification.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
