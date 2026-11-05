"""Verify numerical integrity, units, fixed indices, and processed/plotting separation."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import TARGETS, anchors_from_curve


def verify():
    frame = pd.read_csv(ROOT/'data/archive_199.csv', float_precision='round_trip')
    z = np.load(ROOT/'data/curves.npz')
    assert len(frame) == 199 and frame.curve_id.is_unique
    assert len(z['offsets']) == 200 and len(z['time_ms']) == len(z['force_kN']) == 38403
    assert frame.curve_id.tolist() == z['curve_id'].tolist()
    assert frame.role.value_counts().to_dict() == dict(primary=169, numerical_aux=18, shape_aux=11, quarantine=1)
    mapping = pd.read_csv(ROOT/'data/source_mapping.csv')
    assert mapping.curve_id.is_unique and set(mapping.curve_id) == set(frame.curve_id)
    long = pd.read_csv(ROOT/'data/force_histories_long.csv', float_precision='round_trip')
    assert len(long) == 38403
    for i, row in enumerate(frame.itertuples()):
        a, b = z['offsets'][i:i+2]
        t, f = z['time_ms'][a:b], z['force_kN'][a:b]
        assert np.isfinite(t).all() and np.isfinite(f).all() and (np.diff(t)>0).all()
        for part in (long[long.curve_id==row.curve_id], pd.read_csv(ROOT/'data/force_histories'/f'{row.curve_id}.csv', float_precision='round_trip')):
            np.testing.assert_array_equal(part.time_ms, t)
            np.testing.assert_array_equal(part.force_kN, f)
        np.testing.assert_allclose(anchors_from_curve((t, f)), [getattr(row, x) for x in TARGETS], rtol=2e-11)
    own = pd.read_csv(ROOT/'data/present10_origin_long.csv')
    assert len(own) == 5000 and len(own.curve_id.unique()) == 10
    assert own.groupby('curve_id').size().eq(500).all()
    assert set(own.curve_id) == set(frame[frame.curve_id.str.startswith('PRESENT_')].curve_id)
    counts = {}
    for seed in (20261102, 20261103, 20261104):
        split = pd.read_csv(ROOT/f'data/splits/{seed}.csv')
        assert len(split) == 199 and split.curve_id.is_unique
        joined = frame[['curve_id','role']].merge(split.drop(columns='role'), on='curve_id', validate='one_to_one')
        usable = joined[joined.role!='quarantine']
        assert usable.split.value_counts().to_dict() == dict(train=158, val=20, test=20)
        assert usable.groupby('specimen_group').split.nunique().max() == 1
        assert usable[usable.nested_A].split.eq('train').all() and int(usable.nested_A.sum()) == 131
        excluded = joined[joined.role=='quarantine']
        assert len(excluded) == 1 and not excluded.split.isin(['train','val','test']).any()
        counts[str(seed)] = usable.split.value_counts().to_dict()
    return dict(passed=True, archive_records=199, eligible_records=198, model_ready_points=38403,
                origin_plotting_points=5000, origin_histories=10, partitions=counts,
                csv_npz_exact=True, literature_pointwise_reverification=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path)
    args = p.parse_args(); result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
