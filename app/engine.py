"""Validated, inference-only interface to the saved primary model C."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT/'runtime'))
import joblib
import numpy as np
import pandas as pd
# Explicit imports also make the trusted pickle classes visible to the bundler.
import anchors
import shapes
import classical
from common import RAW

FIELDS = [
    dict(key='v', label='Impact velocity', symbol='v', unit='m/s', group='impact', required=True),
    dict(key='m', label='Impactor mass', symbol='m', unit='kg', group='impact', required=True),
    dict(key='N', label='Axial compression', symbol='N', unit='kN', group='impact'),
    dict(key='b', label='Section width', symbol='b', unit='mm', group='geometry', required=True),
    dict(key='h', label='Impact-direction depth', symbol='h', unit='mm', group='geometry', required=True),
    dict(key='H0', label='Member height / length', symbol='H<sub>0</sub>', unit='mm', group='geometry'),
    dict(key='d', label='Effective depth', symbol='d', unit='mm', group='geometry'),
    dict(key='c', label='Concrete cover', symbol='c', unit='mm', group='geometry'),
    dict(key='fc_reported', label='Concrete strength', symbol="f′<sub>c</sub>", unit='MPa', group='material', required=True),
    dict(key='fyl', label='Longitudinal steel yield', symbol='f<sub>yl</sub>', unit='MPa', group='material'),
    dict(key='fyt', label='Stirrup steel yield', symbol='f<sub>yt</sub>', unit='MPa', group='material'),
    dict(key='dl', label='Longitudinal bar diameter', symbol='d<sub>l</sub>', unit='mm', group='reinforcement'),
    dict(key='Asl', label='Total longitudinal area', symbol='A<sub>sl</sub>', unit='mm²', group='reinforcement'),
    dict(key='dt', label='Stirrup diameter', symbol='d<sub>t</sub>', unit='mm', group='reinforcement'),
    dict(key='Asv', label='Effective stirrup area', symbol='A<sub>sv</sub>', unit='mm²', group='reinforcement'),
    dict(key='s', label='Stirrup spacing', symbol='s', unit='mm', group='reinforcement'),
    dict(key='corrosion_pct', label='Corrosion ratio', symbol='η<sub>cor</sub>', unit='%', group='reinforcement'),
]
LABELS = {f['key']: f['label'] for f in FIELDS}
TARGETS = ['Fmax', 'Td', 'I', 'Astar', 'tp_ratio']


class InputError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or []


def csv_text(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow(header)
    for row in rows:
        safe = ["'"+x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x for x in row]
        writer.writerow(safe)
    return stream.getvalue()


class Predictor:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'models/manifest.json').read_text(encoding='utf-8'))
        self.examples = json.loads((self.root/'data/examples.json').read_text(encoding='utf-8'))
        self.ranges = json.loads((self.root/'data/training_ranges.json').read_text(encoding='utf-8'))
        self.lock = threading.RLock()
        for item in self.manifest['files']:
            actual = hashlib.sha256((self.root/item['path']).read_bytes()).hexdigest()
            if actual != item['sha256']:
                raise RuntimeError('Model package integrity check failed: '+item['path'])
        import sklearn
        if sklearn.__version__ != self.manifest['sklearn_version']:
            raise RuntimeError('This frozen model requires scikit-learn '+self.manifest['sklearn_version'])
        model_path = self.manifest.get('model_path', 'models/C_final.joblib')
        self.model = joblib.load(self.root/model_path)
        if self.model.shape.spec != dict(structure='impulse', features='physics', kind='PCA', reg='ExtraTrees', k=8):
            raise RuntimeError('Unexpected model structure; refused to substitute another model')

    def metadata(self):
        return dict(version=self.manifest['version'], model='C · 20261102',
                    model_sha256=self.manifest['source_model_sha256'], fields=FIELDS,
                    ranges=self.ranges, examples=self.examples, max_batch=100,
                    structure=dict(independent=['Fmax','Td','I','tp_ratio'], derived=['Astar'], shape='PCA8 + ExtraTrees'),
                    performance=dict(validation=dict(Fmax=.708100,Td=.875170,I=.888084,Astar=-.524023,tp_ratio=.801862,median_curve_R2=.420561,negative_curves=4,n=20),
                                     test=dict(Fmax=.923739,Td=.814012,I=.983894,Astar=.218431,tp_ratio=.819957,median_curve_R2=.515544,negative_curves=6,n=20)))

    def validate(self, values):
        if not isinstance(values, dict):
            raise InputError('Inputs must be a parameter object.')
        unknown = set(values)-set(RAW)
        if unknown:
            raise InputError('Unknown parameter fields.', [dict(field=k,message=k+' is not a model input.') for k in sorted(unknown)])
        result, details = {}, []
        for field in FIELDS:
            key = field['key']; raw = values.get(key)
            if raw is None or raw == '':
                if field.get('required'): details.append(dict(field=key,message=field['label']+' is required.'))
                result[key] = None
                continue
            try:
                if isinstance(raw, bool): raise ValueError()
                number = float(raw)
                if not math.isfinite(number): raise ValueError()
                if number < 0: raise ValueError()
                if field.get('required') and number <= 0: raise ValueError()
                if key in ['H0','d'] and number <= 0: raise ValueError()
                if key == 'corrosion_pct' and number > 100: raise ValueError()
                result[key] = number
            except (TypeError, ValueError, OverflowError):
                details.append(dict(field=key,message='Invalid '+field['label'].lower()+': check the value and unit.'))
        if details: raise InputError('Check the highlighted parameters.', details)
        return {key: result[key] for key in RAW}

    def warnings(self, values):
        missing, outside, result = [], [], []
        for key, v in values.items():
            if v is None: missing.append(key)
            elif v < self.ranges[key]['min'] or v > self.ranges[key]['max']: outside.append(key)
        if missing:
            result.append(dict(code='missing', level='notice', fields=missing,
                               message='Training-fitted imputation used for: '+', '.join(LABELS[k].lower() for k in missing)+'.'))
        if outside:
            result.append(dict(code='outside_range', level='warning', fields=outside,
                               message='Outside the univariate training range: '+', '.join(LABELS[k].lower() for k in outside)+'.'))
        if values['d'] is not None and values['d'] >= values['h']:
            result.append(dict(code='effective_depth',level='warning',fields=['d','h'],message='Effective depth d is not smaller than section depth h. Check the impact direction and dimension definitions.'))
        if values['c'] is not None and 2*values['c'] >= min(values['b'],values['h']):
            result.append(dict(code='cover',level='warning',fields=['c'],message='Concrete cover is inconsistent with the section dimensions. Check the input.'))
        if values['s'] == 0 and not (values['dt'] == 0 and values['Asv'] == 0):
            result.append(dict(code='stirrups',level='warning',fields=['s','dt','Asv'],message='Zero stirrup spacing is treated as an invalid denominator. If stirrups are absent, confirm that both their diameter and area are zero.'))
        return result

    def predict_many(self, cases):
        if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
            raise InputError('Each batch must contain 1–100 cases.')
        validated = []
        for i, case in enumerate(cases):
            if not isinstance(case, dict): raise InputError('Invalid case format.')
            name = case.get('name') or 'Case '+str(i+1)
            if not isinstance(name, str) or len(name)>100: raise InputError('Case names must be text of at most 100 characters.')
            values = self.validate(case.get('inputs'))
            validated.append((name, values))
        frame = pd.DataFrame([v for _,v in validated], columns=RAW).astype(float)
        started = time.perf_counter()
        with self.lock:
            aa, curves, extras = self.model.predict(frame)
        elapsed = time.perf_counter()-started
        output = []
        for index, ((name, inputs), anchors_, (t, force)) in enumerate(zip(validated, aa, curves)):
            if not np.all(np.isfinite(anchors_)) or not np.all(np.isfinite(force)) or not np.all(np.diff(t)>0):
                raise RuntimeError('Model returned a non-finite or unordered prediction')
            if np.min(force)<0: raise RuntimeError('Model returned negative force')
            integral = float(np.trapezoid(force,t))
            peak_index = int(np.argmax(force))
            reconstructed = dict(Fmax=float(np.max(force)), Td=float(t[-1]-t[0]), I=integral,
                                 Astar=float(integral/(max(force)*(t[-1]-t[0]))),
                                 tp_ratio=float((t[peak_index]-t[0])/(t[-1]-t[0])))
            csv_data = csv_text(['time_ms','time_s','force_kN','normalized_time','normalized_force'],
                zip(t, t/1000, force, t/anchors_[1], force/anchors_[0]))
            output.append(dict(name=name,inputs=inputs, anchors=dict(zip(TARGETS,map(float,anchors_))),
                curve=dict(time_ms=t.tolist(),force_kN=force.tolist()), reconstructed=reconstructed,
                diagnostics=dict(area_error=float(extras['area_error'][index]),
                                 projection_change=float(extras['projection_change'][index]),
                                 anchor_log_projection=float(extras['consistency_projection_log_delta'][index])),
                warnings=self.warnings(inputs), curve_csv=csv_data,
                model=dict(version=self.manifest['version'],partition=20261102,sha256=self.manifest['source_model_sha256']),
                elapsed_ms=round(1000*elapsed,1)))
        return output

    def predict(self, case):
        return self.predict_many([case])[0]

    def template_csv(self):
        return csv_text(['case_id']+RAW, [[e['name']]+[e['inputs'][k] for k in RAW] for e in self.examples])

    def batch_csv(self, text):
        reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
        if not reader.fieldnames or not set(RAW).issubset(reader.fieldnames):
            raise InputError('CSV columns are incomplete. Use the batch template.')
        if len(set(reader.fieldnames)) != len(reader.fieldnames): raise InputError('CSV contains duplicate column names.')
        if set(reader.fieldnames)-set(RAW)-{'case_id'}: raise InputError('CSV contains unknown parameter columns. Use the batch template.')
        cases, errors = [], []
        for i, row in enumerate(reader, 2):
            if i>101: raise InputError('A batch can contain at most 100 cases.')
            if None in row: raise InputError('CSV rows have inconsistent column counts.')
            try:
                inputs = self.validate({k:row.get(k) for k in RAW})
                cases.append(dict(name=row.get('case_id') or 'Case '+str(i-1), inputs=inputs))
            except InputError as e:
                errors.append(dict(row=i,message=str(e),details=e.details))
        if errors: raise InputError('Batch validation failed. No cases have been skipped or evaluated.',errors)
        if len({c['name'] for c in cases}) != len(cases): raise InputError('Duplicate case names. Assign a unique case_id to each row.')
        results = self.predict_many(cases)
        summary = csv_text(['case_id']+TARGETS+['warning_count'],
            [[r['name']]+[r['anchors'][k] for k in TARGETS]+[len(r['warnings'])] for r in results])
        long = csv_text(['case_id','time_ms','time_s','force_kN'],
            ([r['name'],t,t/1000,f] for r in results for t,f in zip(r['curve']['time_ms'],r['curve']['force_kN'])))
        return dict(results=results,summary_csv=summary,curves_csv=long)
