"""CPU-only inference compatibility for the frozen classical model C.

The pickle classes and stored estimators are unchanged. Only the float64
reconstruction forward pass is evaluated with NumPy instead of PyTorch.
"""
import numpy as np
from scipy.special import expit, logit

AREA_EPS = 1e-5


def consistent_anchors(base, structure):
    base = np.asarray(base, dtype=np.float64)
    logf = np.log(np.maximum(base[:, 0], 1e-8))
    logt = np.log(np.maximum(base[:, 1], 1e-8))
    phase = expit(logit(np.clip(base[:, 4], AREA_EPS, 1-AREA_EPS)))
    if structure == 'impulse':
        logi = np.log(np.maximum(base[:, 2], 1e-8))
        ratio = logi - logf - logt
        delta = ratio - np.clip(ratio, np.log(AREA_EPS), np.log(1-AREA_EPS))
        f, t, impulse = np.exp(logf+delta/3), np.exp(logt+delta/3), np.exp(logi-delta/3)
        area = impulse/(f*t)
    elif structure == 'fullness':
        area = np.clip(expit(logit(np.clip(base[:, 3], AREA_EPS, 1-AREA_EPS))), AREA_EPS, 1-AREA_EPS)
        f, t = np.exp(logf), np.exp(logt)
        impulse = f*t*area
        delta = np.zeros_like(f)
    else:
        raise ValueError('Unsupported frozen anchor structure')
    return np.column_stack((f, t, impulse, area, phase)), delta


def reconstruct(raw, anchors, projection='power'):
    f, t, impulse, area, phase = anchors.T
    n, points = raw.shape
    delta = np.minimum(np.minimum(area/20, (1-area)/20), np.minimum(phase/4, (1-phase)/4))
    delta = np.minimum(delta, .0002)
    grid = np.broadcast_to(np.linspace(0, 1, 201), (n, 201))
    extra = np.column_stack((delta, phase-delta, phase, phase+delta, 1-delta))
    tau = np.sort(np.column_stack((grid, extra)), axis=1, kind='stable')
    canonical = np.where(tau <= phase[:, None], .5*tau/phase[:, None],
                         .5+.5*(tau-phase[:, None])/(1-phase[:, None]))
    u = np.linspace(0, 1, points)
    base = np.array([np.interp(q, u, y, left=0, right=0) for q, y in zip(canonical, raw)])
    base = np.clip(base, 1e-6, 1-1e-6)
    base = np.where(tau == phase[:, None], 1., base)
    base = np.where((tau == 0) | (tau == 1), 0., base)
    dt = np.diff(tau, axis=1)
    weights = np.column_stack((dt[:, :1]/2, (dt[:, :-1]+dt[:, 1:])/2, dt[:, -1:]/2))
    if projection != 'power':
        raise ValueError('Only the frozen production power projection is enabled')
    logs = np.log(np.maximum(base, 1e-30))
    lo, hi = np.full(n, np.log(1e-10)), np.full(n, np.log(1e9))
    for _ in range(64):
        mid = (lo+hi)/2
        value = np.where(base == 0, 0., np.exp(logs*np.exp(mid)[:, None]))
        above = np.sum(weights*value, axis=1) > area
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    gamma = np.exp((lo+hi)/2)
    y = np.where(base == 0, 0., np.exp(logs*gamma[:, None]))
    return tau*t[:, None], y*f[:, None], dict(
        area_error=np.sum(weights*y, axis=1)-area,
        projection_change=np.sum(weights*np.abs(y-base), axis=1),
        area_before=np.sum(weights*base, axis=1))


class ClassicalModel:
    def predict(self, frame, base_anchors, seed_index=None, projection='power', **kwargs):
        if kwargs:
            raise ValueError('Oracle or training-only arguments are not available in the prediction tool')
        anchors, delta = consistent_anchors(base_anchors, self.spec['structure'])
        raw = self.shape.predict(frame, anchors, seed_index)
        raw = np.maximum(raw, 1e-6)
        raw /= np.maximum(raw.max(axis=1, keepdims=True), 1e-8)
        raw[:, [0, -1]] = 0
        raw[:, raw.shape[1]//2] = 1
        time, force, info = reconstruct(raw, anchors, projection)
        curves = []
        for x, y in zip(time, force):
            unique, index = np.unique(x, return_index=True)
            curves.append((unique, y[index]))
        return anchors, curves, info | {'canonical_raw': raw, 'consistency_projection_log_delta': delta}


class OptimizedModel:
    def predict(self, frame, seed_index=None):
        from anchors import predict_anchors
        return self.shape.predict(frame, predict_anchors(self.anchors, frame, seed_index), seed_index=seed_index)
