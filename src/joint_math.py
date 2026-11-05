"""Differentiable reconstruction and exact physical-time PWL evaluation."""
from __future__ import annotations

import numpy as np
import torch

AREA_EPS = 1e-5


class GatherRows(torch.autograd.Function):
    """Fixed-size deterministic reduction avoids CUDA scatter's host sync."""
    @staticmethod
    def forward(ctx, values, index):
        ctx.save_for_backward(index)
        ctx.columns = values.shape[1]
        return values.gather(1, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        columns = torch.arange(ctx.columns, device=index.device)
        matrix = (index[:, :, None] == columns).to(grad.dtype)
        return torch.bmm(matrix.transpose(1, 2), grad[:, :, None]).squeeze(2), None


class SortRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        result = torch.sort(values, dim=1, stable=True)
        ctx.save_for_backward(result.indices)
        return result.values

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        inverse = torch.argsort(index, dim=1)
        return grad.gather(1, inverse)


def interp_batch(query, xp, yp):
    """Linear interpolation with zero extension; knots can repeat."""
    ix = torch.searchsorted(xp.contiguous(), query.contiguous(), right=True)
    lo = (ix - 1).clamp(0, xp.shape[1] - 2)
    hi = lo + 1
    x0, x1 = GatherRows.apply(xp, lo), GatherRows.apply(xp, hi)
    y0, y1 = GatherRows.apply(yp, lo), GatherRows.apply(yp, hi)
    a = (query - x0) / (x1 - x0).clamp_min(torch.finfo(xp.dtype).eps)
    out = y0 + a * (y1 - y0)
    return torch.where((query < xp[:, :1]) | (query > xp[:, -1:]), 0., out)


def consistent_anchors(base, structure, correction=None):
    """Four production outputs; fifth derived, with an explicit log projection."""
    if correction is None:
        correction = torch.zeros((len(base), 4), device=base.device, dtype=base.dtype)
    logf = torch.log(base[:, 0].clamp_min(1e-8)) + np.log(1.1) * torch.tanh(correction[:, 0])
    logt = torch.log(base[:, 1].clamp_min(1e-8)) + np.log(1.1) * torch.tanh(correction[:, 1])
    phase = torch.sigmoid(torch.logit(base[:, 4].clamp(AREA_EPS, 1-AREA_EPS))
                          + .2 * torch.tanh(correction[:, 3]))
    if structure == 'impulse':
        logi = torch.log(base[:, 2].clamp_min(1e-8)) + np.log(1.1) * torch.tanh(correction[:, 2])
        ratio = logi - logf - logt
        delta = ratio - ratio.clamp(np.log(AREA_EPS), np.log(1-AREA_EPS))
        # Euclidean projection in three log scales onto the feasible area slab.
        logf, logt, logi = logf + delta/3, logt + delta/3, logi - delta/3
        F, T, I = logf.exp(), logt.exp(), logi.exp()
        A = I / (F*T)
    elif structure == 'fullness':
        A = torch.sigmoid(torch.logit(base[:, 3].clamp(AREA_EPS, 1-AREA_EPS))
                          + .2 * torch.tanh(correction[:, 2]))
        A = A.clamp(AREA_EPS, 1-AREA_EPS)
        F, T = logf.exp(), logt.exp()
        I = F*T*A
        delta = torch.zeros_like(F)
    else:
        raise ValueError(structure)
    return torch.stack((F, T, I, A, phase), 1), delta


class AreaPower(torch.autograd.Function):
    """Implicit derivative of sum(w * base**gamma) = requested area."""
    @staticmethod
    def forward(ctx, base, weights, area):
        logs = base.clamp_min(1e-30).log()
        # Log brackets cover both nearly flat and very sharp feasible pulses.
        lo, hi = torch.full_like(area, np.log(1e-10)), torch.full_like(area, np.log(1e9))
        iterations = 64 if base.dtype == torch.float64 else 44
        for _ in range(iterations):
            mid = (lo+hi)/2
            value = torch.exp(logs*mid.exp()[:, None])
            value = torch.where(base == 0, 0., value)
            above = (weights*value).sum(1) > area
            lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
        gamma = ((lo+hi)/2).exp()
        y = torch.exp(logs*gamma[:, None])
        y = torch.where(base == 0, 0., y)
        ctx.save_for_backward(base, weights, logs, gamma, y)
        return y

    @staticmethod
    def backward(ctx, grad):
        base, weights, logs, gamma, y = ctx.saved_tensors
        dgamma = y*logs
        denom = (weights*dgamma).sum(1).clamp_max(-1e-20)
        factor = (grad*dgamma).sum(1)/denom
        direct = gamma[:, None]*y/base.clamp_min(1e-30)
        direct = torch.where(base > 0, direct, 0.)
        return (direct*(grad-factor[:, None]*weights),
                -factor[:, None]*y, factor)


def reconstruct(raw, anchors, projection='power'):
    F, T, I, A, p = anchors.unbind(1)
    n, points = raw.shape
    delta = torch.minimum(torch.minimum(A/20, (1-A)/20),
                          torch.minimum(p/4, (1-p)/4)).clamp_max(.0002)
    grid = torch.linspace(0, 1, 201, dtype=raw.dtype, device=raw.device).expand(n, -1)
    tau = SortRows.apply(torch.cat((grid, torch.stack((delta, p-delta, p, p+delta, 1-delta), 1)), 1))
    canonical = torch.where(tau <= p[:, None], .5*tau/p[:, None],
                            .5+.5*(tau-p[:, None])/(1-p[:, None]))
    u = torch.linspace(0, 1, points, dtype=raw.dtype, device=raw.device).expand(n, -1)
    base = interp_batch(canonical, u, raw).clamp(1e-6, 1-1e-6)
    base = torch.where(tau == p[:, None], 1., base)
    base = torch.where((tau == 0) | (tau == 1), 0., base)
    dt = tau[:, 1:]-tau[:, :-1]
    w = torch.cat((dt[:, :1]/2, (dt[:, :-1]+dt[:, 1:])/2, dt[:, -1:]/2), 1)
    if projection == 'power':
        y = AreaPower.apply(base, w, A)
    elif projection == 'shift':
        low = torch.where(tau == p[:, None], 1., 0.)
        high = torch.where((tau == 0) | (tau == 1), 0., 1.)
        lo, hi = torch.full_like(A, -2.), torch.full_like(A, 2.)
        for _ in range(50):
            mid = (lo+hi)/2
            v = torch.maximum(low, torch.minimum(high, base+mid[:, None]))
            below = (w*v).sum(1) < A
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        y = torch.maximum(low, torch.minimum(high, base+((lo+hi)/2)[:, None]))
    elif projection == 'none':
        y = base
    else:
        raise ValueError(projection)
    area = (w*y).sum(1)
    return tau*T[:, None], y*F[:, None], {
        'area_error': area-A,
        'projection_change': (w*(y-base).abs()).sum(1),
        'area_before': (w*base).sum(1),
    }


def pwl_metrics(true_t, true_f, pred_t, pred_f):
    """Same exact integral and support union as common.physical_metrics."""
    grid = SortRows.apply(torch.cat((true_t, pred_t), 1))
    scale = true_f.max(1).values.clamp_min(1e-8)
    dt = grid[:, 1:]-grid[:, :-1]
    middle = (grid[:, 1:]+grid[:, :-1])/2
    mt = ((middle >= true_t[:, :1]) & (middle <= true_t[:, -1:])).to(grid.dtype)
    mp = ((middle >= pred_t[:, :1]) & (middle <= pred_t[:, -1:])).to(grid.dtype)
    true = interp_batch(grid, true_t, true_f)/scale[:, None]
    pred = interp_batch(grid, pred_t, pred_f)/scale[:, None]
    t0,t1 = true[:, :-1]*mt,true[:, 1:]*mt
    p0,p1 = pred[:, :-1]*mp,pred[:, 1:]*mp
    a,b = p0-t0,p1-t1
    sse = (dt*(a*a+a*b+b*b)/3).sum(1)
    span = grid[:, -1]-grid[:, 0]
    integral = (dt*(t0+t1)/2).sum(1)
    second = (dt*(t0*t0+t0*t1+t1*t1)/3).sum(1)
    sst = second-integral.square()/span
    ratio = sse/sst.clamp_min(1e-12)
    return ratio, torch.sqrt(sse/span), sst


def anchor_coordinates(anchors):
    return torch.cat((anchors[:, :3].clamp_min(1e-8).log(),
                      torch.logit(anchors[:, 3:].clamp(AREA_EPS, 1-AREA_EPS))), 1)
