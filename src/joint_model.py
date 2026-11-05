"""Fold-local templates, PCA residuals and joint physical-curve fitting."""
from __future__ import annotations

import copy
import itertools
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA

from common import TARGETS, anchors_from_curve, pad_curves
from features import feature_matrix, preprocessing
from shapes import aligned
from joint_math import consistent_anchors, reconstruct, pwl_metrics, anchor_coordinates


def specifications(config):
    return [dict(structure=s, features=f, template=t, k=k, width=w,
                 calibration=c, anchor_weight=config['anchor_loss_weight'])
            for s, f, t, k, w, c in itertools.product(config['structures'], config['feature_sets'],
                config['templates'], config['dimensions'], config['widths'], config['calibration_modes'])]


def engineering_inputs(frame, anchors, structure, features):
    aa, _ = consistent_anchors(torch.as_tensor(anchors, dtype=torch.float64), structure)
    coords = anchor_coordinates(aa).numpy()
    return np.column_stack((feature_matrix(frame, features), coords))


class Representation:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config

    def fit(self, frame, anchors, curves):
        assert frame.fit_eligible.all() and frame.split.eq('train').all()
        self.train_ids = frame.curve_id.tolist()
        self.train_groups = frame.specimen_group.to_numpy()
        self.group_by_id = dict(zip(self.train_ids,self.train_groups))
        self.pre = preprocessing()
        self.x = self.pre.fit_transform(engineering_inputs(frame, anchors, self.spec['structure'], self.spec['features']))
        self.u = np.linspace(0, 1, self.config['canonical_points'])
        self.y = np.array([aligned(curves[cid], self.u) for cid in self.train_ids])
        template = self.templates(self.x, frame.curve_id.tolist())
        residual = self.logits(self.y)-self.logits(template)
        residual[:, [0, len(self.u)//2, -1]] = 0
        k = min(self.spec['k'], len(frame)-1)
        self.basis = PCA(n_components=k, svd_solver='full').fit(residual)
        self.coefficient_scale = np.sqrt(self.basis.explained_variance_).clip(.01)
        # The PCA mean is a learnable direction, not an unconditional residual.
        self.residual_basis = np.vstack((self.basis.mean_,self.basis.components_))
        self.coefficient_scale = np.r_[1.,self.coefficient_scale]
        self.anchor_scale = anchor_coordinates(torch.tensor(frame[TARGETS].to_numpy(), dtype=torch.float64)).std(0).numpy().clip(.1)
        return self

    @staticmethod
    def logits(y):
        y = np.clip(y, 1e-5, 1-1e-5)
        return np.log(y)-np.log1p(-y)

    def templates(self, x, query_ids=None):
        out = []
        for i, q in enumerate(x):
            d = np.linalg.norm(self.x-q, axis=1)
            if query_ids is not None:
                group = self.group_by_id.get(query_ids[i])
                if group is not None: d[self.train_groups == group] = np.inf
            if self.spec['template'] == 'mean':
                out.append(self.y[np.isfinite(d)].mean(0)); continue
            ix = np.argsort(d, kind='stable')[:min(self.config['neighbor_count'], np.isfinite(d).sum())]
            if not len(ix):
                out.append(self.y.mean(0)); continue
            w = np.exp(-(d[ix]-d[ix].min())/max(np.median(d[ix]), 1e-8))
            out.append(np.average(self.y[ix], axis=0, weights=w))
        return np.asarray(out)

    def arrays(self, frame, anchors, curves=None, leave_self_out=False):
        x = self.pre.transform(engineering_inputs(frame, anchors, self.spec['structure'], self.spec['features']))
        template = self.templates(x, frame.curve_id.tolist() if leave_self_out else None)
        out = dict(x=x, template=self.logits(template), anchors=np.asarray(anchors))
        if all(t in frame for t in TARGETS):
            out['target'] = frame[TARGETS].to_numpy(float)
        if curves is not None:
            out['true_t'],out['true_f'] = pad_curves(frame.curve_id.tolist(),curves)
        return out


class ResidualNetwork(nn.Module):
    def __init__(self, representation):
        super().__init__()
        width = representation.spec['width']
        self.body = nn.Sequential(nn.Linear(representation.x.shape[1], width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.coefficients = nn.Linear(width, len(representation.coefficient_scale))
        self.calibration = nn.Linear(width, 4)
        nn.init.zeros_(self.coefficients.weight); nn.init.zeros_(self.coefficients.bias)
        nn.init.zeros_(self.calibration.weight); nn.init.zeros_(self.calibration.bias)
        self.register_buffer('basis', torch.tensor(representation.residual_basis, dtype=torch.float32))
        self.calibration_enabled = representation.spec['calibration']=='limited'
        self.register_buffer('score_scale', torch.tensor(representation.coefficient_scale, dtype=torch.float32))
        points = self.basis.shape[1]
        mask = torch.ones(points); mask[[0, points//2, -1]] = 0
        peak = torch.zeros(points); peak[points//2] = 1
        self.register_buffer('mask', mask); self.register_buffer('peak', peak)

    def forward(self, x, template):
        z = self.body(x)
        coeff = self.coefficients(z)
        residual = (coeff*self.score_scale)@self.basis
        raw = torch.sigmoid(template+residual)*self.mask+self.peak
        correction = self.calibration(z)
        if not self.calibration_enabled: correction = correction*0
        return raw, correction, coeff


def tensor_pack(arrays, device, dtype=torch.float32):
    return {k:torch.as_tensor(v, dtype=dtype, device=device) for k,v in arrays.items()}


def objective(model, pack, representation, config):
    raw, correction, coeff = model(pack['x'], pack['template'])
    anchors, _ = consistent_anchors(pack['anchors'], representation.spec['structure'], correction)
    t, f, info = reconstruct(raw, anchors)
    ratio, nrmse, _ = pwl_metrics(pack['true_t'], pack['true_f'], t, f)
    scale = pack['anchor_scale']
    ae = ((anchor_coordinates(anchors)-anchor_coordinates(pack['target']))/scale).square().mean()
    loss = torch.log1p(ratio).mean()+config['tail_loss_weight']*torch.relu(ratio-1).mean()
    loss = loss+representation.spec['anchor_weight']*ae+config['residual_loss_weight']*coeff.square().mean()
    return loss, ratio, anchors, info


def train_network(representation, arrays, config, seed, stop_arrays=None, epochs=None, checkpoint=None, budget=None):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    model = ResidualNetwork(representation).to(device)
    pack = tensor_pack(arrays, device)
    stop = tensor_pack(stop_arrays, device) if stop_arrays is not None else None
    pack['anchor_scale'] = torch.as_tensor(representation.anchor_scale, dtype=torch.float32, device=device)
    if stop is not None: stop['anchor_scale'] = pack['anchor_scale']
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],
                           capturable=(device == 'cuda'), foreach=False)
    maximum = epochs if epochs is not None else config['maximum_epochs']
    start_epoch, best_epoch, best_score = 0, 0, float('inf')
    best_state = None
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        start_epoch, best_epoch, best_score = state['epoch'], state['best_epoch'], state['best_score']
        best_state = state['best_state']
        torch.set_rng_state(state['rng'].cpu())
    started = time.monotonic()
    epoch = start_epoch
    with torch.no_grad():
        loss, _, _, _ = objective(model, pack, representation, config)

    def step():
        value, _, _, _ = objective(model, pack, representation, config)
        value.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        return value

    graph_enabled = device == 'cuda' and config.get('cuda_graph', False) and maximum-start_epoch > 3
    graph = None
    stream = torch.cuda.Stream() if graph_enabled else None
    for epoch in range(start_epoch+1, maximum+1):
        if graph_enabled and epoch == start_epoch+4:
            graph = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                loss = step()
            graph.replay()
        elif graph is not None:
            graph.replay()
        elif graph_enabled:
            # Three real warmup updates count toward the unchanged epoch budget.
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                opt.zero_grad(set_to_none=True); loss = step()
            torch.cuda.current_stream().wait_stream(stream)
        else:
            opt.zero_grad(set_to_none=True); loss = step()
        if epoch % config['early_stopping_check_every'] == 0 or epoch == maximum:
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite joint physical loss')
            if stop is not None:
                with torch.no_grad():
                    _, ratios, _, _ = objective(model, stop, representation, config)
                    score = float(torch.quantile(ratios, .5).cpu())
                if score < best_score-1e-6:
                    best_score, best_epoch = score, epoch
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                best_epoch = epoch
            if checkpoint is not None and (epoch % 100 == 0 or epoch == maximum):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temp = checkpoint.with_suffix('.tmp')
                torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                    best_epoch=best_epoch, best_score=best_score, best_state=best_state,
                    rng=torch.get_rng_state()), temp)
                os.replace(temp, checkpoint)
            if budget is not None: budget.check()
            if stop is not None and epoch-best_epoch >= config['early_stopping_patience']:
                break
    if stop is not None:
        if best_state is None: raise RuntimeError('No early-stopping checkpoint')
        model.load_state_dict(best_state)
    elapsed = time.monotonic()-started
    profile = dict(seed=seed, epochs_completed=epoch, selected_epochs=best_epoch, cuda_graph=graph_enabled,
                   best_stop_median_R2=1-best_score if stop is not None else None,
                   wall_seconds=elapsed, final_training_loss=float(loss.detach().cpu()))
    return model.cpu().double().eval(), profile


class JointModel:
    def __init__(self, representation, networks, profiles):
        self.representation, self.networks, self.profiles = representation, networks, profiles
        self.train_ids = representation.train_ids

    def predict(self, frame, base_anchors, projection='power', calibrated=True, seed_index=None,
                oracle_anchors=None, oracle_raw=None, outputs_only=False, leave_self_out=False):
        rep = self.representation
        arrays = rep.arrays(frame, base_anchors, leave_self_out=leave_self_out)
        pack = tensor_pack(arrays, 'cpu', torch.float64)
        models = self.networks if seed_index is None else [self.networks[seed_index]]
        with torch.no_grad():
            outputs = [m(pack['x'], pack['template']) for m in models]
            raw = torch.stack([o[0] for o in outputs]).mean(0)
            reconciled = [consistent_anchors(pack['anchors'], rep.spec['structure'], o[1] if calibrated else None)
                          for o in outputs]
            aa = [item[0] for item in reconciled]
            anchors, projection_delta = consistent_anchors(torch.stack(aa).mean(0), rep.spec['structure'])
            if oracle_anchors is not None:
                anchors = torch.tensor(oracle_anchors, dtype=torch.float64)
            if oracle_raw is not None:
                raw = torch.tensor(oracle_raw, dtype=torch.float64)
            if outputs_only:
                return anchors.numpy()
            t, f, info = reconstruct(raw, anchors, projection)
        out = []
        for ti, fi in zip(t.numpy(), f.numpy()):
            unique, index = np.unique(ti, return_index=True)
            out.append((unique, fi[index]))
        extras = {k:v.numpy() for k,v in info.items()}
        extras['consistency_projection_log_delta'] = projection_delta.numpy()
        extras['seed_max_abs_consistency_projection_log_delta'] = torch.stack([item[1].abs() for item in reconciled]).max(0).values.numpy()
        extras['canonical_raw'] = raw.numpy()
        return anchors.numpy(), out, extras
