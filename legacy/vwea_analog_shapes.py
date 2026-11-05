#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vwea_analog_shapes.py

Drop-in strict SJJ-only enhancer for vwea_core.py.
Run this file in the same folder as vwea_core.py.

Purpose
-------
V3 improved the strict route with a time-conditioned model, but peak time and local
morphology may still be poor when only tabular SJJ parameters are used. V5 keeps the
same strict boundary and adds three no-leakage mechanisms:

1) Peak-centered time features for the time-conditioned model
   The time model sees tau, predicted tp/Td, tau - tp, rising/falling phase,
   local Gaussian bands around predicted peak, etc. This directly targets peak-time
   misalignment.

2) Local analog curve-memory candidates
   At prediction time, the model retrieves nearest training specimens in the space of
   SJJ engineering features + predicted anchors. It never uses source ID, true target
   curve, MorphSignature, true Fmax, or true Td as an input. It only uses training-set
   normalized curves as supervised labels, which is allowed.

3) Peak-warped analog candidates
   Training curves are warped into a canonical coordinate where the peak is at u=0.5.
   The query curve is obtained through analog averaging in that canonical space and then
   unwarped using the predicted tp/Td. This is designed for the specific problem that
   the peak time is often inaccurate.

This script is intentionally a monkey-patched wrapper around vwea_core.py
so that all split logic, leakage filtering, output files, metrics, and model bundle
format remain the same.

Examples
--------
python vwea_analog_shapes.py --data rectangular_curve_dataset_200pts.xlsx ^
  --sheet Dataset --curve-prefix f_norm_ --split random ^
  --outdir strict_v5_random_peakwarp_nosmooth --no-smooth --n-jobs 8

python vwea_analog_shapes.py --data rectangular_curve_dataset_200pts.xlsx ^
  --sheet Dataset --curve-prefix f_norm_ --split source_group --source-col source_pdf_filename ^
  --outdir strict_v5_source_peakwarp_nosmooth --no-smooth --n-jobs 8
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import vwea_core as v3
except Exception as e:
    raise SystemExit(
        "Cannot import vwea_core.py. Put this file in the same folder as "
        "vwea_core.py and run from that folder. Error: %s" % e
    )

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

try:
    from scipy.fft import dct, idct
    HAS_DCT = True
except Exception:
    HAS_DCT = False

EPS = 1e-9

_BASE_LONG_FRAME = v3.long_frame
_BASE_FIT_CANDIDATES = v3.fit_candidates
_BASE_PRED_CANDIDATE = v3.pred_candidate


def _xaug(X, A):
    return pd.concat([X.reset_index(drop=True), A.reset_index(drop=True)], axis=1)


def long_frame_v5(X, A, tau, prior=None):
    """V3 long frame + explicit peak-centered time interactions."""
    L = _BASE_LONG_FRAME(X, A, tau, prior)
    n = len(X)
    m = len(tau)
    tau_rep = np.tile(tau, n)

    # Use predicted peak ratio if available. Fall back to 0.2 for RC impact curves.
    if isinstance(A, pd.DataFrame) and "hat_tp_ratio" in A.columns:
        tp = np.repeat(np.clip(pd.to_numeric(A["hat_tp_ratio"], errors="coerce").fillna(0.2).to_numpy(float), 0.015, 0.985), m)
    else:
        tp = np.full(n * m, 0.2, dtype=float)

    L["tau_minus_hat_tp"] = tau_rep - tp
    L["abs_tau_minus_hat_tp"] = np.abs(tau_rep - tp)
    L["is_rising_pred"] = (tau_rep <= tp).astype(float)
    L["tau_over_hat_tp"] = tau_rep / (tp + EPS)
    L["fall_tau_over_one_minus_tp"] = (tau_rep - tp) / (1.0 - tp + EPS)
    L["fall_tau_over_one_minus_tp"] = np.clip(L["fall_tau_over_one_minus_tp"], 0.0, 1.0)
    L["rising_progress"] = np.clip(tau_rep / (tp + EPS), 0.0, 1.0)
    L["falling_progress"] = np.clip((tau_rep - tp) / (1.0 - tp + EPS), 0.0, 1.0)

    # Local peak basis. These allow tree/boosting models to place high force near the
    # predicted peak and avoid spreading the maximum across the full time axis.
    for bw in (0.025, 0.05, 0.10, 0.18):
        L[f"peak_gauss_bw_{bw}"] = np.exp(-0.5 * ((tau_rep - tp) / bw) ** 2)

    # Useful anchor interactions. Missing columns are ignored by the preprocessor.
    if isinstance(A, pd.DataFrame):
        for col in ["hat_area_from_impulse", "hat_area", "hat_Impulse", "hat_Td", "hat_Fmax"]:
            if col in A.columns:
                val = np.repeat(pd.to_numeric(A[col], errors="coerce").fillna(0).to_numpy(float), m)
                L[f"{col}_x_tau"] = val * tau_rep
                L[f"{col}_x_peak_gauss_005"] = val * L["peak_gauss_bw_0.05"]

    return L


class LocalKNNCurveRegressor:
    """Weighted local average of training curves in preprocessed X/A feature space."""
    def __init__(self, pre, k=8, temperature=0.8, power=2.0):
        self.pre = pre
        self.k = int(k)
        self.temperature = float(temperature)
        self.power = float(power)

    def fit(self, X, Y):
        self.pre_ = clone(self.pre)
        self.Z_ = self.pre_.fit_transform(X)
        self.Y_ = np.asarray(Y, float)
        self.k_ = min(self.k, len(self.Y_))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_, metric="euclidean")
        self.nn_.fit(self.Z_)
        loo_k = min(self.k_ + 1, len(self.Z_))
        loo = self.nn_.kneighbors(self.Z_, n_neighbors=loo_k)[0][:, -1]
        self.training_distance_scale_ = max(float(np.median(loo)), EPS)
        return self

    def predict(self, X):
        Z = self.pre_.transform(X)
        dist, ind = self.nn_.kneighbors(Z, return_distance=True)
        # Adaptive kernel: robust when feature scales and local densities vary.
        local_scale = np.maximum(np.median(dist, axis=1, keepdims=True), EPS)
        global_scale = self.training_distance_scale_
        scale = 0.5 * local_scale + 0.5 * global_scale
        w = np.exp(-np.power(dist / (self.temperature * scale + EPS), self.power))
        w = w / (w.sum(axis=1, keepdims=True) + EPS)
        return np.einsum("ij,ijk->ik", w, self.Y_[ind])


def _peak_ratio(Y, tau):
    Y = np.asarray(Y, float)
    return tau[np.nanargmax(Y, axis=1)]


def _warp_to_peak(Y, tau, peak_ratio=None):
    """Map every curve to canonical u in [0,1], with peak at u=0.5."""
    Y = np.asarray(Y, float)
    tau = np.asarray(tau, float)
    u = tau.copy()
    if peak_ratio is None:
        peak_ratio = _peak_ratio(Y, tau)
    H = np.zeros_like(Y)
    for i, y in enumerate(Y):
        p = float(np.clip(peak_ratio[i], 0.015, 0.985))
        tau_src = np.where(u <= 0.5, 2.0 * p * u, p + 2.0 * (1.0 - p) * (u - 0.5))
        h = np.interp(tau_src, tau, y)
        h[0] = 0.0
        h[-1] = 0.0
        # Force canonical peak anchor.
        mid = int(np.argmin(np.abs(u - 0.5)))
        h[mid] = max(h[mid], 1.0)
        mx = np.nanmax(h)
        if np.isfinite(mx) and mx > EPS:
            h = h / mx
        H[i] = np.clip(h, 0.0, 1.0)
    return H


def _unwarp_from_peak(H, tau, peak_ratio):
    """Map canonical peak-warped curves H(u) back to original tau coordinate."""
    H = np.asarray(H, float)
    tau = np.asarray(tau, float)
    u_grid = tau.copy()
    peak_ratio = np.asarray(peak_ratio, float)
    Y = np.zeros_like(H)
    for i, h in enumerate(H):
        p = float(np.clip(peak_ratio[i], 0.015, 0.985))
        u_query = np.where(tau <= p, 0.5 * tau / (p + EPS), 0.5 + 0.5 * (tau - p) / (1.0 - p + EPS))
        y = np.interp(u_query, u_grid, h)
        y[0] = 0.0
        y[-1] = 0.0
        # Enforce a peak near predicted tp without changing the full shape too much.
        j = int(np.argmin(np.abs(tau - p)))
        y[j] = max(y[j], 1.0)
        mx = np.nanmax(y)
        if np.isfinite(mx) and mx > EPS:
            y = y / mx
        Y[i] = np.clip(y, 0.0, 1.0)
    return Y


class PeakWarpLocalKNNRegressor:
    """KNN analog model in canonical peak-warped curve space."""
    def __init__(self, pre, tau, k=8, temperature=0.8, power=2.0, tp_col="hat_tp_ratio"):
        self.pre = pre
        self.tau = np.asarray(tau, float)
        self.k = int(k)
        self.temperature = float(temperature)
        self.power = float(power)
        self.tp_col = tp_col

    def fit(self, X, Y):
        self.pre_ = clone(self.pre)
        self.Z_ = self.pre_.fit_transform(X)
        self.H_ = _warp_to_peak(Y, self.tau)
        self.k_ = min(self.k, len(self.H_))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_, metric="euclidean")
        self.nn_.fit(self.Z_)
        loo_k = min(self.k_ + 1, len(self.Z_))
        loo = self.nn_.kneighbors(self.Z_, n_neighbors=loo_k)[0][:, -1]
        self.training_distance_scale_ = max(float(np.median(loo)), EPS)
        return self

    def predict(self, X):
        Z = self.pre_.transform(X)
        dist, ind = self.nn_.kneighbors(Z, return_distance=True)
        local_scale = np.maximum(np.median(dist, axis=1, keepdims=True), EPS)
        global_scale = self.training_distance_scale_
        scale = 0.5 * local_scale + 0.5 * global_scale
        w = np.exp(-np.power(dist / (self.temperature * scale + EPS), self.power))
        w = w / (w.sum(axis=1, keepdims=True) + EPS)
        Hq = np.einsum("ij,ijk->ik", w, self.H_[ind])
        if isinstance(X, pd.DataFrame) and self.tp_col in X.columns:
            tp = pd.to_numeric(X[self.tp_col], errors="coerce").fillna(0.2).to_numpy(float)
        else:
            tp = np.full(len(X), 0.2, dtype=float)
        return _unwarp_from_peak(Hq, self.tau, tp)


class DCTCurveRegressor:
    """Predict low-frequency DCT coefficients and reconstruct normalized curves."""
    def __init__(self, pre, reg, ncoef=32, peak_warp=False, tau=None):
        self.pre = pre
        self.reg = reg
        self.ncoef = int(ncoef)
        self.peak_warp = bool(peak_warp)
        self.tau = None if tau is None else np.asarray(tau, float)

    def fit(self, X, Y):
        if not HAS_DCT:
            raise RuntimeError("scipy.fft.dct is unavailable")
        Yfit = _warp_to_peak(Y, self.tau) if self.peak_warp else np.asarray(Y, float)
        self.n_points_ = Yfit.shape[1]
        self.pre_ = clone(self.pre)
        Z = self.pre_.fit_transform(X)
        C = dct(Yfit, type=2, norm="ortho", axis=1)[:, :self.ncoef]
        self.reg_ = clone(self.reg)
        self.reg_.fit(Z, C)
        return self

    def predict(self, X):
        Z = self.pre_.transform(X)
        Csmall = self.reg_.predict(Z)
        C = np.zeros((len(X), self.n_points_), dtype=float)
        C[:, :self.ncoef] = Csmall
        H = idct(C, type=2, norm="ortho", axis=1)
        if self.peak_warp:
            if isinstance(X, pd.DataFrame) and "hat_tp_ratio" in X.columns:
                tp = pd.to_numeric(X["hat_tp_ratio"], errors="coerce").fillna(0.2).to_numpy(float)
            else:
                tp = np.full(len(X), 0.2, dtype=float)
            return _unwarp_from_peak(H, self.tau, tp)
        return H


class PeakShiftWrapper:
    """Wrap an existing candidate and relocate its maximum to predicted tp/Td."""
    def __init__(self, base_candidate, tau):
        self.base_candidate = base_candidate
        self.tau = np.asarray(tau, float)


def _peak_shift(Y, tau, tp_hat):
    """Shift each predicted shape so current peak moves to tp_hat by canonical warping."""
    Y = np.asarray(Y, float)
    p0 = _peak_ratio(Y, tau)
    H = _warp_to_peak(Y, tau, p0)
    return _unwarp_from_peak(H, tau, tp_hat)


def fit_candidates_v5(X, A, Y, idx, tau, prior, seed, jobs, fast, do_smooth):
    candidates = _BASE_FIT_CANDIDATES(X, A, Y, idx, tau, prior, seed, jobs, fast, do_smooth)

    train = idx["train"]
    Xaug_all = _xaug(X, A)
    Xaug_tr = Xaug_all.iloc[train]
    pre_aug = v3.make_pre(Xaug_tr)
    n_train = len(train)

    # Local analog candidates. These often help random split because nearby specimens
    # preserve multi-peak and plateau shapes better than low-K prototypes.
    for k in [1, 3, 5, 8, 12, 18, 25, 35]:
        if k >= n_train:
            continue
        for temp in ([0.55, 0.85] if k <= 12 else [0.85]):
            try:
                model = LocalKNNCurveRegressor(pre=pre_aug, k=k, temperature=temp)
                model.fit(Xaug_tr, Y[train])
                key = f"LocalAnalog_K{k}_T{str(temp).replace('.', 'p')}"
                candidates[key] = {"kind": "local_knn_v5", "model": model}
                print(f"[SHAPE-V5] fitted {key}")
            except Exception as e:
                print(f"[WARN] LocalAnalog_K{k} failed: {e}")

    # Peak-warped local analog candidates. These are specifically designed to improve
    # peak occurrence time without using the target curve at prediction time.
    for k in [1, 3, 5, 8, 12, 18, 25, 35]:
        if k >= n_train:
            continue
        for temp in ([0.55, 0.85] if k <= 12 else [0.85]):
            try:
                model = PeakWarpLocalKNNRegressor(pre=pre_aug, tau=tau, k=k, temperature=temp)
                model.fit(Xaug_tr, Y[train])
                key = f"PeakWarpAnalog_K{k}_T{str(temp).replace('.', 'p')}"
                candidates[key] = {"kind": "peakwarp_knn_v5", "model": model}
                print(f"[SHAPE-V5] fitted {key}")
            except Exception as e:
                print(f"[WARN] PeakWarpAnalog_K{k} failed: {e}")

    # DCT functional regressors, normal and peak-warped.
    if HAS_DCT:
        for ncoef in [12, 20, 32, 48, 64]:
            if ncoef >= Y.shape[1] or ncoef >= n_train:
                continue
            for peak_warp in [False, True]:
                try:
                    base = ExtraTreesRegressor(
                        n_estimators=250 if fast else 750,
                        min_samples_leaf=2,
                        random_state=seed + 700 + ncoef + (100 if peak_warp else 0),
                        n_jobs=jobs,
                    )
                    reg = MultiOutputRegressor(base, n_jobs=max(1, min(jobs, 4)))
                    model = DCTCurveRegressor(pre=pre_aug, reg=reg, ncoef=ncoef, peak_warp=peak_warp, tau=tau)
                    model.fit(Xaug_tr, Y[train])
                    key = ("PeakWarp" if peak_warp else "") + f"DCT{ncoef}_ET"
                    candidates[key] = {"kind": "dct_v5", "model": model}
                    print(f"[SHAPE-V5] fitted {key}")
                except Exception as e:
                    print(f"[WARN] DCT{ncoef} peak_warp={peak_warp} failed: {e}")

    # PCA on peak-warped canonical curves. This is more stable than direct PCA when
    # the same physical shape appears with different peak locations.
    try:
        Htr = _warp_to_peak(Y[train], tau)
        ncomp = min(24, max(3, min(n_train - 2, Y.shape[1] // 3)))
        pca = PCA(n_components=ncomp, random_state=seed + 888).fit(Htr)
        ztr = pca.transform(Htr)
        rf = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=250 if fast else 750,
                min_samples_leaf=2,
                random_state=seed + 889,
                n_jobs=jobs,
            ),
            n_jobs=max(1, min(jobs, 4)),
        )
        pipe = Pipeline([("pre", clone(pre_aug)), ("reg", rf)])
        pipe.fit(Xaug_tr, ztr)
        candidates[f"PeakWarpPCA{ncomp}_RF"] = {"kind": "peakwarp_pca_v5", "pca": pca, "model": pipe, "tau": np.asarray(tau, float)}
        print(f"[SHAPE-V5] fitted PeakWarpPCA{ncomp}_RF")
    except Exception as e:
        print(f"[WARN] PeakWarpPCA_RF failed: {e}")

    # Peak-shifted copies of the original V3 candidates. Validation decides whether
    # forcing the maximum to predicted tp is beneficial.
    base_names = [name for name, c in candidates.items() if c.get("kind") in {"time", "pca", "proto"}]
    for name in base_names:
        candidates[f"PeakShift_{name}"] = {"kind": "peakshift_v5", "base": candidates[name], "tau": np.asarray(tau, float)}

    return candidates


def pred_candidate_v5(c, X, A, tau, prior=None):
    kind = c.get("kind")
    if kind in {"local_knn_v5", "peakwarp_knn_v5", "dct_v5"}:
        return c["model"].predict(_xaug(X, A))
    if kind == "peakwarp_pca_v5":
        Xaug = _xaug(X, A)
        H = c["pca"].inverse_transform(c["model"].predict(Xaug))
        if "hat_tp_ratio" in Xaug.columns:
            tp = pd.to_numeric(Xaug["hat_tp_ratio"], errors="coerce").fillna(0.2).to_numpy(float)
        else:
            tp = np.full(len(Xaug), 0.2, dtype=float)
        return _unwarp_from_peak(H, c["tau"], tp)
    if kind == "peakshift_v5":
        raw = pred_candidate_v5(c["base"], X, A, tau, prior)
        if isinstance(A, pd.DataFrame) and "hat_tp_ratio" in A.columns:
            tp = pd.to_numeric(A["hat_tp_ratio"], errors="coerce").fillna(0.2).to_numpy(float)
        else:
            tp = np.full(len(X), 0.2, dtype=float)
        return _peak_shift(raw, np.asarray(tau, float), tp)
    return _BASE_PRED_CANDIDATE(c, X, A, tau, prior)


# Monkey patches. V3 main will now train/evaluate V5 candidates and peak-centered
# time features while keeping all leakage filters and outputs unchanged.
v3.long_frame = long_frame_v5
v3.fit_candidates = fit_candidates_v5
v3.pred_candidate = pred_candidate_v5


if __name__ == "__main__":
    v3.main()
