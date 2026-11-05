#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vwea_keypoint_shapes.py

V13 strict prediction route:
1) use the V12 anchor model zoo to predict response anchors
   Fmax, Td, Impulse, normalized area, and tp/Td;
2) keep all V5 curve candidates: time-conditioned, PCA, prototype,
   local analog, peak-warped analog, DCT, and peak-shift candidates;
3) add keypoint-wise curve candidates that predict interpretable
   normalized force-time control points and reconstruct the complete curve
   under physical constraints;
4) select the final curve ensemble by validation-set performance only.

No target leakage is introduced: true Fmax, true Td, true impulse, true
curve shape, MorphSignature, source filename, and post-test damage labels are
not used as predictors. True keypoint values are used only as supervised labels.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

warnings.filterwarnings("ignore")

import vwea_core as v3
import vwea_anchor_ensemble as v12

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    XGBRegressor = None
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LGBM = False


# V13 diagnostics are intentionally small: V3 already writes the selected
# ensemble and final per-curve metrics; V12 writes full anchor-model metrics.
KEYPOINT_DIAGNOSTICS: list[dict] = []


def _xaug(X: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([X.reset_index(drop=True), A.reset_index(drop=True)], axis=1)


def _interp_rows(Y: np.ndarray, tau: np.ndarray, positions) -> np.ndarray:
    """Interpolate each row of Y at either common or row-wise positions."""
    Y = np.asarray(Y, float)
    tau = np.asarray(tau, float)
    pos = np.asarray(positions, float)
    if pos.ndim == 1:
        return np.vstack([np.interp(pos, tau, row) for row in Y])
    out = np.zeros(pos.shape, dtype=float)
    for i, row in enumerate(Y):
        out[i] = np.interp(pos[i], tau, row)
    return out


def _unique_sorted_curve(tau: np.ndarray, pos: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Build one nonnegative normalized curve from unordered control points."""
    pos = np.asarray(pos, float)
    val = np.asarray(val, float)
    ok = np.isfinite(pos) & np.isfinite(val)
    pos = np.clip(pos[ok], 0.0, 1.0)
    val = np.clip(val[ok], 0.0, 1.0)
    if len(pos) < 2:
        y = np.zeros_like(tau)
        y[np.argmin(np.abs(tau - 0.2))] = 1.0
        return y

    # Round positions before grouping so duplicated anchors such as 0, 1, and
    # predicted peak are merged deterministically. Use max value to preserve
    # sharp peak constraints.
    pos_round = np.round(pos, 6)
    uniq = np.unique(pos_round)
    pu, vu = [], []
    for p in uniq:
        m = pos_round == p
        pu.append(float(np.mean(pos[m])))
        vu.append(float(np.max(val[m])))
    pu = np.asarray(pu)
    vu = np.asarray(vu)
    order = np.argsort(pu)
    pu, vu = pu[order], vu[order]
    if pu[0] > 0:
        pu = np.r_[0.0, pu]
        vu = np.r_[0.0, vu]
    if pu[-1] < 1:
        pu = np.r_[pu, 1.0]
        vu = np.r_[vu, 0.0]
    y = np.interp(tau, pu, vu)
    y[0] = 0.0
    y[-1] = 0.0
    if y.max() > v3.EPS:
        y = y / y.max()
    return np.clip(y, 0.0, 1.0)


class KeypointCurveRegressor:
    """Predict normalized curve keypoints and reconstruct a full curve.

    mode="original":
        Predicts values at fixed tau positions and at semantic locations around
        the true peak during training. At prediction, the semantic locations are
        placed around predicted tp/Td.

    mode="peakwarp":
        Warps training curves into a canonical coordinate with peak at 0.5,
        predicts canonical keypoints, then unwarps with predicted tp/Td.
    """

    def __init__(self, model, tau, fixed_tau, peak_offsets=None, mode="original"):
        self.model = model
        self.tau = np.asarray(tau, float)
        self.fixed_tau = np.asarray(fixed_tau, float)
        self.peak_offsets = np.asarray([] if peak_offsets is None else peak_offsets, float)
        self.mode = str(mode)

    def fit(self, Xaug: pd.DataFrame, Y: np.ndarray):
        Y = np.asarray(Y, float)
        if self.mode == "peakwarp":
            import vwea_analog_shapes as v5
            H = v5._warp_to_peak(Y, self.tau)
            target = _interp_rows(H, self.tau, self.fixed_tau)
        else:
            p_true = v3.peak_tau(Y, self.tau)
            fixed_y = _interp_rows(Y, self.tau, self.fixed_tau)
            dyn_pos = np.clip(p_true[:, None] + self.peak_offsets[None, :], 0.0, 1.0)
            dyn_y = _interp_rows(Y, self.tau, dyn_pos) if len(self.peak_offsets) else np.zeros((len(Y), 0))
            target = np.hstack([fixed_y, dyn_y])
        target = np.clip(target, 0.0, 1.0)
        self.model.fit(Xaug, target)
        return self

    def _tp_hat(self, Xaug: pd.DataFrame) -> np.ndarray:
        if "hat_tp_ratio" in Xaug.columns:
            tp = pd.to_numeric(Xaug["hat_tp_ratio"], errors="coerce").fillna(0.2).to_numpy(float)
        else:
            tp = np.full(len(Xaug), 0.2, dtype=float)
        return np.clip(tp, 0.015, 0.985)

    def predict(self, Xaug: pd.DataFrame) -> np.ndarray:
        z = np.asarray(self.model.predict(Xaug), float)
        z = np.clip(z, 0.0, 1.0)
        tp = self._tp_hat(Xaug)
        out = np.zeros((len(Xaug), len(self.tau)), dtype=float)

        if self.mode == "peakwarp":
            for i in range(len(Xaug)):
                pos = np.r_[0.0, self.fixed_tau, 0.5, 1.0]
                val = np.r_[0.0, z[i, : len(self.fixed_tau)], 1.0, 0.0]
                out[i] = _unique_sorted_curve(self.tau, pos, val)
            import vwea_analog_shapes as v5
            return v5._unwarp_from_peak(out, self.tau, tp)

        n_fixed = len(self.fixed_tau)
        for i in range(len(Xaug)):
            fixed_val = z[i, :n_fixed]
            dyn_val = z[i, n_fixed:]
            dyn_pos = np.clip(tp[i] + self.peak_offsets, 0.0, 1.0)
            pos = np.r_[0.0, self.fixed_tau, dyn_pos, tp[i], 1.0]
            val = np.r_[0.0, fixed_val, dyn_val, 1.0, 0.0]
            out[i] = _unique_sorted_curve(self.tau, pos, val)
        return out


def _keypoint_regressors(seed: int, jobs: int, fast: bool = False) -> dict:
    n_tree = 300 if fast else 700
    regs = {
        "ET": ExtraTreesRegressor(
            n_estimators=n_tree,
            min_samples_leaf=1,
            max_features=0.90,
            random_state=seed + 1301,
            n_jobs=jobs,
        ),
        "RF": RandomForestRegressor(
            n_estimators=240 if fast else 520,
            min_samples_leaf=2,
            max_features=0.80,
            random_state=seed + 1302,
            n_jobs=jobs,
        ),
    }
    if HAS_XGB:
        regs["XGB"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=180 if fast else 420,
                max_depth=3,
                learning_rate=0.025,
                subsample=0.88,
                colsample_bytree=0.88,
                reg_lambda=6.0,
                objective="reg:squarederror",
                random_state=seed + 1303,
                n_jobs=1,
            ),
            n_jobs=max(1, min(jobs, 4)),
        )
    if HAS_LGBM:
        regs["LGBM"] = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=200 if fast else 480,
                num_leaves=15,
                learning_rate=0.025,
                subsample=0.88,
                colsample_bytree=0.88,
                reg_lambda=6.0,
                random_state=seed + 1304,
                n_jobs=1,
                verbose=-1,
            ),
            n_jobs=max(1, min(jobs, 4)),
        )
    return regs


def _candidate_score(Ytrue: np.ndarray, Ypred: np.ndarray, tau: np.ndarray) -> dict:
    r2s = []
    maes = []
    areas = []
    for i in range(len(Ytrue)):
        try:
            r2s.append(v3.r2_score(Ytrue[i], Ypred[i]))
        except Exception:
            r2s.append(np.nan)
        maes.append(float(np.mean(np.abs(Ytrue[i] - Ypred[i]))))
        areas.append(abs(float(np.trapz(Ytrue[i], tau) - np.trapz(Ypred[i], tau))))
    return {
        "median_R2": float(np.nanmedian(r2s)),
        "median_MAE": float(np.median(maes)),
        "median_area_abs_error": float(np.median(areas)),
    }


def fit_candidates_v13(X, A, Y, idx, tau, prior, seed, jobs, fast, do_smooth):
    candidates = _BASE_FIT_CANDIDATES(X, A, Y, idx, tau, prior, seed, jobs, fast, do_smooth)
    train, val = idx["train"], idx["val"]
    Xaug = _xaug(X, A)
    pre_aug = v3.make_pre(Xaug.iloc[train])

    fixed_tau = np.array([
        0.00, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.125,
        0.15, 0.20, 0.275, 0.35, 0.45, 0.60, 0.75, 0.90, 1.00
    ])
    peak_offsets = np.array([-0.15, -0.10, -0.07, -0.05, -0.025, 0.025, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35])
    canonical_tau = np.array([
        0.00, 0.03, 0.06, 0.10, 0.16, 0.24, 0.34, 0.42,
        0.46, 0.49, 0.50, 0.51, 0.54, 0.58, 0.66, 0.76, 0.88, 1.00
    ])

    for reg_name, reg in _keypoint_regressors(seed, jobs, fast).items():
        for mode, ktau, offsets in [
            ("original", fixed_tau, peak_offsets),
            ("peakwarp", canonical_tau, []),
        ]:
            try:
                pipe = Pipeline([("pre", clone(pre_aug)), ("reg", clone(reg))])
                model = KeypointCurveRegressor(pipe, tau=tau, fixed_tau=ktau, peak_offsets=offsets, mode=mode)
                model.fit(Xaug.iloc[train], Y[train])
                key = f"V13Keypoint_{mode}_{reg_name}"
                candidates[key] = {"kind": "keypoint_v13", "model": model}

                raw = model.predict(Xaug.iloc[val])
                pp = v3.post(raw, tau, A["hat_area_from_impulse"].to_numpy()[val], do_smooth=do_smooth)
                sc = _candidate_score(Y[val], pp, tau)
                KEYPOINT_DIAGNOSTICS.append({"candidate": key, "split": "val", **sc})
                print(
                    f"[SHAPE-V13] fitted {key}: "
                    f"val_median_R2={sc['median_R2']:.4f}, val_median_MAE={sc['median_MAE']:.4f}"
                )
            except Exception as exc:
                print(f"[WARN] V13 keypoint candidate {reg_name}/{mode} failed: {exc}")
    return candidates


def pred_candidate_v13(c, X, A, tau, prior=None):
    if c.get("kind") == "keypoint_v13":
        return c["model"].predict(_xaug(X, A))
    return _BASE_PRED_CANDIDATE(c, X, A, tau, prior)


def _save_v13_tables(outdir: Path) -> None:
    if not KEYPOINT_DIAGNOSTICS:
        return
    df = pd.DataFrame(KEYPOINT_DIAGNOSTICS)
    df.to_csv(outdir / "v13_keypoint_candidate_validation_metrics.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(outdir / "v13_keypoint_candidate_validation_metrics.xlsx", engine="openpyxl") as writer:
        df.sort_values(["split", "median_MAE", "median_R2"], ascending=[True, True, False]).to_excel(
            writer, sheet_name="Keypoint_Candidates", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir", required=True)
    args, _ = parser.parse_known_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reuse V12 scalar anchor fitting, then import V5 so its peak-aware curve
    # candidates become the base candidate set for V13.
    v3.fit_anchor = v12.fit_anchor_v12
    import vwea_analog_shapes as _v5  # noqa: F401

    global _BASE_FIT_CANDIDATES, _BASE_PRED_CANDIDATE
    _BASE_FIT_CANDIDATES = v3.fit_candidates
    _BASE_PRED_CANDIDATE = v3.pred_candidate
    v3.fit_candidates = fit_candidates_v13
    v3.pred_candidate = pred_candidate_v13

    v3.main()
    v12._save_anchor_tables(outdir)
    _save_v13_tables(outdir)


if __name__ == "__main__":
    main()
