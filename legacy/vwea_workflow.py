#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vwea_workflow.py

V14 adds explicit morphology-anchor prediction to the strict RC impact route.

Intermediate targets:
    Fmax, Td, Impulse, normalized area, tp/Td
    rise t10/t50/t90, post-peak t90/t50/t10
    rise duration Tr, fall duration Tf, half-peak width W50
    peak count Np, secondary-peak ratio R21, peak-spacing dtp12
    tail impulse ratio Rtail, centroid time tc
    cumulative impulse times tI10/tI50/tI90

These targets are used in two ways:
1) their prediction accuracies are reported directly;
2) predicted morphology anchors are converted into a constrained normalized
   curve candidate, which is added to the V13 candidate pool and selected only
   through validation-set ensemble weighting.
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
import vwea_keypoint_shapes as v13

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

try:
    from scipy.signal import find_peaks
    HAS_FIND_PEAKS = True
except Exception:
    find_peaks = None
    HAS_FIND_PEAKS = False

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


MORPH_DIAGNOSTICS: list[dict] = []
MORPH_PREDICTIONS: list[pd.DataFrame] = []
MORPH_TARGETS: pd.DataFrame | None = None
MORPH_ANCHORS: dict[str, v3.Anchor] = {}


def _cross_rise(y: np.ndarray, tau: np.ndarray, level: float, pidx: int) -> float:
    idx = np.where(y[: pidx + 1] >= level)[0]
    if len(idx) == 0:
        return float(tau[pidx])
    j = int(idx[0])
    if j == 0:
        return float(tau[0])
    y0, y1 = float(y[j - 1]), float(y[j])
    t0, t1 = float(tau[j - 1]), float(tau[j])
    if abs(y1 - y0) < v3.EPS:
        return t1
    return float(np.clip(t0 + (level - y0) * (t1 - t0) / (y1 - y0), 0.0, 1.0))


def _cross_fall(y: np.ndarray, tau: np.ndarray, level: float, pidx: int) -> float:
    idx = np.where(y[pidx:] <= level)[0]
    if len(idx) == 0:
        return 1.0
    j = int(idx[0] + pidx)
    if j <= pidx:
        return float(tau[pidx])
    y0, y1 = float(y[j - 1]), float(y[j])
    t0, t1 = float(tau[j - 1]), float(tau[j])
    if abs(y1 - y0) < v3.EPS:
        return t1
    return float(np.clip(t0 + (level - y0) * (t1 - t0) / (y1 - y0), 0.0, 1.0))


def _width_above(y: np.ndarray, tau: np.ndarray, level: float) -> float:
    hi = np.where(y >= level)[0]
    if len(hi) == 0:
        return 0.0
    start = _cross_rise(y, tau, level, int(hi[0]))
    last = int(hi[-1])
    if last >= len(y) - 1:
        end = float(tau[last])
    else:
        y0, y1 = float(y[last]), float(y[last + 1])
        t0, t1 = float(tau[last]), float(tau[last + 1])
        end = t1 if abs(y1 - y0) < v3.EPS else t0 + (level - y0) * (t1 - t0) / (y1 - y0)
    return float(np.clip(end - start, 0.0, 1.0))


def _cum_time(y: np.ndarray, tau: np.ndarray, q: float) -> float:
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(tau)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = float(cum[-1])
    if total <= v3.EPS:
        return 0.0
    return float(np.interp(q * total, cum, tau))


def _peak_features(y: np.ndarray, tau: np.ndarray) -> tuple[float, float, float]:
    ys = np.clip(v3.smooth(y), 0.0, 1.0)
    if HAS_FIND_PEAKS:
        peaks, props = find_peaks(ys, height=0.15, prominence=0.04, distance=max(2, len(ys) // 80))
    else:
        peaks = np.where((ys[1:-1] >= ys[:-2]) & (ys[1:-1] >= ys[2:]) & (ys[1:-1] >= 0.15))[0] + 1
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(ys))])
    n_peaks = float(len(peaks))
    heights = ys[peaks]
    if len(peaks) < 2:
        return n_peaks, 0.0, 0.0
    order = np.argsort(heights)[::-1]
    p1, p2 = int(peaks[order[0]]), int(peaks[order[1]])
    r21 = float(np.clip(ys[p2] / (ys[p1] + v3.EPS), 0.0, 1.0))
    dt = float(abs(tau[p2] - tau[p1]))
    return n_peaks, r21, dt


def extract_morph_targets(Y: np.ndarray, tau: np.ndarray) -> pd.DataFrame:
    rows = []
    for row in np.asarray(Y, float):
        y = np.clip(row, 0.0, 1.0)
        if y.max() > v3.EPS:
            y = y / y.max()
        pidx = int(np.argmax(y))
        tp = float(tau[pidx])
        t10 = _cross_rise(y, tau, 0.10, pidx)
        t50 = _cross_rise(y, tau, 0.50, pidx)
        t90 = _cross_rise(y, tau, 0.90, pidx)
        f90 = _cross_fall(y, tau, 0.90, pidx)
        f50 = _cross_fall(y, tau, 0.50, pidx)
        f10 = _cross_fall(y, tau, 0.10, pidx)
        area = float(np.trapz(y, tau))
        tail = float(np.trapz(y[tau >= 0.8], tau[tau >= 0.8]) / (area + v3.EPS)) if area > v3.EPS else 0.0
        centroid = float(np.trapz(tau * y, tau) / (area + v3.EPS)) if area > v3.EPS else tp
        n_peaks, r21, dtp12 = _peak_features(y, tau)
        rows.append({
            "rise_t10": t10,
            "rise_t50": t50,
            "rise_t90": t90,
            "fall_t90": f90,
            "fall_t50": f50,
            "fall_t10": f10,
            "Tr": max(0.0, t90 - t10),
            "Tf": max(0.0, f10 - f90),
            "W50": _width_above(y, tau, 0.50),
            "N_p": n_peaks,
            "R21": r21,
            "dtp12": dtp12,
            "Rtail": np.clip(tail, 0.0, 1.0),
            "tc": np.clip(centroid, 0.0, 1.0),
            "tI10": _cum_time(y, tau, 0.10),
            "tI50": _cum_time(y, tau, 0.50),
            "tI90": _cum_time(y, tau, 0.90),
        })
    return pd.DataFrame(rows).clip(lower=0.0)


def _morph_models(seed: int, jobs: int, fast: bool = False) -> dict:
    n_tree = 220 if fast else 420
    models = {
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 4, 25)),
        "Huber": HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
        "SVR_RBF": SVR(C=12.0, gamma="scale", epsilon=0.025),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n_tree, min_samples_leaf=2, max_features=0.9,
            random_state=seed + 1, n_jobs=jobs,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=180 if fast else 360, min_samples_leaf=2, max_features=0.8,
            random_state=seed + 2, n_jobs=jobs,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=160 if fast else 320, learning_rate=0.03, max_depth=2,
            random_state=seed + 3,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=180 if fast else 360, max_depth=3, learning_rate=0.025,
            subsample=0.88, colsample_bytree=0.88, reg_lambda=6.0,
            objective="reg:squarederror", random_state=seed + 4, n_jobs=jobs,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=180 if fast else 360, num_leaves=15, learning_rate=0.025,
            subsample=0.88, colsample_bytree=0.88, reg_lambda=6.0,
            random_state=seed + 5, n_jobs=jobs, verbose=-1,
        )
    return models


def _denom(y: np.ndarray, target: str) -> np.ndarray:
    if target == "N_p":
        return np.maximum(np.abs(y), 1.0)
    return np.maximum(np.abs(y), 0.03)


def _score(target: str, model: str, split: str, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = np.abs(y_pred - y_true) / _denom(y_true, target) * 100.0
    row = {
        "target": target,
        "model": model,
        "split": split,
        "n": len(y_true),
        "R2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "median_abs_error": float(np.median(np.abs(y_pred - y_true))),
        "median_error_%": float(np.median(err)),
        "mean_error_%": float(np.mean(err)),
        "median_accuracy_%": float(100.0 - np.clip(np.median(err), 0.0, 100.0)),
    }
    if target == "N_p":
        row["rounded_exact_accuracy_%"] = float(100.0 * np.mean(np.rint(y_pred) == np.rint(y_true)))
        row["rounded_within1_accuracy_%"] = float(100.0 * np.mean(np.abs(np.rint(y_pred) - np.rint(y_true)) <= 1))
    return row


def _fit_morph_anchor(target: str, y: np.ndarray, kind: str, Xaug: pd.DataFrame, idx, pre, seed, jobs, fast):
    train, val = idx["train"], idx["val"]
    models = _morph_models(seed, jobs, fast)
    fitted = []
    for name, reg in models.items():
        try:
            pipe = Pipeline([("pre", clone(pre)), ("reg", clone(reg))])
            pipe.fit(Xaug.iloc[train], v3.y_transform(y[train], kind))
            pred_val = v3.y_inverse(pipe.predict(Xaug.iloc[val]), kind)
            if target == "N_p":
                pred_val = np.clip(pred_val, 1.0, 8.0)
            fitted.append((float(np.median(np.abs(pred_val - y[val]))), name, pipe))
        except Exception as exc:
            print(f"[WARN] morph anchor {target}/{name} failed: {exc}")
    if not fitted:
        raise RuntimeError(f"No morphology models fitted for {target}")
    fitted = sorted(fitted, key=lambda x: x[0])
    top = fitted[: min(5, len(fitted))]
    P = np.vstack([v3.y_inverse(m.predict(Xaug.iloc[val]), kind) for _, _, m in top]).T
    if target == "N_p":
        P = np.clip(P, 1.0, 8.0)
    w = v3.opt_weights(P, y[val])
    anchor = v3.Anchor(target, kind, [(name, model) for _, name, model in top], w)
    print(f"[MORPH] {target}: top={[n for _, n, _ in top]}, w={np.round(w, 3)}")

    for _, model_name, model in fitted:
        for split, ix in idx.items():
            pred = v3.y_inverse(model.predict(Xaug.iloc[ix]), kind)
            if target == "N_p":
                pred = np.clip(pred, 1.0, 8.0)
            MORPH_DIAGNOSTICS.append(_score(target, model_name, split, y[ix], pred))
            MORPH_PREDICTIONS.append(pd.DataFrame({
                "target": target,
                "model": model_name,
                "split": split,
                "row_index": ix,
                "y_true": y[ix],
                "y_pred": pred,
                "abs_error": np.abs(pred - y[ix]),
            }))
    for split, ix in idx.items():
        pred = anchor.predict(Xaug.iloc[ix])
        if target == "N_p":
            pred = np.clip(pred, 1.0, 8.0)
        MORPH_DIAGNOSTICS.append(_score(target, "ValWeightedEnsemble", split, y[ix], pred))
        MORPH_PREDICTIONS.append(pd.DataFrame({
            "target": target,
            "model": "ValWeightedEnsemble",
            "split": split,
            "row_index": ix,
            "y_true": y[ix],
            "y_pred": pred,
            "abs_error": np.abs(pred - y[ix]),
        }))
    return anchor


class MorphAnchorCurveReconstructor:
    def __init__(self, anchors: dict[str, v3.Anchor], tau: np.ndarray):
        self.anchors = anchors
        self.tau = np.asarray(tau, float)

    def _predict_frame(self, Xaug: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for name, anchor in self.anchors.items():
            pred = anchor.predict(Xaug)
            if name == "N_p":
                pred = np.clip(pred, 1.0, 8.0)
            else:
                pred = np.clip(pred, 0.0, 1.0)
            out[name] = pred
        return pd.DataFrame(out)

    def predict(self, Xaug: pd.DataFrame) -> np.ndarray:
        M = self._predict_frame(Xaug)
        if "hat_tp_ratio" in Xaug.columns:
            tp = pd.to_numeric(Xaug["hat_tp_ratio"], errors="coerce").fillna(0.2).to_numpy(float)
        else:
            tp = M.get("tI50", pd.Series(np.full(len(Xaug), 0.2))).to_numpy(float)
        tp = np.clip(tp, 0.015, 0.985)
        area = pd.to_numeric(Xaug.get("hat_area_from_impulse", 0.15), errors="coerce")
        if np.isscalar(area):
            area = np.full(len(Xaug), float(area))
        else:
            area = area.fillna(0.15).to_numpy(float)

        out = np.zeros((len(Xaug), len(self.tau)), dtype=float)
        for i in range(len(Xaug)):
            pre = np.sort(np.clip([
                M.loc[i, "rise_t10"], M.loc[i, "rise_t50"], M.loc[i, "rise_t90"]
            ], 0.0, tp[i]))
            post = np.sort(np.clip([
                M.loc[i, "fall_t90"], M.loc[i, "fall_t50"], M.loc[i, "fall_t10"]
            ], tp[i], 1.0))
            rtail = float(np.clip(M.loc[i, "Rtail"], 0.0, 1.0))
            ytail = float(np.clip(rtail * area[i] / 0.20, 0.0, 0.45))
            r21 = float(np.clip(M.loc[i, "R21"], 0.0, 1.0))
            dt = float(np.clip(M.loc[i, "dtp12"], 0.0, 1.0))
            t2 = float(np.clip(tp[i] + dt, 0.0, 1.0))
            pos = np.array([
                0.0, pre[0], pre[1], pre[2], tp[i],
                post[0], post[1], post[2], t2, 0.80, 0.90, 1.0
            ])
            val = np.array([
                0.0, 0.10, 0.50, 0.90, 1.0,
                0.90, 0.50, 0.10, r21, ytail, 0.5 * ytail, 0.0
            ])
            out[i] = v13._unique_sorted_curve(self.tau, pos, val)
        return out


def fit_candidates_v14(X, A, Y, idx, tau, prior, seed, jobs, fast, do_smooth):
    candidates = _BASE_FIT_CANDIDATES(X, A, Y, idx, tau, prior, seed, jobs, fast, do_smooth)
    global MORPH_TARGETS, MORPH_ANCHORS
    MORPH_TARGETS = extract_morph_targets(Y, tau)
    Xaug = v13._xaug(X, A)
    pre_aug = v3.make_pre(Xaug.iloc[idx["train"]])
    kinds = {
        "rise_t10": "logit", "rise_t50": "logit", "rise_t90": "logit",
        "fall_t90": "logit", "fall_t50": "logit", "fall_t10": "logit",
        "Tr": "logit", "Tf": "logit", "W50": "logit",
        "N_p": "identity", "R21": "logit", "dtp12": "logit",
        "Rtail": "logit", "tc": "logit",
        "tI10": "logit", "tI50": "logit", "tI90": "logit",
    }
    MORPH_ANCHORS = {}
    for j, (name, kind) in enumerate(kinds.items()):
        y = MORPH_TARGETS[name].to_numpy(float)
        if kind == "logit":
            y = np.clip(y, 1e-4, 1 - 1e-4)
        MORPH_ANCHORS[name] = _fit_morph_anchor(
            name, y, kind, Xaug, idx, pre_aug, seed + 1400 + j * 13, jobs, True
        )
    candidates["V14MorphAnchor_Reconstruction"] = {
        "kind": "morph_anchor_v14",
        "model": MorphAnchorCurveReconstructor(MORPH_ANCHORS, tau),
    }
    raw = candidates["V14MorphAnchor_Reconstruction"]["model"].predict(Xaug.iloc[idx["val"]])
    pp = v3.post(raw, tau, A["hat_area_from_impulse"].to_numpy()[idx["val"]], do_smooth=do_smooth)
    sc = v13._candidate_score(Y[idx["val"]], pp, tau)
    print(
        f"[SHAPE-V14] fitted V14MorphAnchor_Reconstruction: "
        f"val_median_R2={sc['median_R2']:.4f}, val_median_MAE={sc['median_MAE']:.4f}"
    )
    return candidates


def pred_candidate_v14(c, X, A, tau, prior=None):
    if c.get("kind") == "morph_anchor_v14":
        return c["model"].predict(v13._xaug(X, A))
    return _BASE_PRED_CANDIDATE(c, X, A, tau, prior)


def _save_morph_tables(outdir: Path) -> None:
    if MORPH_TARGETS is not None:
        MORPH_TARGETS.to_csv(outdir / "v14_true_morphology_targets.csv", index=False, encoding="utf-8-sig")
    diag = pd.DataFrame(MORPH_DIAGNOSTICS)
    preds = pd.concat(MORPH_PREDICTIONS, ignore_index=True) if MORPH_PREDICTIONS else pd.DataFrame()
    if diag.empty:
        return
    xlsx = outdir / "v14_morph_anchor_model_accuracy.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        diag.sort_values(["target", "split", "median_abs_error"]).to_excel(writer, sheet_name="All_Model_Metrics", index=False)
        diag[diag["model"].eq("ValWeightedEnsemble")].sort_values(["target", "split"]).to_excel(
            writer, sheet_name="Ensemble_Metrics", index=False
        )
        if not preds.empty:
            preds.to_excel(writer, sheet_name="Morph_Predictions", index=False)
    diag.to_csv(outdir / "v14_morph_anchor_model_accuracy.csv", index=False, encoding="utf-8-sig")
    if not preds.empty:
        preds.to_csv(outdir / "v14_morph_anchor_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] morph anchor metrics: {xlsx}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir", required=True)
    args, _ = parser.parse_known_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Core anchors: V12. Base curve candidates: V5 + V13 keypoints.
    v3.fit_anchor = v12.fit_anchor_v12
    import vwea_analog_shapes as _v5  # noqa: F401

    v13._BASE_FIT_CANDIDATES = v3.fit_candidates
    v13._BASE_PRED_CANDIDATE = v3.pred_candidate
    v3.fit_candidates = v13.fit_candidates_v13
    v3.pred_candidate = v13.pred_candidate_v13

    global _BASE_FIT_CANDIDATES, _BASE_PRED_CANDIDATE
    _BASE_FIT_CANDIDATES = v3.fit_candidates
    _BASE_PRED_CANDIDATE = v3.pred_candidate
    v3.fit_candidates = fit_candidates_v14
    v3.pred_candidate = pred_candidate_v14

    v3.main()
    v12._save_anchor_tables(outdir)
    v13._save_v13_tables(outdir)
    _save_morph_tables(outdir)


if __name__ == "__main__":
    main()
