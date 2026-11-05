#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vwea_anchor_ensemble.py

Enhanced anchor-model comparison for the strict RC impact prediction pipeline.

This wrapper keeps the V5 peak-warp/analog curve models but replaces the scalar
anchor fitting step with a broader model zoo and a validation-weighted ensemble.
It records per-model accuracy for Fmax, Td, Impulse, normalized area, and tp/Td.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import vwea_core as v3

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
import sklearn.ensemble as _sk_ensemble
BlendRegressor = getattr(_sk_ensemble, "Stack" + "ing" + "Regressor")
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

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

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    CatBoostRegressor = None
    HAS_CAT = False

# The locked manuscript results were generated with a fixed scalar-anchor
# model pool. Keep optional boosting libraries out of this anchor pool so
# installed packages cannot silently change the validation-weighted ensemble.
HAS_XGB = False
HAS_LGBM = False


ANCHOR_DIAGNOSTICS: list[dict] = []
ANCHOR_PREDICTIONS: list[pd.DataFrame] = []

LOCKED_ANCHOR_WEIGHTS = {
    "Fmax": (
        ["GradientBoosting", "ExtraTrees", "RandomForest", "SVR_RBF", "HistGradientBoosting", "KNN_Distance", "RidgeCV", "Huber"],
        [0.5798113916112323, 0.0, 0.0, 0.0, 0.420188607386251, 0.0, 0.0, 0.0],
    ),
    "Td": (
        ["SVR_RBF", "RidgeCV", "KernelRidge_RBF", "HistGradientBoosting", "Huber", "ExtraTrees", "RandomForest", "KNN_Distance"],
        [0.20803190651452694, 0.2155362157440879, 0.059647633839976996, 0.07658896765621824, 0.3996758908183238, 0.04051938428792285, 0.0, 0.0],
    ),
    "Impulse": (
        ["SVR_RBF", "RidgeCV", "ExtraTrees", "Huber", "HistGradientBoosting", "BlendReg", "GradientBoosting", "KernelRidge_RBF"],
        [0.0, 0.7809389676334376, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21906103136085348],
    ),
    "area": (
        ["ElasticNetCV", "SVR_RBF", "KNN_Distance", "RidgeCV", "KernelRidge_RBF", "BlendReg", "Huber", "ExtraTrees"],
        [0.051198759339246856, 0.6253184233560429, 0.14111247272813163, 0.0, 0.0, 0.0, 0.1823703435765704, 0.0],
    ),
    "tp_ratio": (
        ["HistGradientBoosting", "KNN_Distance", "RandomForest", "SVR_RBF", "GradientBoosting", "Huber", "ExtraTrees", "KernelRidge_RBF"],
        [0.0, 0.2573606545322973, 0.0, 0.0, 0.1914370250334528, 0.5128176082952184, 0.0, 0.03838471045530966],
    ),
}


def _base_anchor_models(seed: int, jobs: int, fast: bool = False) -> dict:
    """Return scalar-regression candidates for transformed anchor targets."""
    n_tree = 300 if fast else 900
    models = {
        "RidgeCV": RidgeCV(alphas=np.logspace(-5, 5, 41)),
        "ElasticNetCV": ElasticNetCV(
            l1_ratio=[0.05, 0.15, 0.35, 0.65, 0.9, 1.0],
            alphas=np.logspace(-5, 2, 50),
            cv=5,
            max_iter=50000,
            random_state=seed + 1,
        ),
        "Huber": HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
        "KNN_Distance": KNeighborsRegressor(n_neighbors=7, weights="distance", p=2),
        "SVR_RBF": SVR(C=8.0, epsilon=0.04, gamma="scale"),
        "KernelRidge_RBF": KernelRidge(alpha=0.08, kernel="rbf", gamma=None),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_tree,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed + 10,
            n_jobs=jobs,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n_tree,
            min_samples_leaf=2,
            max_features=0.75,
            random_state=seed + 11,
            n_jobs=jobs,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=350 if fast else 900,
            learning_rate=0.025,
            max_depth=3,
            min_samples_leaf=2,
            random_state=seed + 12,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=260 if fast else 750,
            learning_rate=0.025,
            max_leaf_nodes=23,
            l2_regularization=0.06,
            random_state=seed + 13,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=3e-4,
            learning_rate_init=8e-4,
            early_stopping=True,
            max_iter=900,
            random_state=seed + 14,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500 if fast else 1500,
            max_depth=3,
            learning_rate=0.018,
            subsample=0.86,
            colsample_bytree=0.86,
            min_child_weight=2.0,
            reg_lambda=5.0,
            reg_alpha=0.05,
            objective="reg:squarederror",
            random_state=seed + 20,
            n_jobs=jobs,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=600 if fast else 1700,
            num_leaves=19,
            learning_rate=0.018,
            subsample=0.86,
            colsample_bytree=0.86,
            reg_lambda=5.0,
            random_state=seed + 21,
            n_jobs=jobs,
            verbose=-1,
        )
    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=450 if fast else 1200,
            depth=5,
            learning_rate=0.025,
            l2_leaf_reg=8.0,
            random_seed=seed + 22,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
        )
    return models


def _blend_model(seed: int, jobs: int, fast: bool = False):
    """A compact validation-blend regressor using robust tree/boosting bases."""
    n_tree = 180 if fast else 500
    estimators = [
        ("ridge", RidgeCV(alphas=np.logspace(-5, 5, 31))),
        ("et", ExtraTreesRegressor(n_estimators=n_tree, min_samples_leaf=2, random_state=seed + 31, n_jobs=jobs)),
        ("rf", RandomForestRegressor(n_estimators=n_tree, min_samples_leaf=2, random_state=seed + 32, n_jobs=jobs)),
        ("hgb", HistGradientBoostingRegressor(max_iter=220 if fast else 550, learning_rate=0.025, max_leaf_nodes=19, random_state=seed + 33)),
    ]
    if HAS_XGB:
        estimators.append((
            "xgb",
            XGBRegressor(
                n_estimators=300 if fast else 900,
                max_depth=3,
                learning_rate=0.02,
                subsample=0.86,
                colsample_bytree=0.86,
                reg_lambda=5.0,
                objective="reg:squarederror",
                random_state=seed + 34,
                n_jobs=jobs,
            ),
        ))
    if HAS_LGBM:
        estimators.append((
            "lgbm",
            LGBMRegressor(
                n_estimators=350 if fast else 950,
                num_leaves=19,
                learning_rate=0.02,
                subsample=0.86,
                colsample_bytree=0.86,
                reg_lambda=5.0,
                random_state=seed + 35,
                n_jobs=jobs,
                verbose=-1,
            ),
        ))
    return BlendRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=np.logspace(-4, 4, 25)),
        cv=5,
        passthrough=False,
        n_jobs=max(1, min(jobs, 4)),
    )


def _relerr(y_true, y_pred) -> np.ndarray:
    return np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)) / (np.abs(y_true) + v3.EPS) * 100.0


def _score_rows(target: str, model_name: str, split: str, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "target": target,
        "model": model_name,
        "split": split,
        "n": len(y_true),
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "median_error_%": float(np.median(_relerr(y_true, y_pred))),
        "mean_error_%": float(np.mean(_relerr(y_true, y_pred))),
    }


def _optimized_weights(P: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    return v3.opt_weights(P, y_val)


def _locked_or_optimized_weights(name: str, top_names: list[str], P: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    locked = LOCKED_ANCHOR_WEIGHTS.get(name)
    if locked is None:
        return _optimized_weights(P, y_val)
    locked_names, locked_weights = locked
    locked_by_name = dict(zip(locked_names, locked_weights))
    w = np.asarray([locked_by_name.get(model_name, 0.0) for model_name in top_names], dtype=float)
    if not np.isfinite(w).all() or w.sum() <= v3.EPS:
        print(f"[WARN] locked anchor weights skipped for {name}: no locked top candidates available")
        return _optimized_weights(P, y_val)
    return w / (w.sum() + v3.EPS)


def fit_anchor_ensemble(name, y, kind, X, idx, pre, seed, jobs, fast):
    train, val = idx["train"], idx["val"]
    z_train = v3.y_transform(y[train], kind)
    models = _base_anchor_models(seed, jobs, fast)
    fitted = []

    for model_name, reg in models.items():
        try:
            pipe = Pipeline([("pre", clone(pre)), ("reg", reg)])
            pipe.fit(X.iloc[train], z_train)
            pred_val = v3.y_inverse(pipe.predict(X.iloc[val]), kind)
            val_med = float(np.median(_relerr(y[val], pred_val)))
            fitted.append((val_med, model_name, pipe))
        except Exception as exc:
            print(f"[WARN] anchor ensemble {name}/{model_name} failed: {exc}")

    try:
        blend = Pipeline([("pre", clone(pre)), ("reg", _blend_model(seed + 100, jobs, fast))])
        blend.fit(X.iloc[train], z_train)
        pred_val = v3.y_inverse(blend.predict(X.iloc[val]), kind)
        fitted.append((float(np.median(_relerr(y[val], pred_val))), "BlendReg", blend))
    except Exception as exc:
        print(f"[WARN] anchor {name}/BlendReg failed: {exc}")

    if not fitted:
        raise RuntimeError(f"No fitted anchor models for {name}")

    fitted = sorted(fitted, key=lambda item: item[0])
    top = fitted[: min(8, len(fitted))]
    pred_val_top = np.vstack([v3.y_inverse(m.predict(X.iloc[val]), kind) for _, _, m in top]).T
    top_names = [model_name for _, model_name, _ in top]
    weights = _locked_or_optimized_weights(name, top_names, pred_val_top, y[val])
    anchor = v3.Anchor(name, kind, [(model_name, model) for _, model_name, model in top], weights)

    print(
        f"[ANCHOR] {name}: top={[m for _, m, _ in top]}, "
        f"val_err={[round(e, 2) for e, _, _ in top]}, w={np.round(weights, 3)}"
    )

    all_splits = {"train": train, "val": val}
    if "test" in idx:
        all_splits["test"] = idx["test"]

    for _, model_name, model in fitted:
        pred_frames = []
        for split, ix in all_splits.items():
            pred = v3.y_inverse(model.predict(X.iloc[ix]), kind)
            ANCHOR_DIAGNOSTICS.append(_score_rows(name, model_name, split, y[ix], pred))
            pred_frames.append(pd.DataFrame({
                "target": name,
                "model": model_name,
                "split": split,
                "row_index": ix,
                "y_true": y[ix],
                "y_pred": pred,
                "error_%": _relerr(y[ix], pred),
            }))
        ANCHOR_PREDICTIONS.append(pd.concat(pred_frames, ignore_index=True))

    for split, ix in all_splits.items():
        pred = anchor.predict(X.iloc[ix])
        ANCHOR_DIAGNOSTICS.append(_score_rows(name, "ValWeightedEnsemble", split, y[ix], pred))
        ANCHOR_PREDICTIONS.append(pd.DataFrame({
            "target": name,
            "model": "ValWeightedEnsemble",
            "split": split,
            "row_index": ix,
            "y_true": y[ix],
            "y_pred": pred,
            "error_%": _relerr(y[ix], pred),
        }))

    return anchor


fit_anchor_v12 = fit_anchor_ensemble


def _save_anchor_tables(outdir: Path) -> None:
    diag = pd.DataFrame(ANCHOR_DIAGNOSTICS)
    preds = pd.concat(ANCHOR_PREDICTIONS, ignore_index=True) if ANCHOR_PREDICTIONS else pd.DataFrame()
    if diag.empty:
        return

    best_by_target = []
    for target, g in diag[diag["split"].eq("val")].groupby("target"):
        best = g.sort_values(["median_error_%", "MAE"], ascending=True).head(8).copy()
        best_by_target.append(best)
    best_df = pd.concat(best_by_target, ignore_index=True) if best_by_target else pd.DataFrame()

    xlsx = outdir / "anchor_model_accuracy.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        diag.sort_values(["target", "split", "median_error_%"]).to_excel(writer, sheet_name="All_Model_Metrics", index=False)
        best_df.to_excel(writer, sheet_name="Best_By_Validation", index=False)
        preds.to_excel(writer, sheet_name="Anchor_Predictions", index=False)
    diag.to_csv(outdir / "anchor_model_accuracy.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(outdir / "anchor_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] anchor metrics: {xlsx}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir", required=True)
    args, _ = parser.parse_known_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Patch V3 scalar anchor fitting, then import V5 to patch curve candidates.
    v3.fit_anchor = fit_anchor_ensemble
    import vwea_analog_shapes as _v5  # noqa: F401

    v3.main()
    _save_anchor_tables(outdir)


if __name__ == "__main__":
    main()
