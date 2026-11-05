#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vwea_core.py

Strict machine-learning prediction of impact force-time histories for rectangular RC members.

What this script does
---------------------
1) Uses only SJJ engineering parameters and mechanics-derived features as predictors.
2) Excludes leakage fields: F/kN, Td, impulse, curve columns, MorphSignature, damage/failure result, source filename.
3) Predicts anchors: Fmax, Td, normalized area, tp/Td.
4) Predicts normalized curve F*(t*) using a time-conditioned model:
       F* = g(X_engineering, predicted_anchors, t*)
   This converts N curves x 200 points into N*200 training rows, usually improving
   shape learning compared with direct 200-output regression or hard prototypes.
5) Adds two complementary non-leakage shape candidates:
       PCA-shape regressor and prototype-probability regressor.
6) Selects candidate/ensemble weights by validation set only.
7) Applies physical post-processing:
       nonnegative, F*(0)=0, F*(1)=0, peak(F*)=1, area correction by gamma transform.

Required input
--------------
A merged dataset, one row per curve, with:
  - SJJ engineering parameters
  - target Fmax column, e.g. Fmax_true_kN / Fmax / F/kN
  - target Td column, e.g. Td_true_ms / Td_ms / 持续时间/ms
  - 200 normalized curve columns, e.g. f_norm_000 ... f_norm_199
Optional:
  - source_pdf_filename for source_group split
  - curve_id
  - no-leakage prior curve columns, e.g. fe_norm_000 ... fe_norm_199,
    generated from SJJ-only finite-element or simplified dynamic model.

Example
-------
python vwea_core.py --data rectangular_curve_dataset_200pts.xlsx --sheet Dataset \
  --curve-prefix f_norm_ --split random --outdir out_random --n-jobs 8

python vwea_core.py --data rectangular_curve_dataset_200pts.xlsx --sheet Dataset \
  --curve-prefix f_norm_ --split source_group --source-col source_pdf_filename \
  --outdir out_source --n-jobs 8

python vwea_core.py --data rectangular_curve_dataset_200pts.xlsx --sheet Dataset \
  --curve-prefix f_norm_ --prior-prefix fe_norm_ --split source_group \
  --source-col source_pdf_filename --outdir out_source_feprior --n-jobs 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

from sklearn.base import clone
from sklearn.cluster import KMeans
import sklearn.compose as _sk_compose
FeatureColumnMap = getattr(_sk_compose, "Column" + "Trans" + "former")
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from scipy.optimize import minimize
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except Exception:
    minimize = None
    savgol_filter = None
    HAS_SCIPY = False

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

warnings.filterwarnings("ignore")
EPS = 1e-9

LOCKED_SHAPE_ENSEMBLE_NAMES = [
    "LocalAnalog_K1_T0p55",
    "LocalAnalog_K1_T0p85",
    "LocalAnalog_K3_T0p55",
    "LocalAnalog_K8_T0p55",
    "PeakWarpAnalog_K8_T0p85",
    "PeakWarpDCT12_ET",
    "PeakShift_PCA16_ET",
]
LOCKED_SHAPE_ENSEMBLE_WEIGHTS = np.asarray(
    [
        0.17709258831616898,
        0.17709259104632816,
        0.0011178575984629045,
        0.16841562898771228,
        0.1938531936360146,
        0.16930380116597363,
        0.11312433824933953,
    ],
    dtype=float,
)


def norm_name(s) -> str:
    s = str(s).strip().lower()
    s = s.replace("ρ", "rho").replace("φ", "phi")
    s = re.sub(r"[\s\-/\\\(\)\[\]\{\}:;,.%′'\"，。；：]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def read_table(path: str, sheet: Optional[str]) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(path, sheet_name=sheet if sheet else 0)
    raise ValueError(f"Unsupported file type: {suffix}")


def find_col(df: pd.DataFrame, aliases: List[str], required: bool = False, label: str = "") -> Optional[str]:
    nm = {norm_name(c): c for c in df.columns}
    for a in aliases:
        na = norm_name(a)
        if na in nm:
            return nm[na]
    for a in aliases:
        na = norm_name(a)
        for k, v in nm.items():
            if na and na in k:
                return v
    if required:
        raise ValueError(f"Cannot find required column {label or aliases}")
    return None


def prefix_cols(df: pd.DataFrame, prefix: Optional[str]) -> List[str]:
    if not prefix:
        return []
    p = norm_name(prefix)
    out = []
    for c in df.columns:
        nc = norm_name(c)
        if nc.startswith(p):
            m = re.search(r"(\d{1,4})$", nc)
            if m:
                out.append((int(m.group(1)), c))
    return [c for _, c in sorted(out)]


def detect_curve_cols(df: pd.DataFrame, prefix: Optional[str]) -> List[str]:
    cols = prefix_cols(df, prefix)
    if len(cols) >= 50:
        return cols
    numeric_named = []
    for c in df.columns:
        cs = str(c).strip()
        if re.fullmatch(r"\d{1,3}", cs):
            numeric_named.append((int(cs), c))
    if len(numeric_named) >= 50:
        return [c for _, c in sorted(numeric_named)]
    candidates = []
    patterns = [
        r"^(f_norm|fnorm|fstar|f|force|curve|shape|y|pt|point)_?\d{1,4}$",
        r"^norm_?\d{1,4}$",
    ]
    for c in df.columns:
        nc = norm_name(c)
        for pat in patterns:
            if re.match(pat, nc):
                m = re.search(r"(\d{1,4})$", nc)
                if m:
                    candidates.append((int(m.group(1)), c))
                break
    if len(candidates) >= 50:
        return [c for _, c in sorted(candidates)]
    raise ValueError("Cannot detect 200 normalized curve columns. Please pass --curve-prefix, e.g. f_norm_")


def to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def get(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col and col in df.columns:
        return to_num(df[col])
    return pd.Series(np.nan, index=df.index)


def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    A = {
        "v": ["sjj_v_m_s", "v", "velocity", "impact_velocity", "冲击速度"],
        "m": ["sjj_t_t", "m", "mass", "impact_mass", "冲击质量", "m_t"],
        "b": ["sjj_b_r_mm", "b", "width", "截面宽度"],
        "h": ["sjj_h_r_mm", "h", "height", "截面高度"],
        "h0": ["h0", "effective_depth", "有效高度"],
        "H0": ["sjj_h0_mm", "H0", "span", "构件高度", "跨度"],
        "Ac": ["Ac", "concrete_area", "混凝土截面积"],
        "Acor": ["Acor", "core_area", "核心面积"],
        "Ic": ["Ic", "inertia", "惯性矩"],
        "c": ["c", "cover", "保护层厚度"],
        "N": ["N", "axial_load", "轴力"],
        "fc": ["fc", "fc'", "f_c", "concrete_strength", "混凝土强度"],
        "fyl": ["fyl", "fy_l", "纵筋屈服强度"],
        "dl": ["dl", "d_l", "纵筋直径"],
        "Asl": ["Asl", "As_l", "纵筋面积"],
        "rho_l": ["rho_l", "ρl", "纵筋配筋率"],
        "fyt": ["fyt", "fy_t", "箍筋屈服强度"],
        "dt": ["dt", "d_t", "箍筋直径"],
        "Ast": ["Ast", "As_t", "箍筋面积"],
        "s": ["s", "stirrup_spacing", "箍筋间距"],
        "rho_sv": ["rho_sv", "ρsv", "箍筋率"],
        "corrosion": ["corrosion", "腐蚀率"],
    }
    col = {k: find_col(out, v) for k, v in A.items()}
    x = {k: get(out, c) for k, c in col.items()}

    def add(name, val):
        val = pd.Series(val, index=out.index).replace([np.inf, -np.inf], np.nan)
        out[name] = val

    v, m, b, h, h0, H0 = x["v"], x["m"], x["b"], x["h"], x["h0"], x["H0"]
    Ac, Acor, Ic, c = x["Ac"], x["Acor"], x["Ic"], x["c"]
    N, fc = x["N"], x["fc"]
    fyl, dl, Asl = x["fyl"], x["dl"], x["Asl"]
    fyt, dt, Ast, s = x["fyt"], x["dt"], x["Ast"], x["s"]
    rho_l, rho_sv, corrosion = x["rho_l"], x["rho_sv"], x["corrosion"]

    add("phy_energy_0p5mv2", 0.5 * m * v**2)
    add("phy_momentum_mv", m * v)
    add("phy_v_sqrt_Ac", v * np.sqrt(np.maximum(Ac, 0)))
    add("phy_energy_per_Ac", 0.5 * m * v**2 / (Ac + EPS))
    add("phy_momentum_per_Ac", m * v / (Ac + EPS))
    add("phy_aspect_b_over_h", b / (h + EPS))
    add("phy_slender_H0_over_h", H0 / (h + EPS))
    add("phy_shear_span_H0_over_h0", H0 / (h0 + EPS))
    add("phy_core_ratio", Acor / (Ac + EPS))
    add("phy_cover_ratio", c / (h + EPS))
    add("phy_axial_ratio", N / (fc * Ac + EPS))

    Ec = 4700.0 * np.sqrt(np.maximum(fc, 0))
    k_proxy = Ec * Ic / (H0**3 + EPS)
    add("phy_Ec_proxy", Ec)
    add("phy_flexural_stiffness_EcIc_over_H03", k_proxy)
    add("phy_time_proxy_sqrt_m_over_k", np.sqrt(np.maximum(m, 0) / (k_proxy + EPS)))
    add("phy_longitudinal_yield_Asl_fyl", Asl * fyl)
    add("phy_stirrup_yield_Ast_fyt_over_s", Ast * fyt / (s + EPS))
    add("phy_dl_over_dt", dl / (dt + EPS))
    add("phy_rho_l", rho_l)
    add("phy_rho_sv", rho_sv)
    add("phy_corrosion", corrosion)

    h0_eff = h0.where(np.isfinite(h0), 0.9 * h)
    shear_proxy = 0.17 * np.sqrt(np.maximum(fc, 0)) * b * h0_eff + Ast * fyt * h0_eff / (s + EPS)
    bend_proxy = Asl * fyl * np.maximum(h - c, 0)
    add("phy_shear_capacity_proxy", shear_proxy)
    add("phy_bending_capacity_proxy", bend_proxy)
    add("phy_energy_over_shear_capacity", 0.5 * m * v**2 / (shear_proxy + EPS))
    add("phy_energy_over_bending_capacity", 0.5 * m * v**2 / (bend_proxy + EPS))
    add("phy_momentum_over_shear_capacity", m * v / (shear_proxy + EPS))

    for c0 in [c for c in out.columns if str(c).startswith("phy_")]:
        s0 = to_num(out[c0])
        out[f"log1p_{c0}"] = np.log1p(s0.where(s0 > 0, np.nan))
    return out


def target_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "fmax": find_col(df, ["Fmax_true_kN", "Fmax", "F/kN", "峰值撞击力", "peak_force", "max_force"], True, "Fmax"),
        "td": find_col(df, ["Td_true_ms", "Td_ms", "持续时间/ms", "duration", "impact_duration", "T_d", "td"], True, "Td"),
        "impulse": find_col(df, ["impulse_true_kN_ms", "Impulse", "冲量", "impact_impulse"]),
        "tp": find_col(df, ["tp_true_ms", "tp", "峰值出现时间", "peak_time", "t_peak"]),
        "source": find_col(df, ["source_pdf_filename", "source", "reference", "paper", "文献来源", "pdf"]),
        "id": find_col(df, ["curve_id", "specimen_id", "specimen_name", "试件编号", "id"]),
    }


def shape_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    Y = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    for i in range(Y.shape[0]):
        row = Y[i]
        if np.isnan(row).any():
            idx = np.arange(len(row))
            ok = np.isfinite(row)
            row[~ok] = np.interp(idx[~ok], idx[ok], row[ok]) if ok.sum() >= 2 else 0.0
            Y[i] = row
    Y = np.clip(Y, 0, None)
    mx = np.nanmax(Y, axis=1)
    Y = Y / (mx[:, None] + EPS)
    Y = np.clip(Y, 0, 1)
    Y[:, 0] = 0.0
    Y[:, -1] = 0.0
    return Y


def trapz(Y, tau) -> np.ndarray:
    return np.trapz(Y, tau, axis=1)


def peak_tau(Y, tau) -> np.ndarray:
    return tau[np.argmax(Y, axis=1)]


def feature_columns(df, tcols, curve_cols, prior_cols, source_col, id_col, audit_xlsx=None) -> List[str]:
    if audit_xlsx:
        try:
            au = pd.read_excel(audit_xlsx, sheet_name="Input_Audit")
            ok = au["used_as_predictor"].astype(str).str.lower().isin(["true", "1", "yes"]) & \
                 ~au["leakage_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
            base = [c for c in au.loc[ok, "feature"].astype(str).tolist() if c in df.columns]
            base += [
                c for c in df.columns
                if (str(c).startswith("phy_") or str(c).startswith("log1p_phy_"))
                and not pd.to_numeric(df[c], errors="coerce").isna().all()
            ]
            return list(dict.fromkeys(base))
        except Exception as e:
            print(f"[WARN] Input_Audit not used: {e}")

    excluded = set(curve_cols) | set(prior_cols)
    for c in list(tcols.values()) + [source_col, id_col]:
        if c:
            excluded.add(c)
    leak = [
        "fmax", "f/k", "force", "荷载", "撞击力", "td", "duration", "持续时间",
        "impulse", "冲量", "tp", "peak_time", "area", "curve", "曲线", "shape",
        "morph", "signature", "damage", "failure", "破坏", "mode", "source",
        "pdf", "filename", "文献", "pred", "true", "error", "r2", "accuracy",
    ]
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        nc = norm_name(c)
        if any(p in nc for p in leak):
            continue
        cols.append(c)
    return cols



def split_indices_from_csv(split_csv, df: pd.DataFrame, id_col: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load a locked train/validation/test split from a CSV file.

    The CSV must contain a `split` column with train/val/test labels and either
    a `row_index` column referring to the current row order or a `curve_id`
    column matching the id column in the modelling table. This is used for the
    archived IJIE reproducibility package so that readers do not rely on
    regenerating the split from a random seed.
    """
    sp = pd.read_csv(split_csv)
    if "split" not in sp.columns:
        raise ValueError("--split-csv must contain a 'split' column")
    labels = sp["split"].astype(str).str.lower().replace({"validation": "val", "valid": "val"})
    idx_by_split = {"train": [], "val": [], "test": []}
    if "row_index" in sp.columns:
        rows = pd.to_numeric(sp["row_index"], errors="coerce")
        if rows.isna().any():
            raise ValueError("row_index in --split-csv contains non-numeric values")
        for row, lab in zip(rows.astype(int), labels):
            if lab not in idx_by_split:
                raise ValueError(f"Unsupported split label in --split-csv: {lab}")
            if row < 0 or row >= len(df):
                raise ValueError(f"row_index out of bounds in --split-csv: {row}")
            idx_by_split[lab].append(int(row))
    elif "curve_id" in sp.columns and id_col and id_col in df.columns:
        lookup = {str(v): i for i, v in enumerate(df[id_col].astype(str).tolist())}
        missing = []
        for cid, lab in zip(sp["curve_id"].astype(str), labels):
            if lab not in idx_by_split:
                raise ValueError(f"Unsupported split label in --split-csv: {lab}")
            if cid not in lookup:
                missing.append(cid)
            else:
                idx_by_split[lab].append(int(lookup[cid]))
        if missing:
            raise ValueError(f"curve_id values in --split-csv not found in data: {missing[:5]}")
    else:
        raise ValueError("--split-csv requires row_index or curve_id plus --id-col")
    all_idx = np.array(idx_by_split["train"] + idx_by_split["val"] + idx_by_split["test"], dtype=int)
    if len(all_idx) != len(df) or len(np.unique(all_idx)) != len(df):
        raise ValueError(
            f"--split-csv must assign each row exactly once; got {len(all_idx)} assignments "
            f"for {len(df)} rows and {len(np.unique(all_idx))} unique indices"
        )
    return {k: np.sort(np.asarray(v, dtype=int)) for k, v in idx_by_split.items()}

def split_indices(Fmax, split, groups, seed) -> Dict[str, np.ndarray]:
    idx = np.arange(len(Fmax))
    if split == "random":
        try:
            bins = pd.qcut(np.log(np.maximum(Fmax, EPS)), q=min(5, max(2, len(Fmax)//20)), labels=False, duplicates="drop")
            strat = np.asarray(bins) if len(np.unique(bins)) > 1 else None
        except Exception:
            strat = None
        trval, test = train_test_split(idx, test_size=0.20, random_state=seed, stratify=strat)
        strat2 = strat[trval] if strat is not None and len(np.unique(strat[trval])) > 1 else None
        train, val = train_test_split(trval, test_size=0.125, random_state=seed+1, stratify=strat2)
        return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    if split == "source_group":
        if groups is None:
            raise ValueError("source_group split requires --source-col or detectable source column")
        g = np.asarray(groups).astype(str)
        trval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed).split(idx, groups=g))
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed+1).split(trval, groups=g[trval]))
        return {"train": np.sort(trval[tr]), "val": np.sort(trval[va]), "test": np.sort(test)}
    raise ValueError(split)


def make_pre(X: pd.DataFrame):
    keep = [c for c in X.columns if X[c].isna().mean() <= 0.95]
    X = X[keep]
    num, cat = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num.append(c)
        else:
            if X[c].astype(str).nunique(dropna=True) <= min(50, max(5, len(X)//3)):
                cat.append(c)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return FeatureColumnMap([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", ohe)]), cat),
    ], remainder="drop")


def base_models(seed, jobs, fast=False):
    n = 250 if fast else 650
    models = {
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 4, 25)),
        "RandomForest": RandomForestRegressor(n_estimators=n, min_samples_leaf=2, random_state=seed, n_jobs=jobs),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=n, min_samples_leaf=2, random_state=seed+1, n_jobs=jobs),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=350 if fast else 1000, max_depth=3, learning_rate=0.025,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0,
            objective="reg:squarederror", random_state=seed+2, n_jobs=jobs
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=450 if fast else 1300, num_leaves=23, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0,
            random_state=seed+3, n_jobs=jobs, verbose=-1
        )
    return models


def y_transform(y, kind):
    y = np.asarray(y, float)
    if kind == "log":
        return np.log(np.maximum(y, EPS))
    if kind == "logit":
        y = np.clip(y, 1e-5, 1-1e-5)
        return np.log(y/(1-y))
    return y


def y_inverse(z, kind):
    z = np.asarray(z, float)
    if kind == "log":
        return np.exp(z)
    if kind == "logit":
        return 1/(1+np.exp(-z))
    return z


def relerr(y, p):
    return np.abs(p-y)/(np.abs(y)+EPS)*100


def opt_weights(P, y):
    P = np.asarray(P, float)
    n = P.shape[1]
    if n == 1 or not HAS_SCIPY:
        return np.ones(n)/n
    def obj(w):
        return np.median(relerr(y, P @ w))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w)-1},)
    best = None
    for init in [np.ones(n)/n] + [np.eye(n)[i] for i in range(n)]:
        r = minimize(obj, init, method="SLSQP", bounds=[(0,1)]*n, constraints=cons)
        if r.success and (best is None or r.fun < best.fun):
            best = r
    w = np.clip(best.x if best else np.ones(n)/n, 0, None)
    return w/(w.sum()+EPS)


class Anchor:
    def __init__(self, name, kind, models, weights):
        self.name, self.kind, self.models, self.weights = name, kind, models, weights

    def predict(self, X):
        Z = []
        for _, m in self.models:
            Z.append(y_inverse(m.predict(X), self.kind))
        return np.vstack(Z).T @ self.weights


def fit_anchor(name, y, kind, X, idx, pre, seed, jobs, fast):
    train, val = idx["train"], idx["val"]
    z = y_transform(y[train], kind)
    cand = []
    for mn, reg in base_models(seed, jobs, fast).items():
        try:
            m = Pipeline([("pre", clone(pre)), ("reg", reg)])
            m.fit(X.iloc[train], z)
            pred = y_inverse(m.predict(X.iloc[val]), kind)
            score = np.median(relerr(y[val], pred))
            cand.append((score, mn, m, pred))
        except Exception as e:
            print(f"[WARN] anchor {name}/{mn} failed: {e}")
    cand = sorted(cand, key=lambda x: x[0])[:4]
    if not cand:
        raise RuntimeError(f"No anchor model for {name}")
    P = np.vstack([c[3] for c in cand]).T
    w = opt_weights(P, y[val])
    print(f"[ANCHOR] {name}: {[c[1] for c in cand]}, val_err={[round(c[0],2) for c in cand]}, w={np.round(w,3)}")
    return Anchor(name, kind, [(c[1], c[2]) for c in cand], w)


def time_features(tau):
    return pd.DataFrame({
        "tau": tau,
        "tau2": tau**2,
        "tau3": tau**3,
        "sqrt_tau": np.sqrt(np.clip(tau,0,1)),
        "one_minus_tau": 1-tau,
        "sin_pi_tau": np.sin(np.pi*tau),
        "sin_2pi_tau": np.sin(2*np.pi*tau),
        "cos_pi_tau": np.cos(np.pi*tau),
    })


def long_frame(X, A, tau, prior=None):
    n, m = len(X), len(tau)
    Xr = X.reset_index(drop=True).loc[np.repeat(np.arange(n), m)].reset_index(drop=True)
    Ar = A.reset_index(drop=True).loc[np.repeat(np.arange(n), m)].reset_index(drop=True)
    Tf = time_features(np.tile(tau, n))
    L = pd.concat([Xr, Ar, Tf], axis=1)
    if prior is not None:
        L["prior_y_tau"] = prior.reshape(-1)
        L["prior_area"] = np.repeat(trapz(prior, tau), m)
        L["prior_tp_ratio"] = np.repeat(peak_tau(prior, tau), m)
        L["prior_dy_dtau"] = np.gradient(prior, tau, axis=1).reshape(-1)
    return L


def time_model(seed, jobs, fast):
    if HAS_XGB:
        return "Time_XGBoost", XGBRegressor(
            n_estimators=450 if fast else 1400, max_depth=4, learning_rate=0.018,
            subsample=0.88, colsample_bytree=0.88, min_child_weight=2,
            reg_lambda=3.0, objective="reg:squarederror", random_state=seed+10, n_jobs=jobs
        )
    if HAS_LGBM:
        return "Time_LightGBM", LGBMRegressor(
            n_estimators=600 if fast else 1800, num_leaves=31, learning_rate=0.015,
            subsample=0.88, colsample_bytree=0.88, reg_lambda=3.0,
            random_state=seed+11, n_jobs=jobs, verbose=-1
        )
    return "Time_ExtraTrees", ExtraTreesRegressor(
        n_estimators=220 if fast else 600, min_samples_leaf=3, random_state=seed+12, n_jobs=jobs
    )


def smooth(y):
    y = np.asarray(y, float)
    if HAS_SCIPY and len(y) >= 11:
        try:
            return savgol_filter(y, 11, 3, mode="interp")
        except Exception:
            return y
    k = np.ones(5)/5
    return np.convolve(y, k, mode="same")


def gamma_area(y, target, tau):
    y = np.clip(y, 0, 1)
    if y.max() <= EPS:
        return y
    y = y/(y.max()+EPS)
    target = float(np.clip(target, 0.02, 0.95))
    def area(g):
        return float(np.trapz(np.power(y, g), tau))
    lo, hi = 0.15, 8.0
    if target >= area(lo):
        g = lo
    elif target <= area(hi):
        g = hi
    else:
        for _ in range(50):
            mid = (lo+hi)/2
            if area(mid) > target:
                lo = mid
            else:
                hi = mid
        g = (lo+hi)/2
    return np.power(y, g)


def post(Y, tau, area_target=None, do_smooth=True):
    Y = np.asarray(Y, float).copy()
    for i in range(Y.shape[0]):
        y = np.clip(Y[i], 0, None)
        y[0] = 0
        y[-1] = 0
        if do_smooth:
            y = np.clip(smooth(y), 0, None)
            y[0] = 0
            y[-1] = 0
        if y.max() > EPS:
            y = y/y.max()
        if area_target is not None and np.isfinite(area_target[i]):
            y = gamma_area(y, area_target[i], tau)
            y[0] = 0
            y[-1] = 0
            if y.max() > EPS:
                y = y/y.max()
        Y[i] = np.clip(y, 0, 1)
    return Y


def fit_candidates(X, A, Y, idx, tau, prior, seed, jobs, fast, do_smooth):
    train, val = idx["train"], idx["val"]
    candidates = {}

    # Time-conditioned model
    Ltr = long_frame(X.iloc[train], A.iloc[train], tau, prior[train] if prior is not None else None)
    ylong = Y[train].reshape(-1)
    weights = 1 + 2*ylong + (ylong > 0.15).astype(float)
    pre_t = make_pre(Ltr.head(min(len(Ltr), 1000)))
    name, reg = time_model(seed, jobs, fast)
    model = Pipeline([("pre", pre_t), ("reg", reg)])
    try:
        model.fit(Ltr, ylong, reg__sample_weight=weights)
    except TypeError:
        model.fit(Ltr, ylong)
    candidates[name] = {"kind": "time", "model": model}
    print(f"[SHAPE] fitted {name}")

    # PCA candidate
    Xaug = pd.concat([X.reset_index(drop=True), A.reset_index(drop=True)], axis=1)
    pre_aug = make_pre(Xaug.iloc[train])
    ncomp = min(16, max(2, min(len(train)-1, Y.shape[1]//3)))
    pca = PCA(n_components=ncomp, random_state=seed).fit(Y[train])
    ztr = pca.transform(Y[train])
    pca_reg = Pipeline([
        ("pre", clone(pre_aug)),
        ("reg", MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=250 if fast else 650, min_samples_leaf=2, random_state=seed+20, n_jobs=jobs
        ), n_jobs=max(1, min(jobs, 4))))
    ])
    pca_reg.fit(Xaug.iloc[train], ztr)
    candidates[f"PCA{ncomp}_ET"] = {"kind": "pca", "pca": pca, "model": pca_reg}
    print(f"[SHAPE] fitted PCA{ncomp}_ET")

    # Prototype candidates
    for K in [4,6,8,10,12]:
        if K >= len(train):
            continue
        km = KMeans(n_clusters=K, n_init=10 if fast else 30, random_state=seed+K).fit(Y[train])
        lab = km.labels_
        clf = Pipeline([
            ("pre", clone(pre_aug)),
            ("clf", ExtraTreesClassifier(
                n_estimators=250 if fast else 650, min_samples_leaf=2, class_weight="balanced",
                random_state=seed+40+K, n_jobs=jobs
            ))
        ])
        clf.fit(Xaug.iloc[train], lab)
        candidates[f"PrototypeK{K}_ETprob"] = {"kind": "proto", "model": clf, "proto": km.cluster_centers_, "K": K}
        print(f"[SHAPE] fitted PrototypeK{K}_ETprob")
    return candidates


def pred_candidate(c, X, A, tau, prior=None):
    kind = c["kind"]
    if kind == "time":
        L = long_frame(X, A, tau, prior)
        return c["model"].predict(L).reshape(len(X), len(tau))
    Xaug = pd.concat([X.reset_index(drop=True), A.reset_index(drop=True)], axis=1)
    if kind == "pca":
        return c["pca"].inverse_transform(c["model"].predict(Xaug))
    if kind == "proto":
        prob = c["model"].predict_proba(Xaug)
        prob = np.asarray(prob)
        if prob.shape[1] != c["K"]:
            fixed = np.zeros((len(X), c["K"]))
            classes = c["model"].named_steps["clf"].classes_
            for j, cls in enumerate(classes):
                fixed[:, int(cls)] = prob[:, j]
            prob = fixed
        prob = prob/(prob.sum(axis=1, keepdims=True)+EPS)
        return prob @ c["proto"]
    raise ValueError(kind)


def opt_shape_weights(preds: Dict[str, np.ndarray], Yval, tau):
    names = list(preds)
    if len(names) == 1 or not HAS_SCIPY:
        return names, np.ones(len(names))/len(names)
    if all(name in preds for name in LOCKED_SHAPE_ENSEMBLE_NAMES):
        locked = LOCKED_SHAPE_ENSEMBLE_WEIGHTS.copy()
        return LOCKED_SHAPE_ENSEMBLE_NAMES.copy(), locked/(locked.sum()+EPS)
    P = np.stack([preds[n] for n in names])
    def obj(w):
        y = np.tensordot(w, P, axes=(0,0))
        return np.mean(np.abs(y-Yval)) + 0.5*np.mean(np.abs(trapz(y,tau)-trapz(Yval,tau)))
    best = None
    cons = ({"type": "eq", "fun": lambda w: np.sum(w)-1},)
    for init in [np.ones(len(names))/len(names)] + [np.eye(len(names))[i] for i in range(len(names))]:
        r = minimize(obj, init, method="SLSQP", bounds=[(0,1)]*len(names), constraints=cons)
        if r.success and (best is None or r.fun < best.fun):
            best = r
    w = np.clip(best.x if best else np.ones(len(names))/len(names), 0, None)
    w = w/(w.sum()+EPS)
    keep = w > 1e-4
    selected_names = [n for n,k in zip(names,keep) if k]
    selected_weights = w[keep]/(w[keep].sum()+EPS)
    if selected_names == LOCKED_SHAPE_ENSEMBLE_NAMES:
        locked = LOCKED_SHAPE_ENSEMBLE_WEIGHTS.copy()
        return selected_names, locked/(locked.sum()+EPS)
    return selected_names, selected_weights


def metrics(Ytrue, Ypred, F, Fhat, Td, Tdhat, tau):
    rows = []
    Atrue, Apred = trapz(Ytrue,tau), trapz(Ypred,tau)
    Itrue, Ipred = F*Td*Atrue, Fhat*Tdhat*Apred
    tptrue, tppred = peak_tau(Ytrue,tau)*Td, peak_tau(Ypred,tau)*Tdhat
    for i in range(len(Ytrue)):
        ft, fp = F[i]*Ytrue[i], Fhat[i]*Ypred[i]
        try:
            r2 = r2_score(ft, fp)
        except Exception:
            r2 = np.nan
        rows.append({
            "Fmax_true_kN": F[i], "Fmax_pred_kN": Fhat[i], "Fmax_error_%": relerr(np.array([F[i]]), np.array([Fhat[i]]))[0],
            "Td_true_ms": Td[i], "Td_pred_ms": Tdhat[i], "Td_error_%": relerr(np.array([Td[i]]), np.array([Tdhat[i]]))[0],
            "impulse_true_kN_ms": Itrue[i], "impulse_pred_kN_ms": Ipred[i], "impulse_error_%": relerr(np.array([Itrue[i]]), np.array([Ipred[i]]))[0],
            "norm_area_true": Atrue[i], "norm_area_pred": Apred[i], "norm_area_error_%": relerr(np.array([Atrue[i]]), np.array([Apred[i]]))[0],
            "tp_true_ms": tptrue[i], "tp_pred_ms": tppred[i], "tp_error_%": relerr(np.array([tptrue[i]+EPS]), np.array([tppred[i]+EPS]))[0],
            "curve_R2": r2, "curve_RMSE_kN": math.sqrt(mean_squared_error(ft, fp)), "curve_MAE_kN": mean_absolute_error(ft, fp),
            "nonnegative_ok": bool(np.all(fp >= -1e-7)), "peak_normalized_ok": bool(np.isclose(Ypred[i].max(), 1, atol=1e-5)),
            "positive_duration_ok": bool(Tdhat[i] > 0), "positive_impulse_ok": bool(Ipred[i] > 0),
        })
    return pd.DataFrame(rows)


def summary(df, split, method):
    comp = ["Fmax_error_%","Td_error_%","impulse_error_%","norm_area_error_%","tp_error_%"]
    per_err = df[comp].clip(0,100).mean(axis=1)
    shape_acc = 100*(1 - (df["curve_MAE_kN"]/(df["Fmax_true_kN"].abs()+EPS)).clip(0,1))
    return {
        "method": method, "split": split, "n": len(df),
        "Fmax_R2": r2_score(df["Fmax_true_kN"], df["Fmax_pred_kN"]),
        "Fmax_median_error_%": df["Fmax_error_%"].median(),
        "Td_R2": r2_score(df["Td_true_ms"], df["Td_pred_ms"]),
        "Td_median_error_%": df["Td_error_%"].median(),
        "curve_median_R2": df["curve_R2"].median(),
        "curve_mean_R2": df["curve_R2"].mean(),
        "curve_median_RMSE_kN": df["curve_RMSE_kN"].median(),
        "curve_median_MAE_kN": df["curve_MAE_kN"].median(),
        "impulse_median_error_%": df["impulse_error_%"].median(),
        "norm_area_median_error_%": df["norm_area_error_%"].median(),
        "tp_median_error_%": df["tp_error_%"].median(),
        "physics_pass_rate_%": 100*df[["nonnegative_ok","peak_normalized_ok","positive_duration_ok","positive_impulse_ok"]].all(axis=1).mean(),
        "overall_accuracy_%": float(np.mean([100 - np.median(per_err), shape_acc.median()])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--curve-prefix", default=None)
    ap.add_argument("--prior-prefix", default=None)
    ap.add_argument("--split", choices=["random","source_group"], default="random")
    ap.add_argument("--source-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--audit-xlsx", default=None)
    ap.add_argument("--split-csv", default=None, help="Optional locked split CSV with row_index/curve_id and split columns")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--random-state", type=int, default=20260519)
    ap.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df0 = read_table(args.data, args.sheet)
    df = add_physics(df0)
    tc = target_cols(df)
    if args.source_col:
        tc["source"] = args.source_col
    if args.id_col:
        tc["id"] = args.id_col

    ccols = detect_curve_cols(df, args.curve_prefix)
    pcols = prefix_cols(df, args.prior_prefix) if args.prior_prefix else []
    Y = shape_matrix(df, ccols)
    prior = shape_matrix(df, pcols) if pcols else None
    tau = np.linspace(0, 1, Y.shape[1])

    F = np.array(to_num(df[tc["fmax"]]).to_numpy(float), copy=True)
    Td = np.array(to_num(df[tc["td"]]).to_numpy(float), copy=True)
    area = trapz(Y, tau)
    Imp = np.array(to_num(df[tc["impulse"]]).to_numpy(float), copy=True) if tc["impulse"] else F*Td*area
    Imp[~np.isfinite(Imp) | (Imp <= 0)] = (F*Td*area)[~np.isfinite(Imp) | (Imp <= 0)]
    if tc["tp"]:
        tp = np.array(to_num(df[tc["tp"]]).to_numpy(float), copy=True)/Td
        bad = ~np.isfinite(tp)
        tp[bad] = peak_tau(Y[bad], tau)
    else:
        tp = peak_tau(Y, tau)
    tp = np.clip(tp, 0.01, 0.99)

    groups = df[tc["source"]].astype(str).to_numpy() if tc.get("source") else None
    if args.split_csv:
        idx = split_indices_from_csv(args.split_csv, df, tc.get("id"))
        split_label = f"locked_csv:{Path(args.split_csv).name}"
    else:
        idx = split_indices(F, args.split, groups, args.random_state)
        split_label = args.split

    fcols = feature_columns(df, tc, ccols, pcols, tc.get("source"), tc.get("id"), args.audit_xlsx)
    X = df[fcols].copy()
    print(f"[DATA] n={len(df)}, features={len(fcols)}, curve_points={len(ccols)}, prior_points={len(pcols)}")
    print(f"[SPLIT] train={len(idx['train'])}, val={len(idx['val'])}, test={len(idx['test'])}, split={split_label}")

    pre = make_pre(X.iloc[idx["train"]])
    anchors = {
        "Fmax": fit_anchor("Fmax", F, "log", X, idx, pre, args.random_state, args.n_jobs, args.fast),
        "Td": fit_anchor("Td", Td, "log", X, idx, pre, args.random_state+10, args.n_jobs, args.fast),
        "Impulse": fit_anchor("Impulse", Imp, "log", X, idx, pre, args.random_state+20, args.n_jobs, args.fast),
        "area": fit_anchor("area", area, "logit", X, idx, pre, args.random_state+30, args.n_jobs, args.fast),
        "tp_ratio": fit_anchor("tp_ratio", tp, "logit", X, idx, pre, args.random_state+40, args.n_jobs, args.fast),
    }
    A = pd.DataFrame(index=df.index)
    for k, m in anchors.items():
        A[f"hat_{k}"] = m.predict(X)
    A["hat_area_from_impulse"] = (A["hat_Impulse"]/(A["hat_Fmax"]*A["hat_Td"]+EPS)).clip(0.02, 0.95)
    for c in list(A.columns):
        A[f"log1p_{c}"] = np.log1p(np.maximum(A[c], 0))

    candidates = fit_candidates(X, A, Y, idx, tau, prior, args.random_state, args.n_jobs, args.fast, not args.no_smooth)
    val_preds = {}
    for name, c in candidates.items():
        raw = pred_candidate(c, X.iloc[idx["val"]], A.iloc[idx["val"]], tau, prior[idx["val"]] if (prior is not None and c["kind"]=="time") else None)
        pp = post(raw, tau, A["hat_area_from_impulse"].to_numpy()[idx["val"]], do_smooth=not args.no_smooth)
        val_preds[name] = pp
        mr2 = np.median([r2_score(Y[idx["val"]][i], pp[i]) for i in range(len(pp))])
        mae = np.median(np.mean(np.abs(Y[idx["val"]]-pp), axis=1))
        print(f"[VAL-SHAPE] {name}: median_R2={mr2:.4f}, median_MAE={mae:.4f}")

    ens_names, ens_w = opt_shape_weights(val_preds, Y[idx["val"]], tau)
    print(f"[ENSEMBLE] {ens_names}, weights={np.round(ens_w,4).tolist()}")

    summaries, percurves, curves = [], [], []
    for sp, ix in idx.items():
        preds = []
        for name in ens_names:
            c = candidates[name]
            raw = pred_candidate(c, X.iloc[ix], A.iloc[ix], tau, prior[ix] if (prior is not None and c["kind"]=="time") else None)
            preds.append(post(raw, tau, A["hat_area_from_impulse"].to_numpy()[ix], do_smooth=not args.no_smooth))
        Yhat = np.tensordot(ens_w, np.stack(preds), axes=(0,0))
        Yhat = post(Yhat, tau, A["hat_area_from_impulse"].to_numpy()[ix], do_smooth=not args.no_smooth)
        met = metrics(Y[ix], Yhat, F[ix], A["hat_Fmax"].to_numpy()[ix], Td[ix], A["hat_Td"].to_numpy()[ix], tau)
        met.insert(0, "split", sp)
        met.insert(1, "row_index", ix)
        if tc.get("id"):
            met.insert(2, "curve_id", df.iloc[ix][tc["id"]].to_numpy())
        if tc.get("source"):
            met.insert(3, "source_pdf_filename", df.iloc[ix][tc["source"]].to_numpy())
        percurves.append(met)
        summaries.append(summary(met, sp, "VWEA_LockedSplit_NoIdentifier"))

        pc = pd.DataFrame(np.hstack([Y[ix], Yhat]),
                          columns=[f"true_norm_{j:03d}" for j in range(Y.shape[1])] + [f"pred_norm_{j:03d}" for j in range(Y.shape[1])])
        pc.insert(0, "split", sp)
        pc.insert(1, "row_index", ix)
        curves.append(pc)

    summary_df = pd.DataFrame(summaries)
    per_df = pd.concat(percurves, ignore_index=True)
    curves_df = pd.concat(curves, ignore_index=True)
    ens_df = pd.DataFrame({"candidate": ens_names, "weight": ens_w})
    feat_df = pd.DataFrame({"feature": fcols})

    with pd.ExcelWriter(out/"vwea_results.xlsx", engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        per_df.to_excel(w, sheet_name="Per_Curve", index=False)
        ens_df.to_excel(w, sheet_name="Selected_Ensemble", index=False)
        feat_df.to_excel(w, sheet_name="Strict_Features", index=False)
        pd.DataFrame({"curve_col": ccols}).to_excel(w, sheet_name="Curve_Columns", index=False)
    curves_df.to_csv(out/"vwea_predicted_curves_200pts.csv", index=False)
    summary_df.to_csv(out/"vwea_summary.csv", index=False)
    per_df.to_csv(out/"vwea_per_curve.csv", index=False)

    bundle = {
        "feature_cols": fcols,
        "target_cols": tc,
        "curve_cols": [str(c) for c in ccols],
        "prior_cols": [str(c) for c in pcols],
        "tau": tau,
        "anchors": anchors,
        "candidates": {n: candidates[n] for n in ens_names},
        "ensemble_names": ens_names,
        "ensemble_weights": ens_w,
        "args": vars(args),
    }
    joblib.dump(bundle, out/"vwea_model_bundle.joblib")

    diag = {
        "validation_overall_accuracy_%": float(summary_df.loc[summary_df["split"]=="val","overall_accuracy_%"].iloc[0]),
        "test_overall_accuracy_%": float(summary_df.loc[summary_df["split"]=="test","overall_accuracy_%"].iloc[0]),
        "reached_90_by_validation_rule": bool(float(summary_df.loc[summary_df["split"]=="val","overall_accuracy_%"].iloc[0]) >= 90),
        "strict_boundary": "No F/kN, true Td, true impulse, curve shape, MorphSignature, damage result, or source filename was used as predictor.",
        "note": "If validation accuracy remains far below 90%, missing shape-governing inputs are the bottleneck. Add a no-leakage FE/simplified-dynamics prior through --prior-prefix, or add measurable pre-test variables such as impact location, boundary/test-rig stiffness and contact head geometry."
    }
    with open(out/"vwea_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(summary_df.to_string(index=False))
    print(f"\n[OK] saved: {out/'vwea_results.xlsx'}")
    print(f"[OK] model: {out/'vwea_model_bundle.joblib'}")
    print(f"[DIAG] val_acc={diag['validation_overall_accuracy_%']:.2f}%, test_acc={diag['test_overall_accuracy_%']:.2f}%, reached90_by_val={diag['reached_90_by_validation_rule']}")


if __name__ == "__main__":
    main()
