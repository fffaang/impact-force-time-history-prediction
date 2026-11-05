#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create alternative effective-duration datasets for sensitivity analysis.

Input curves are normalized on the original duration. For an effective end ratio
r in (0, 1], the curve is cropped to [0, r], stretched back to 200 points, and
targets are updated:
    Td_eff = Td_original * r
    Fmax unchanged
    Impulse = Fmax * Td_eff * integral(F*_eff dt*)
    tp = tp*_eff * Td_eff
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import vwea_core as v3


def effective_ratios(Y: np.ndarray, tau: np.ndarray, mode: str) -> np.ndarray:
    out = []
    for y in np.asarray(Y, float):
        y = np.clip(y, 0, None)
        if y.max() > v3.EPS:
            y = y / y.max()
        pidx = int(np.argmax(y))
        if mode == "Td99":
            seg = 0.5 * (y[1:] + y[:-1]) * np.diff(tau)
            cum = np.r_[0.0, np.cumsum(seg)]
            total = cum[-1]
            r = float(np.interp(0.99 * total, cum, tau)) if total > v3.EPS else 1.0
        elif mode == "Td995":
            seg = 0.5 * (y[1:] + y[:-1]) * np.diff(tau)
            cum = np.r_[0.0, np.cumsum(seg)]
            total = cum[-1]
            r = float(np.interp(0.995 * total, cum, tau)) if total > v3.EPS else 1.0
        elif mode == "Td002":
            below = np.where(y[pidx:] <= 0.02)[0]
            r = float(tau[pidx + below[0]]) if len(below) else 1.0
        else:
            raise ValueError(mode)
        out.append(float(np.clip(r, max(float(tau[2]), 0.01), 1.0)))
    return np.asarray(out, float)


def rebuild(Y: np.ndarray, tau: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    tau_new = np.linspace(0, 1, Y.shape[1])
    out = np.zeros_like(Y, dtype=float)
    for i, y in enumerate(Y):
        x_old = tau_new * ratio[i]
        yy = np.interp(x_old, tau, y)
        yy = np.clip(yy, 0, None)
        yy[0] = 0.0
        yy[-1] = 0.0
        if yy.max() > v3.EPS:
            yy = yy / yy.max()
        out[i] = yy
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sheet", default="Dataset")
    ap.add_argument("--curve-prefix", default="f_norm_")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.data, sheet_name=args.sheet)
    ccols = [c for c in df.columns if str(c).startswith(args.curve_prefix)]
    Y = v3.shape_matrix(df, ccols)
    tau = np.linspace(0, 1, len(ccols))
    F = pd.to_numeric(df["Fmax_true_kN"], errors="coerce").to_numpy(float)
    Td0 = pd.to_numeric(df["Td_true_ms"], errors="coerce").to_numpy(float)

    summary = []
    for mode in ["Td99", "Td995", "Td002"]:
        ratio = effective_ratios(Y, tau, mode)
        Yeff = rebuild(Y, tau, ratio)
        Tdeff = Td0 * ratio
        area = np.trapz(Yeff, tau, axis=1)
        impulse = F * Tdeff * area
        tp = tau[np.argmax(Yeff, axis=1)] * Tdeff
        dfo = df.copy()
        dfo["Td_original_ms_before_sensitivity"] = Td0
        dfo["Td_effective_ratio"] = ratio
        dfo["Td_definition"] = mode
        dfo["Td_true_ms"] = Tdeff
        dfo["Impulse_true_kN_ms"] = impulse
        dfo["tp_true_ms"] = tp
        for j, c in enumerate(ccols):
            dfo[c] = Yeff[:, j]
        path = outdir / f"rectangular_curve_dataset_200pts_{mode}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            dfo.to_excel(w, "Dataset", index=False)
        summary.append({
            "definition": mode,
            "path": str(path),
            "n": len(dfo),
            "median_ratio": float(np.median(ratio)),
            "min_ratio": float(np.min(ratio)),
            "max_ratio": float(np.max(ratio)),
            "median_Td_ms": float(np.median(Tdeff)),
        })
    pd.DataFrame(summary).to_excel(outdir / "td_definition_dataset_summary.xlsx", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
