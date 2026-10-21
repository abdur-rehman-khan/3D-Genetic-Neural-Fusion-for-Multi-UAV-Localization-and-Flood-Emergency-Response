"""Localization metrics. All errors are 3D Euclidean unless stated otherwise."""
from __future__ import annotations

import numpy as np


def position_errors(estimated: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-target 3D Euclidean error (same unit as inputs, metres in the simulator)."""
    e = np.atleast_2d(np.asarray(estimated, float)); t = np.atleast_2d(np.asarray(true, float))
    if e.shape != t.shape:
        raise ValueError(f"shape mismatch: {e.shape} vs {t.shape}")
    return np.sqrt(np.sum((e - t) ** 2, axis=1))


def rmse_3d(estimated, true) -> float:
    """RMSE = sqrt(mean_i(||est_i - true_i||^2)). 3D Euclidean position RMSE (NOT per-coordinate)."""
    err = position_errors(estimated, true)
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")


def mean_error(estimated, true) -> float:
    """Mean 3D Euclidean error."""
    err = position_errors(estimated, true)
    return float(np.mean(err)) if err.size else float("nan")


def mae_per_axis(estimated, true) -> float:
    """MAE = mean(|est - true|) averaged over targets AND the three coordinates."""
    e = np.atleast_2d(np.asarray(estimated, float)); t = np.atleast_2d(np.asarray(true, float))
    return float(np.mean(np.abs(e - t))) if e.size else float("nan")


def success_rate(errors, threshold: float) -> float:
    errors = np.asarray(errors, float)
    return float(np.mean(errors <= threshold)) if errors.size else float("nan")


def summarize(estimated, true, threshold: float) -> dict:
    est = np.asarray(estimated, float).reshape(-1, 3); tru = np.asarray(true, float).reshape(-1, 3)
    err = position_errors(est, tru) if len(est) else np.array([])
    nan = float("nan")
    return {
        "n": int(err.size),
        "mean_error_m": float(np.mean(err)) if err.size else nan,
        "std_error_m": float(np.std(err)) if err.size else nan,
        "rmse_3d_m": rmse_3d(est, tru) if err.size else nan,
        "mse_3d_m": float(np.mean(err ** 2)) if err.size else nan,
        "mae_axis_m": mae_per_axis(est, tru) if err.size else nan,
        "success_rate": success_rate(err, threshold),
    }
