"""Repeatable localization experiments (research mode). Compares clearly separated methods:
  gennf            - GenNF (this repo's corrected, leak-free implementation)
  geometric_only   - GenNF's geometric stage without the GA/ANN refinement (ablation)
  ls_same_anchors  - baseline: least-squares multilateration on the same top-4 anchors
  ls_all_anchors   - baseline: least-squares on every in-range anchor
Nothing here reproduces published numbers; it generates NEW simulation results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from . import metrics as M
from .genetic_optimizer import GAConfig
from .localization import GenNFConfig, GenNFLocalizer, least_squares_localize
from .rssi_model import RSSIModel, euclidean_distances
from .uav_network import make_formation

TARGET_Z = (0.0, 35.0)


@dataclass
class ExperimentConfig:
    seed: int = 0
    n_trials: int = 100
    n_uavs: int = 10
    n_available: int = 10               # UAVs that can take part (random subset per trial)
    deployment: str = "random"          # random | grid | ring | two_layer
    area: float = 200.0
    noise_db: float = 1.0
    comm_range: float = 150.0
    p0_dbm: float = -40.0
    path_loss_exp: float = 2.7
    target: tuple[float, float, float] | None = None   # fixed target; None = random per trial
    geometry: str = "trilateration"
    use_ga: bool = True
    ga_population: int = 100
    ga_generations: int = 50
    ga_mutation_rate: float = 0.05
    n_train: int = 3000
    epochs: int = 150
    success_threshold_m: float = 20.0
    train_noise_db: float | None = None  # None = same as test noise


def _sampler(cfg: ExperimentConfig, rng_fixed: np.random.Generator):
    if cfg.deployment == "random":
        return lambda rng: rng.uniform((0, 0, 22), (cfg.area, cfg.area, 50), size=(cfg.n_uavs, 3)), None
    fixed = make_formation(cfg.deployment, cfg.n_uavs, cfg.area, rng_fixed)
    return (lambda rng: fixed), fixed


def build_localizer(cfg: ExperimentConfig) -> GenNFLocalizer:
    """Train GenNF on simulated labelled targets (offline phase)."""
    model = RSSIModel(cfg.p0_dbm, cfg.path_loss_exp, 1.0,
                      cfg.noise_db if cfg.train_noise_db is None else cfg.train_noise_db, cfg.comm_range)
    sampler, _ = _sampler(cfg, np.random.default_rng(cfg.seed + 1))
    gc = GenNFConfig(feature_mode="extended" if cfg.deployment == "random" else "geom3",
                     geometry=cfg.geometry, use_ga=cfg.use_ga, n_train=cfg.n_train, epochs=cfg.epochs,
                     bounds_hi=(cfg.area, cfg.area, TARGET_Z[1]), seed=cfg.seed,
                     ga=GAConfig(cfg.ga_population, cfg.ga_generations, cfg.ga_mutation_rate))
    loc = GenNFLocalizer(model, gc); loc.fit(sampler)
    return loc


def run_trials(cfg: ExperimentConfig, localizer: GenNFLocalizer | None = None) -> pd.DataFrame:
    """One row per (trial, method). Returns a tidy DataFrame; config is attached in ``df.attrs``.
    The test-time RSSI model uses cfg.noise_db even if the localizer was trained at another noise level."""
    rng = np.random.default_rng(cfg.seed)
    test_model = RSSIModel(cfg.p0_dbm, cfg.path_loss_exp, 1.0, cfg.noise_db, cfg.comm_range)
    sampler, _ = _sampler(cfg, np.random.default_rng(cfg.seed + 1))
    localizer = localizer or build_localizer(cfg)
    localizer.rssi_model = test_model
    rows = []
    for trial in range(cfg.n_trials):
        uav = sampler(rng)
        tgt = np.array(cfg.target, float) if cfg.target else rng.uniform((0, 0, TARGET_Z[0]), (cfg.area, cfg.area, TARGET_Z[1]))
        rssi = test_model.measure(euclidean_distances(uav, tgt), rng)
        if cfg.n_available < cfg.n_uavs:
            off = rng.choice(cfg.n_uavs, cfg.n_uavs - cfg.n_available, replace=False); rssi = rssi.copy(); rssi[off] = np.nan
        outs = {}
        t0 = time.perf_counter(); r = localizer.localize(uav, rssi); t1 = time.perf_counter()
        outs["gennf"] = (r.estimated_position, r.status, t1 - t0)
        outs["geometric_only"] = (r.geometric_estimate if r.geometric_estimate is not None else np.full(3, np.nan), r.status, 0.0)
        for name, k in (("ls_same_anchors", 4), ("ls_all_anchors", None)):
            t0 = time.perf_counter(); b = least_squares_localize(uav, rssi, test_model, (0, 0, 0), (cfg.area, cfg.area, 60), k)
            outs[name] = (b.estimated_position, b.status, time.perf_counter() - t0)
        for name, (est, status, rt) in outs.items():
            ok = np.all(np.isfinite(est))
            err = float(np.linalg.norm(est - tgt)) if ok else np.nan
            rows.append({"trial": trial, "method": name, "status": status, "error_m": err, "runtime_ms": rt * 1e3,
                         "true_x": tgt[0], "true_y": tgt[1], "true_z": tgt[2],
                         "est_x": est[0], "est_y": est[1], "est_z": est[2],
                         "n_links": int(np.isfinite(rssi).sum())})
    df = pd.DataFrame(rows); df.attrs["config"] = asdict(cfg)
    return df


def summarize_trials(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = []
    for method, g in df.groupby("method", sort=False):
        ok = g.dropna(subset=["error_m"])
        est = ok[["est_x", "est_y", "est_z"]].to_numpy(); tru = ok[["true_x", "true_y", "true_z"]].to_numpy()
        s = M.summarize(est, tru, threshold)
        out.append({"method": method, "n_valid": s["n"], "n_failed": int(len(g) - len(ok)),
                    "mean_error_m": s["mean_error_m"], "std_error_m": s["std_error_m"], "rmse_3d_m": s["rmse_3d_m"],
                    "mae_axis_m": s["mae_axis_m"], "success_rate": s["success_rate"],
                    "mean_runtime_ms": float(g["runtime_ms"].mean())})
    return pd.DataFrame(out)


def noise_sweep(cfg: ExperimentConfig, noise_levels, retrain: bool = False) -> pd.DataFrame:
    """Error vs RSSI noise. Default: GenNF is trained once at cfg.noise_db and tested at every level
    (shows robustness to noise mismatch). retrain=True trains a matched model per level."""
    frames = []; loc = None if retrain else build_localizer(cfg)
    for s in noise_levels:
        c = ExperimentConfig(**{**asdict(cfg), "noise_db": float(s)})
        df = run_trials(c, loc); df["noise_db"] = float(s); frames.append(df)
    return pd.concat(frames, ignore_index=True)
