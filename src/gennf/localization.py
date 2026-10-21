"""GenNF localization (leak-free corrected implementation) and a least-squares baseline.

Pipeline at inference (NO ground truth is used):
  RSSI -> distance estimates -> select the strongest anchors (UAV IDs) -> geometric estimate
  (weighted pairwise/triple combination, as in the MATLAB script) -> ANN refinement whose weights
  were initialised by a genetic algorithm and fine-tuned offline on *simulated* labelled targets.

Ground-truth positions appear only (a) as supervised labels of synthetic OFFLINE training targets and
(b) in evaluation helpers. See docs/methodology.md for every deviation from the MATLAB source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .genetic_optimizer import GAConfig, genetic_optimize
from .metrics import position_errors
from .neural_model import MLP
from .rssi_model import RSSIModel, euclidean_distances

EPS = 1e-9


# --------------------------------------------------------------------------- geometry
def select_anchors(rssi: np.ndarray, k: int = 4) -> np.ndarray:
    """IDs (indices into the UAV array) of the k strongest valid RSSI links, strongest first.

    Ties are broken by lowest UAV id (stable sort). Returned IDs are unique by construction.
    """
    rssi = np.asarray(rssi, float)
    valid = np.flatnonzero(np.isfinite(rssi))
    order = valid[np.argsort(-rssi[valid], kind="stable")]
    return order[:k]


def pair_point(a: np.ndarray, b: np.ndarray, ra: float, rb: float) -> np.ndarray:
    """Point on the line through anchors a, b implied by the two range estimates.

    * spheres do not touch (d >= ra+rb): point on the segment at distance ra/(ra+rb)*d from a
      (the MATLAB 'Dab > C1+C2' branch).
    * otherwise: foot of the radical plane, t = (d^2 + ra^2 - rb^2) / (2d) from a.
    Degenerate cases (coincident anchors, zero radii) return finite values.
    """
    d = float(np.linalg.norm(b - a))
    if d < EPS:
        return a.copy()
    u = (b - a) / d
    rs = ra + rb
    if d >= rs:
        t = d * 0.5 if rs < EPS else d * ra / rs
    else:
        t = (d * d + ra * ra - rb * rb) / (2.0 * d)
    return a + t * u


def _weighted(points: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.array(weights, float); w = w / w.sum()
    return np.sum(np.array(points) * w[:, None], axis=0)


def trilaterate_triple(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Intersection of 3 spheres (closed form). Of the two mirror solutions the LOWER one is returned
    (targets lie below the airborne anchors). If z^2 < 0 (noisy, non-intersecting) it is clipped to 0, i.e.
    the point is projected onto the anchors' plane. Collinear anchors fall back to the pairwise estimate."""
    p1, p2, p3 = p; d = np.linalg.norm(p2 - p1)
    if d < EPS:
        return triple_estimate(p, r, mode="pairwise")
    ex = (p2 - p1) / d; v = p3 - p1; i = ex @ v; ey_raw = v - i * ex; j = np.linalg.norm(ey_raw)
    if j < 1e-6 * max(d, 1.0):
        return triple_estimate(p, r, mode="pairwise")
    ey = ey_raw / j; ez = np.cross(ex, ey)
    x = (r[0] ** 2 - r[1] ** 2 + d * d) / (2 * d)
    y = (r[0] ** 2 - r[2] ** 2 + i * i + j * j) / (2 * j) - (i / j) * x
    z = np.sqrt(max(r[0] ** 2 - x * x - y * y, 0.0))
    base = p1 + x * ex + y * ey
    a, b = base + z * ez, base - z * ez
    return a if a[2] < b[2] else b


def triple_estimate(p: np.ndarray, r: np.ndarray, mode: str = "pairwise") -> np.ndarray:
    """Per-triple estimate. 'pairwise': inverse-range-sum weighted mean of the 3 pairwise points (MATLAB
    structure). 'trilateration': corrected variant, see trilaterate_triple."""
    if mode == "trilateration":
        return trilaterate_triple(p, r)
    pts, wts = [], []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        pts.append(pair_point(p[i], p[j], r[i], r[j])); wts.append(1.0 / max(r[i] + r[j], EPS))
    return _weighted(pts, wts)


def geometric_estimate(anchor_pos: np.ndarray, ranges: np.ndarray, mode: str = "pairwise") -> np.ndarray:
    """Geometric stage. 4 anchors -> 4 cyclic triples (ABC, BCD, CDA, DAB) combined with weights
    1/(sum of the triple's ranges); 3 anchors -> single triple."""
    p = np.asarray(anchor_pos, float); r = np.asarray(ranges, float)
    if len(p) < 3:
        raise ValueError("need >= 3 anchors")
    if len(p) == 3:
        return triple_estimate(p, r, mode)
    groups = ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1))
    ests = [triple_estimate(p[list(g)], r[list(g)], mode) for g in groups]
    wts = [1.0 / max(r[list(g)].sum(), EPS) for g in groups]
    return _weighted(ests, wts)


# --------------------------------------------------------------------------- GenNF
@dataclass
class GenNFConfig:
    n_anchors: int = 4
    geometry: str = "pairwise"          # "pairwise" (MATLAB structure) | "trilateration" (corrected variant)
    feature_mode: str = "extended"      # "geom3" (original 3-input topology) | "extended"
    hidden: tuple[int, ...] = (64, 32)
    ga: GAConfig = field(default_factory=GAConfig)
    use_ga: bool = True
    n_train: int = 4000
    ga_subset: int = 1000
    epochs: int = 500
    lr: float = 3e-3
    bounds_lo: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_hi: tuple[float, float, float] = (200.0, 200.0, 60.0)
    seed: int = 0


@dataclass
class LocalizationResult:
    """Output of localization. Contains NO ground truth."""
    estimated_position: np.ndarray
    geometric_estimate: np.ndarray | None
    anchor_ids: list[int]
    rssi: np.ndarray
    estimated_distances: np.ndarray
    status: str                           # ok | geometric_fallback | insufficient_anchors
    diagnostics: dict = field(default_factory=dict)


@dataclass
class LocalizationRecord:
    """Result + ground truth + error, assembled by the evaluation layer (never fed to the model)."""
    result: LocalizationResult
    true_position: np.ndarray
    error_m: float


def evaluate(result: LocalizationResult, true_position: np.ndarray) -> LocalizationRecord:
    err = float(position_errors(result.estimated_position, np.asarray(true_position))[0])
    return LocalizationRecord(result, np.asarray(true_position, float), err)


class GenNFLocalizer:
    def __init__(self, rssi_model: RSSIModel, config: GenNFConfig | None = None):
        self.rssi_model = rssi_model
        self.cfg = config or GenNFConfig()
        lo = np.array(self.cfg.bounds_lo, float); hi = np.array(self.cfg.bounds_hi, float)
        self.center = (lo + hi) / 2.0
        self.scale = float(np.max(hi - lo))
        n_in = 3 if self.cfg.feature_mode == "geom3" else 3 + 3 * self.cfg.n_anchors + self.cfg.n_anchors
        if self.cfg.feature_mode not in ("geom3", "extended"):
            raise ValueError("feature_mode must be 'geom3' or 'extended'")
        self.net = MLP(n_in, self.cfg.hidden, 3, np.random.default_rng(self.cfg.seed))
        self.fitted = False
        self.train_info: dict = {}

    # -- features
    def _prepare(self, uav_pos: np.ndarray, rssi: np.ndarray):
        ids = select_anchors(rssi, self.cfg.n_anchors)
        if len(ids) < 3:
            return ids, None, None, None
        ap = uav_pos[ids]; dist = self.rssi_model.distance_from_rssi(rssi[ids])
        geo = np.clip(geometric_estimate(ap, dist, self.cfg.geometry),   # keep inside the deployment volume
                      np.array(self.cfg.bounds_lo, float), np.array(self.cfg.bounds_hi, float))
        if self.cfg.feature_mode == "geom3":
            feat = (geo - self.center) / self.scale
        else:
            feat = np.concatenate([(geo - self.center) / self.scale,
                                   ((ap - self.center) / self.scale).ravel(), dist / self.scale]) \
                if len(ids) == self.cfg.n_anchors else None
        return ids, geo, dist, feat

    # -- offline training on simulated labelled targets
    def fit(self, anchor_sampler: Callable[[np.random.Generator], np.ndarray]) -> dict:
        cfg = self.cfg; rng = np.random.default_rng(cfg.seed)
        lo = np.array(cfg.bounds_lo, float); hi = np.array(cfg.bounds_hi, float)
        X, Y = [], []
        tries = 0
        while len(X) < cfg.n_train and tries < cfg.n_train * 5:
            tries += 1
            uav = anchor_sampler(rng); tgt = rng.uniform(lo, hi)   # synthetic training target
            rssi = self.rssi_model.measure(euclidean_distances(uav, tgt), rng)
            ids, geo, dist, feat = self._prepare(uav, rssi)
            if feat is None:
                continue
            X.append(feat); Y.append((tgt - geo) / self.scale)      # label = residual to the truth (offline only)
        if not X:
            raise RuntimeError("no valid training samples (communication range too small?)")
        X = np.array(X); Y = np.array(Y)
        ga_hist: list[float] = []
        if cfg.use_ga:
            sub = rng.choice(len(X), size=min(cfg.ga_subset, len(X)), replace=False)
            tmp = MLP(self.net.sizes[0], cfg.hidden, 3, rng)
            pop = []
            for _ in range(cfg.ga.population_size):
                tmp = MLP(self.net.sizes[0], cfg.hidden, 3, rng); pop.append(tmp.get_flat())
            def fit_fn(ind):
                tmp.set_flat(ind); return -tmp.mse(X[sub], Y[sub])
            res = genetic_optimize(fit_fn, np.array(pop), cfg.ga, rng)
            self.net.set_flat(res.best); ga_hist = res.history
        hist = self.net.fit(X, Y, epochs=cfg.epochs, lr=cfg.lr, rng=rng)
        self.fitted = True
        self.train_info = {"n_samples": int(len(X)), "ga_history": ga_hist, "train_loss": hist,
                           "final_mse_norm": hist[-1], "use_ga": cfg.use_ga}
        return self.train_info

    # -- inference (no ground truth anywhere)
    def localize(self, uav_positions: np.ndarray, rssi: np.ndarray) -> LocalizationResult:
        uav = np.asarray(uav_positions, float); rssi = np.asarray(rssi, float)
        if uav.ndim != 2 or uav.shape[1] != 3 or rssi.shape != (len(uav),):
            raise ValueError("uav_positions must be (N,3) and rssi (N,)")
        dist_all = self.rssi_model.distance_from_rssi(rssi)
        ids, geo, dist, feat = self._prepare(uav, rssi)
        if geo is None:
            return LocalizationResult(np.full(3, np.nan), None, [int(i) for i in ids], rssi, dist_all,
                                      "insufficient_anchors", {"n_valid": int(np.isfinite(rssi).sum())})
        if feat is None or not self.fitted:
            return LocalizationResult(geo, geo, [int(i) for i in ids], rssi, dist_all,
                                      "geometric_fallback", {"fitted": self.fitted})
        est = np.clip(geo + self.scale * self.net.predict(feat)[0],
                      np.array(self.cfg.bounds_lo, float), np.array(self.cfg.bounds_hi, float))
        return LocalizationResult(est, geo, [int(i) for i in ids], rssi, dist_all, "ok",
                                  {"geometric_correction_m": float(np.linalg.norm(est - geo))})


# --------------------------------------------------------------------------- baseline
def least_squares_localize(uav_positions: np.ndarray, rssi: np.ndarray, rssi_model: RSSIModel,
                           bounds_lo=(0, 0, 0), bounds_hi=(200, 200, 60), k: int | None = None) -> LocalizationResult:
    """Baseline (NOT GenNF): nonlinear least-squares multilateration on in-range anchors."""
    uav = np.asarray(uav_positions, float); rssi = np.asarray(rssi, float)
    ids = select_anchors(rssi, k if k else len(uav))
    dist_all = rssi_model.distance_from_rssi(rssi)
    if len(ids) < 4:
        return LocalizationResult(np.full(3, np.nan), None, [int(i) for i in ids], rssi, dist_all,
                                  "insufficient_anchors")
    ap, d = uav[ids], dist_all[ids]
    x0 = np.average(ap, axis=0, weights=1.0 / np.maximum(d, EPS))
    sol = least_squares(lambda x: np.linalg.norm(ap - x, axis=1) - d, x0,
                        bounds=(np.array(bounds_lo, float), np.array(bounds_hi, float)))
    return LocalizationResult(sol.x, x0, [int(i) for i in ids], rssi, dist_all, "ok", {"cost": float(sol.cost)})
