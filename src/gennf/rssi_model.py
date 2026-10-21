"""Log-distance RSSI model and distance utilities.

RSSI(d) = P0 - 10 * n * log10(d / d0) + X_sigma        [dBm]

* ``p0_dbm``   : received power at reference distance ``d0`` (dBm).
* ``n``        : path-loss exponent.
* ``sigma_db`` : std-dev of zero-mean Gaussian (log-normal shadowing) noise, in dB.

The original MATLAB script used ``+ sigma`` (a constant 7 dB *offset*, not noise).
Here noise is zero-mean Gaussian. The ``ORIGINAL_MATLAB`` preset keeps the original
numeric parameters (P0=55, n=1, d0=1, sigma=7) for comparison only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

MIN_DISTANCE_M = 1e-3  # guards log10(0)


def euclidean_distances(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    """3D Euclidean distance (uses x, y AND z) from each row of ``points`` to ``target``."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    target = np.asarray(target, dtype=float).reshape(3)
    return np.sqrt(np.sum((points - target) ** 2, axis=1))


@dataclass(frozen=True)
class RSSIModel:
    p0_dbm: float = -40.0
    n: float = 2.7
    d0: float = 1.0
    sigma_db: float = 2.0
    comm_range: float = 150.0  # metres; beyond this no measurement is available

    def with_(self, **kw) -> "RSSIModel":
        return replace(self, **kw)

    def mean_rssi(self, distance: np.ndarray) -> np.ndarray:
        d = np.maximum(np.asarray(distance, dtype=float), MIN_DISTANCE_M)
        return self.p0_dbm - 10.0 * self.n * np.log10(d / self.d0)

    def measure(self, distance: np.ndarray, rng: np.random.Generator | None = None,
                apply_range: bool = True) -> np.ndarray:
        """Noisy RSSI; entries beyond ``comm_range`` become NaN (no link)."""
        d = np.asarray(distance, dtype=float)
        rssi = self.mean_rssi(d)
        if self.sigma_db > 0:
            rng = rng or np.random.default_rng()
            rssi = rssi + rng.normal(0.0, self.sigma_db, size=rssi.shape)
        if apply_range:
            rssi = np.where(d <= self.comm_range, rssi, np.nan)
        return rssi

    def distance_from_rssi(self, rssi: np.ndarray) -> np.ndarray:
        """Invert the path-loss model. NaN stays NaN."""
        rssi = np.asarray(rssi, dtype=float)
        return np.maximum(self.d0 * 10.0 ** ((self.p0_dbm - rssi) / (10.0 * self.n)), MIN_DISTANCE_M)


ORIGINAL_MATLAB = RSSIModel(p0_dbm=55.0, n=1.0, d0=1.0, sigma_db=7.0, comm_range=1e9)
