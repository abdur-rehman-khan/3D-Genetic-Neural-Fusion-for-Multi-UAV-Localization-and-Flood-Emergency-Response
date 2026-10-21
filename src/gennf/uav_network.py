"""UAV nodes, deployment formations and kinematics."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .flood_environment import FloodEnvironment


class UAVStatus(str, Enum):
    IDLE = "idle (patrol)"
    TRANSIT = "en route to victim"
    DELIVERING = "delivering aid"
    RETURNING = "returning to base"
    CHARGING = "charging"


def make_formation(kind: str, n: int, size: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """(n,3) UAV positions: x,y in [0,size], z in metres."""
    rng = rng or np.random.default_rng(0)
    c = size / 2
    if kind == "random":
        return rng.uniform((0.1 * size, 0.1 * size, 22), (0.9 * size, 0.9 * size, 50), size=(n, 3))
    if kind == "ring":
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[c + 0.32 * size * np.cos(ang), c + 0.32 * size * np.sin(ang), np.where(np.arange(n) % 2, 40.0, 25.0)]
    if kind == "grid":
        cols = int(np.ceil(np.sqrt(n * 1.5))); rows = int(np.ceil(n / cols)); pts = []
        for i in range(n):
            r, k = divmod(i, cols)
            pts.append((size * (k + 0.5) / cols, size * (r + 0.5) / rows, 30.0))
        return np.array(pts)
    if kind == "two_layer":
        half = n // 2; lo = make_formation("grid", half, size); hi = make_formation("grid", n - half, size)
        lo[:, 2] = 22.0; hi[:, 2] = 46.0; hi[:, :2] += size * 0.05
        return np.clip(np.vstack([lo, hi]), [0, 0, 0], [size, size, 60])
    raise ValueError(f"unknown formation {kind!r}")


@dataclass
class UAV:
    id: int
    position: np.ndarray
    home: np.ndarray
    battery: float                         # simulated, percent
    status: UAVStatus = UAVStatus.IDLE
    victim_id: int | None = None
    waypoints: list = field(default_factory=list)
    timer: float = 0.0
    distance_flown: float = 0.0
    comm_ok: bool = True

    @property
    def name(self) -> str:
        return f"UAV-{self.id + 1:02d}"

    @property
    def altitude(self) -> float:
        return float(self.position[2])

    @property
    def available(self) -> bool:
        return self.status == UAVStatus.IDLE and self.battery >= 30.0


class UAVNetwork:
    def __init__(self, env: FloodEnvironment, n: int, formation: str, rng: np.random.Generator,
                 speed: float = 8.0, comm_range: float = 150.0):
        self.env, self.rng, self.speed, self.comm_range = env, rng, speed, comm_range
        homes = make_formation(formation, n, env.size, rng)
        self.uavs = [UAV(i, homes[i].copy(), homes[i].copy(), float(rng.uniform(70, 100))) for i in range(n)]

    @property
    def positions(self) -> np.ndarray:
        return np.array([u.position for u in self.uavs])

    def move_toward(self, u: UAV, target: np.ndarray, dt: float) -> bool:
        """Advance toward target at constant speed; returns True on arrival."""
        vec = target - u.position; dist = float(np.linalg.norm(vec)); step = self.speed * dt
        if dist <= step:
            u.position = target.astype(float).copy(); u.distance_flown += dist; self._drain(u, dist, dt); return True
        u.position = u.position + vec / dist * step; u.distance_flown += step; self._drain(u, step, dt)
        return False

    @staticmethod
    def _drain(u: UAV, moved: float, dt: float) -> None:
        u.battery = max(0.0, u.battery - (0.03 * dt + 0.005 * moved))

    def patrol_waypoint(self, u: UAV, radius: float = 30.0) -> np.ndarray:
        for _ in range(30):
            ang, r = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0.3, 1.0) * radius
            p = u.home + np.array([r * np.cos(ang), r * np.sin(ang), self.rng.uniform(-4, 4)])
            p[:2] = np.clip(p[:2], 0, self.env.size); p[2] = np.clip(p[2], 20, 55)
            if not self.env.in_hazard(p[0], p[1], 6):
                return p
        return u.home.copy()

    def update_comm(self) -> None:
        base = self.env.base + np.array([0, 0, 5.0])
        for u in self.uavs:
            u.comm_ok = bool(np.linalg.norm(u.position - base) <= self.comm_range)
