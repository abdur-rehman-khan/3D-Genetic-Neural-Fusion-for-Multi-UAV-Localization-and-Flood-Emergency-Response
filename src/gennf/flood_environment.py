"""Procedural flood-affected settlement (no external assets). Units: metres, z up."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Building:
    x: float; y: float; w: float; d: float; h: float


@dataclass
class FloodZone:
    cx: float; cy: float; rx: float; ry: float; severity: int; depth: float


@dataclass
class HazardZone:
    cx: float; cy: float; radius: float; height: float; label: str


@dataclass
class RoadSegment:
    x0: float; y0: float; x1: float; y1: float; width: float; submerged: bool


@dataclass
class FloodEnvironment:
    size: float
    buildings: list[Building]
    flood_zones: list[FloodZone]
    hazards: list[HazardZone]
    roads: list[RoadSegment]
    trees: np.ndarray                      # (n,3): x, y, height
    river: tuple[float, float]             # x-range of the flooded channel
    bridge: tuple[float, float, float, float, float]   # x0,x1,y,width,deck_z
    base: np.ndarray                       # (3,) rescue coordination station
    base_platform_h: float = 3.0

    # ------------------------------------------------------------------ queries
    def in_hazard(self, x: float, y: float, margin: float = 0.0) -> bool:
        return any(np.hypot(x - h.cx, y - h.cy) <= h.radius + margin for h in self.hazards)

    def in_building(self, x: float, y: float, margin: float = 0.0) -> bool:
        return any(b.x - margin <= x <= b.x + b.w + margin and b.y - margin <= y <= b.y + b.d + margin
                   for b in self.buildings)

    def flood_severity_at(self, x: float, y: float) -> int:
        sev = 0
        for z in self.flood_zones:
            if ((x - z.cx) / z.rx) ** 2 + ((y - z.cy) / z.ry) ** 2 <= 1.0:
                sev = max(sev, z.severity)
        if self.river[0] <= x <= self.river[1]:
            sev = max(sev, 3)
        return sev

    def plan_route(self, a: np.ndarray, b: np.ndarray, clearance: float = 6.0) -> list[np.ndarray]:
        """Waypoints from a to b (excluded a, included b) detouring around hazard cylinders in XY."""
        a = np.asarray(a, float); b = np.asarray(b, float); pts: list[np.ndarray] = []
        cur = a.copy()
        for h in sorted(self.hazards, key=lambda h: np.hypot(h.cx - a[0], h.cy - a[1])):
            ab = b[:2] - cur[:2]; L = float(np.linalg.norm(ab))
            if L < 1e-6:
                break
            u = ab / L; c = np.array([h.cx, h.cy]) - cur[:2]
            t = float(np.clip(c @ u, 0, L)); closest = cur[:2] + u * t
            if np.linalg.norm(closest - [h.cx, h.cy]) < h.radius + clearance:
                n = np.array([-u[1], u[0]])
                side = 1.0 if (closest - [h.cx, h.cy]) @ n >= 0 else -1.0
                wp = np.array([h.cx, h.cy]) + side * n * (h.radius + clearance + 4.0)
                pts.append(np.array([wp[0], wp[1], max(cur[2], b[2])])); cur = pts[-1]
        pts.append(b.copy())
        return pts

    # ------------------------------------------------------------------ generation
    @staticmethod
    def generate(seed: int = 42, size: float = 200.0) -> "FloodEnvironment":
        rng = np.random.default_rng(seed); s = size
        base = np.array([s / 2, s / 2, 0.0])
        river = (0.72 * s, 0.82 * s)
        bridge = (river[0] - 4, river[1] + 4, s / 2, 10.0, 3.0)
        zones = [FloodZone(0.25 * s, 0.30 * s, 0.22 * s, 0.18 * s, 2, 1.5),
                 FloodZone(0.30 * s, 0.75 * s, 0.20 * s, 0.16 * s, 1, 0.6),
                 FloodZone(0.55 * s, 0.20 * s, 0.15 * s, 0.12 * s, 3, 3.0),
                 FloodZone(0.60 * s, 0.82 * s, 0.14 * s, 0.14 * s, 2, 1.4),
                 FloodZone(0.90 * s, 0.30 * s, 0.10 * s, 0.20 * s, 3, 3.0)]
        hazards = [HazardZone(0.42 * s, 0.62 * s, 0.07 * s, 55.0, "Downed power lines"),
                   HazardZone(0.62 * s, 0.38 * s, 0.06 * s, 55.0, "Unstable structure")]
        env = FloodEnvironment(s, [], zones, hazards, [], np.zeros((0, 3)), river, bridge, base)
        # roads: E-W and N-S through the base, split in 20 m segments
        roads = []
        for k in np.arange(0, s, 20.0):
            roads.append(RoadSegment(k, s / 2, k + 20, s / 2, 10.0, False))
            roads.append(RoadSegment(s / 2, k, s / 2, k + 20, 10.0, False))
        for r in roads:
            mx, my = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            r.submerged = env.flood_severity_at(mx, my) >= 2 and not (bridge[0] <= mx <= bridge[1] and abs(my - bridge[2]) < 1)
        env.roads = roads
        # buildings (rejection sampling; keep clear of roads, base, river, hazards, each other)
        blds: list[Building] = []
        tries = 0
        while len(blds) < 30 and tries < 4000:
            tries += 1
            w, d = rng.uniform(8, 16, 2); h = float(rng.choice([8, 10, 12, 16, 20, 26, 32]))
            x, y = rng.uniform(5, s - 20, 2)
            if abs(x + w / 2 - s / 2) < 16 + w / 2 or abs(y + d / 2 - s / 2) < 16 + d / 2:
                continue
            if x + w > river[0] - 4 and x < river[1] + 4:
                continue
            if env.in_hazard(x + w / 2, y + d / 2, 12) or np.hypot(x - base[0], y - base[1]) < 35:
                continue
            if any(x < b.x + b.w + 5 and b.x < x + w + 5 and y < b.y + b.d + 5 and b.y < y + d + 5 for b in blds):
                continue
            blds.append(Building(float(x), float(y), float(w), float(d), h))
        env.buildings = blds
        trees = []
        while len(trees) < 45:
            x, y = rng.uniform(3, s - 3, 2)
            if env.in_building(x, y, 3) or env.in_hazard(x, y, 3) or abs(x - s / 2) < 8 or abs(y - s / 2) < 8 \
                    or river[0] - 2 <= x <= river[1] + 2:
                continue
            trees.append((x, y, float(rng.uniform(5, 10))))
        env.trees = np.array(trees)
        return env

    def sample_victims(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Hidden ground-truth victim positions: rooftops, flooded ground, or on debris/bridge."""
        out: list[np.ndarray] = []; tries = 0
        while len(out) < n and tries < 20000:
            tries += 1
            kind = rng.choice(["roof", "ground", "ground"])
            if kind == "roof" and self.buildings:
                b = self.buildings[int(rng.integers(len(self.buildings)))]
                if self.flood_severity_at(b.x + b.w / 2, b.y + b.d / 2) == 0:
                    continue
                p = np.array([b.x + rng.uniform(1, b.w - 1), b.y + rng.uniform(1, b.d - 1), b.h + 0.5])
            else:
                x, y = rng.uniform(8, self.size - 8, 2)
                if self.flood_severity_at(x, y) == 0 or self.in_building(x, y, 2) or self.in_hazard(x, y, 12) \
                        or self.river[0] <= x <= self.river[1] or np.hypot(x - self.base[0], y - self.base[1]) < 25:
                    continue
                p = np.array([x, y, 1.0])
            if all(np.linalg.norm(p - q) > 30 for q in out):
                out.append(p)
        return np.array(out)
