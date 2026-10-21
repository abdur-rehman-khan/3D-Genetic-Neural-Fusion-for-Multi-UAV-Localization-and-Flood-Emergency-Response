"""Baseline first-aid allocation policy (NEW EXTENSION - not part of the original GenNF paper).

Greedy policy: process victims by (urgency desc, time waiting desc); give each the nearest AVAILABLE
UAV (idle, battery >= 30 %) whose battery can cover the round trip. One UAV <-> one victim at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

SUPPLIES = ["first-aid kit", "drinking water", "life vest", "medication"]


class VictimStatus(str, Enum):
    UNDETECTED = "undetected"
    DETECTED = "detected"
    LOCALIZED = "localized"
    ASSIGNED = "assigned"
    ASSISTED = "assisted"
    FAILED = "failed"


@dataclass
class Victim:
    id: int
    true_position: np.ndarray              # hidden ground truth: measurement generation + evaluation ONLY
    urgency: int                           # 1 (low) .. 3 (critical)
    supply: str
    detect_time: float
    status: VictimStatus = VictimStatus.UNDETECTED
    estimated_position: np.ndarray | None = None
    error_m: float | None = None
    anchor_ids: list = field(default_factory=list)
    last_rssi: np.ndarray | None = None
    localized_at: float | None = None
    next_localize_time: float = 0.0
    attempts: int = 0
    assigned_at: float | None = None
    assisted_at: float | None = None
    uav_id: int | None = None
    localization_ok: bool = False          # error <= success threshold

    @property
    def name(self) -> str:
        return f"V-{self.id + 1:02d}"

    @property
    def mission_time(self) -> float | None:
        return None if self.assisted_at is None else self.assisted_at - self.detect_time


def battery_needed(distance_m: float) -> float:
    """Matches the drain model in uav_network (0.005 %/m moved + hover), x2 for round trip, +10 % reserve."""
    return 2.0 * 0.005 * distance_m + 0.03 * (2 * distance_m / 8.0) + 10.0


def allocate(victims: list[Victim], uavs: list, now: float) -> list[tuple[Victim, object]]:
    """Return (victim, uav) assignments; each UAV and victim appears at most once."""
    queue = sorted([v for v in victims if v.status == VictimStatus.LOCALIZED],
                   key=lambda v: (-v.urgency, v.detect_time))
    free = [u for u in uavs if u.available]; out = []
    for v in queue:
        best, best_d = None, np.inf
        for u in free:
            d = float(np.linalg.norm(u.position - v.estimated_position))
            if u.battery >= battery_needed(d) and d < best_d:
                best, best_d = u, d
        if best is not None:
            out.append((v, best)); free.remove(best)
    return out
