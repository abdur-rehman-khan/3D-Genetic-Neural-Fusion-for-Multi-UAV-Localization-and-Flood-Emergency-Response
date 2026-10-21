"""Frame-based simulation engine: UAV motion, victim localization (GenNF), first-aid missions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .flood_environment import FloodEnvironment
from .localization import GenNFConfig, GenNFLocalizer, evaluate
from .genetic_optimizer import GAConfig
from .rescue_allocation import SUPPLIES, Victim, VictimStatus, allocate
from .rssi_model import RSSIModel, euclidean_distances
from .uav_network import UAV, UAVNetwork, UAVStatus

_LOCALIZER_CACHE: dict = {}
TARGET_Z_MAX = 35.0   # victims are on the ground / rooftops; UAVs fly at 20-55 m


@dataclass
class SimConfig:
    seed: int = 42
    n_uavs: int = 10
    n_victims: int = 6
    dt: float = 0.5
    max_time: float = 300.0
    formation: str = "two_layer"
    area: float = 200.0
    rssi: RSSIModel = field(default_factory=lambda: RSSIModel(sigma_db=1.0))
    success_threshold_m: float = 20.0      # localization 'success' (metrics only)
    delivery_radius_m: float = 30.0        # ASSUMPTION: drop-zone tolerance of a supply drop
    delivery_time_s: float = 4.0
    speed_mps: float = 8.0
    max_attempts: int = 3
    n_train: int = 3000
    epochs: int = 150
    use_ga: bool = True
    geometry: str = "trilateration"
    hidden: tuple[int, ...] = (64, 32)
    ga_population: int = 200
    ga_generations: int = 100
    ga_mutation_rate: float = 0.1
    ga_mutation_sigma: float = 0.05


def get_localizer(cfg: SimConfig) -> GenNFLocalizer:
    """Train (or fetch cached) GenNF localizer on *simulated* random UAV layouts."""
    key = (cfg.rssi, cfg.area, cfg.n_uavs, cfg.n_train, cfg.epochs, cfg.use_ga, cfg.seed, cfg.geometry, cfg.hidden, cfg.ga_population, cfg.ga_generations, cfg.ga_mutation_rate, cfg.ga_mutation_sigma)
    if key not in _LOCALIZER_CACHE:
        gc = GenNFConfig(feature_mode="extended", use_ga=cfg.use_ga, n_train=cfg.n_train, epochs=cfg.epochs,
                         bounds_hi=(cfg.area, cfg.area, TARGET_Z_MAX), seed=cfg.seed,
                         hidden=cfg.hidden, ga=GAConfig(population_size=cfg.ga_population, generations=cfg.ga_generations,
                                                         mutation_rate=cfg.ga_mutation_rate, mutation_sigma=cfg.ga_mutation_sigma),
                         geometry=cfg.geometry)
        loc = GenNFLocalizer(cfg.rssi, gc)
        n, a = cfg.n_uavs, cfg.area
        loc.fit(lambda rng: rng.uniform((0, 0, 20), (a, a, 55), size=(n, 3)))
        _LOCALIZER_CACHE[key] = loc
    return _LOCALIZER_CACHE[key]


class Simulation:
    def __init__(self, cfg: SimConfig | None = None, localizer: GenNFLocalizer | None = None):
        self.cfg = cfg or SimConfig()
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.env = FloodEnvironment.generate(c.seed, c.area)
        self.net = UAVNetwork(self.env, c.n_uavs, c.formation, self.rng, c.speed_mps, c.rssi.comm_range)
        self.localizer = localizer or get_localizer(c)
        pos = self.env.sample_victims(c.n_victims, self.rng)
        times = np.sort(self.rng.uniform(2.0, max(10.0, 0.25 * c.max_time), len(pos)))
        self.victims = [Victim(i, p, int(self.rng.integers(1, 4)), SUPPLIES[int(self.rng.integers(len(SUPPLIES)))],
                               float(times[i])) for i, p in enumerate(pos)]
        self.t = 0.0
        self.events: list[dict] = []
        self.history: list[dict] = []
        self._record()

    # ------------------------------------------------------------------ helpers
    def _log(self, kind: str, who: str, detail: str = "") -> None:
        self.events.append({"time_s": round(self.t, 2), "event": kind, "who": who, "detail": detail})

    def _localize(self, v: Victim) -> None:
        c = self.cfg
        rssi = c.rssi.measure(euclidean_distances(self.net.positions, v.true_position), self.rng)
        res = self.localizer.localize(self.net.positions, rssi)      # ground truth is NOT passed in
        v.last_rssi = rssi
        if res.status == "insufficient_anchors":
            v.next_localize_time = self.t + 2.0
            self._log("localization_retry", v.name, "fewer than 3 UAVs in range")
            return
        est = np.clip(res.estimated_position, [0, 0, 0], [c.area, c.area, TARGET_Z_MAX])
        rec = evaluate(res, v.true_position)                          # evaluation only
        v.estimated_position, v.anchor_ids = est, res.anchor_ids
        v.error_m = float(np.linalg.norm(est - v.true_position))
        v.localization_ok = v.error_m <= c.success_threshold_m
        v.localized_at = self.t; v.attempts += 1; v.status = VictimStatus.LOCALIZED
        self._log("localized", v.name, f"anchors={[a + 1 for a in res.anchor_ids]} error={v.error_m:.1f} m")

    def _assign(self, v: Victim, u: UAV) -> None:
        env = self.env
        v.status, v.assigned_at, v.uav_id = VictimStatus.ASSIGNED, self.t, u.id
        u.status, u.victim_id = UAVStatus.TRANSIT, v.id
        cruise = max(u.position[2], 30.0)
        above = np.array([v.estimated_position[0], v.estimated_position[1], cruise])
        dest = np.array([v.estimated_position[0], v.estimated_position[1], v.estimated_position[2] + 4.0])
        u.waypoints = env.plan_route(u.position, above) + [dest]
        self._log("assigned", v.name, f"{u.name} (urgency {v.urgency}, {v.supply})")

    def _finish_delivery(self, v: Victim, u: UAV) -> None:
        # delivery succeeds only if the drop point (the estimate) is within the drop-zone radius of the true victim
        if np.linalg.norm(v.estimated_position - v.true_position) <= self.cfg.delivery_radius_m:
            v.status, v.assisted_at = VictimStatus.ASSISTED, self.t
            self._log("assisted", v.name, f"{u.name} delivered {v.supply}; mission time {v.mission_time:.1f} s")
        elif v.attempts >= self.cfg.max_attempts:
            v.status = VictimStatus.FAILED
            self._log("failed", v.name, "delivery missed; attempts exhausted")
        else:
            v.status, v.next_localize_time = VictimStatus.DETECTED, self.t
            self._log("delivery_missed", v.name, f"drop {np.linalg.norm(v.estimated_position - v.true_position):.1f} m off; re-localizing")
        u.status, u.victim_id = UAVStatus.RETURNING, None
        u.waypoints = self.env.plan_route(u.position, u.home)

    def _update_uav(self, u: UAV) -> None:
        dt = self.cfg.dt; net = self.net
        if u.status == UAVStatus.CHARGING:
            if np.linalg.norm(u.position - u.home) > 0.5:
                net.move_toward(u, u.home, dt)
            else:
                u.battery = min(100.0, u.battery + 2.0 * dt)
                if u.battery >= 90: u.status = UAVStatus.IDLE
            return
        if u.status == UAVStatus.IDLE:
            if u.battery < 25.0:
                u.status = UAVStatus.CHARGING; u.waypoints = []; self._log("low_battery", u.name, f"{u.battery:.0f} %"); return
            if not u.waypoints: u.waypoints = [net.patrol_waypoint(u)]
        if u.status == UAVStatus.DELIVERING:
            u.timer -= dt
            if u.timer <= 0: self._finish_delivery(self.victims[u.victim_id], u)
            return
        if u.waypoints:
            if net.move_toward(u, u.waypoints[0], dt):
                u.waypoints.pop(0)
                if not u.waypoints:
                    if u.status == UAVStatus.TRANSIT: u.status, u.timer = UAVStatus.DELIVERING, self.cfg.delivery_time_s
                    elif u.status == UAVStatus.RETURNING: u.status = UAVStatus.IDLE

    # ------------------------------------------------------------------ stepping
    @property
    def done(self) -> bool:
        over = all(v.status in (VictimStatus.ASSISTED, VictimStatus.FAILED) for v in self.victims)
        return over or self.t >= self.cfg.max_time

    def step(self) -> None:
        self.t += self.cfg.dt
        for v in self.victims:
            if v.status == VictimStatus.UNDETECTED and self.t >= v.detect_time:
                v.status = VictimStatus.DETECTED; v.next_localize_time = self.t
                self._log("detected", v.name, f"distress beacon; urgency {v.urgency}")
        for v in self.victims:
            if v.status == VictimStatus.DETECTED and self.t >= v.next_localize_time:
                self._localize(v)
        for v, u in allocate(self.victims, self.net.uavs, self.t):
            self._assign(v, u)
        for u in self.net.uavs:
            self._update_uav(u)
        self.net.update_comm()
        self._record()

    def run(self) -> "Simulation":
        while not self.done:
            self.step()
        return self

    def _record(self) -> None:
        self.history.append({
            "t": self.t,
            "pos": self.net.positions.copy(),
            "uav_status": [u.status.value for u in self.net.uavs],
            "battery": [u.battery for u in self.net.uavs],
            "comm": [u.comm_ok for u in self.net.uavs],
            "v_status": [v.status.value for v in self.victims],
            "v_est": [None if v.estimated_position is None else v.estimated_position.copy() for v in self.victims],
            "v_err": [v.error_m for v in self.victims],
            "v_links": [list(v.anchor_ids) if (v.localized_at is not None and self.t - v.localized_at <= 3.0) else []
                        for v in self.victims],
            "v_assisted": [v.assisted_at is not None and v.assisted_at <= self.t for v in self.victims],
        })

    # ------------------------------------------------------------------ outputs (all computed from sim state)
    def metrics(self, frame: int | None = None) -> dict:
        h = self.history[-1 if frame is None else frame]
        st = h["v_status"]
        loc_idx = [i for i, e in enumerate(h["v_est"]) if e is not None]
        est = np.array([h["v_est"][i] for i in loc_idx]).reshape(-1, 3)
        tru = np.array([self.victims[i].true_position for i in loc_idx]).reshape(-1, 3)
        summ = M.summarize(est, tru, self.cfg.success_threshold_m)
        detected = sum(s != VictimStatus.UNDETECTED.value for s in st)
        done = sum(s == VictimStatus.ASSISTED.value for s in st)
        rssi_vals = [np.nanmean(v.last_rssi) for v in self.victims if v.last_rssi is not None and np.isfinite(v.last_rssi).any()]
        links = [np.isfinite(v.last_rssi).mean() for v in self.victims if v.last_rssi is not None]
        return {
            "sim_time_s": h["t"],
            "active_uavs": sum(s != UAVStatus.CHARGING.value for s in h["uav_status"]),
            "victims_detected": detected,
            "victims_localized": len(loc_idx),
            "missions_completed": done,
            "mission_completion_rate": done / len(st) if st else float("nan"),
            "mean_error_m": summ["mean_error_m"], "rmse_3d_m": summ["rmse_3d_m"], "mse_3d_m": summ["mse_3d_m"], "mae_axis_m": summ["mae_axis_m"],
            "localization_success_rate": summ["success_rate"],
            "mean_rssi_dbm": float(np.mean(rssi_vals)) if rssi_vals else float("nan"),
            "link_availability": float(np.mean(links)) if links else float("nan"),
        }

    def victims_table(self) -> pd.DataFrame:
        rows = []
        for v in self.victims:
            e = v.estimated_position
            rows.append({"victim": v.name, "urgency": v.urgency, "supply": v.supply, "status": v.status.value,
                         "detect_time_s": v.detect_time, "true_x": v.true_position[0], "true_y": v.true_position[1],
                         "true_z": v.true_position[2],
                         "est_x": None if e is None else e[0], "est_y": None if e is None else e[1],
                         "est_z": None if e is None else e[2], "error_m": v.error_m,
                         "localized_ok": v.localization_ok, "anchors": [a + 1 for a in v.anchor_ids],
                         "attempts": v.attempts, "uav": None if v.uav_id is None else f"UAV-{v.uav_id + 1:02d}",
                         "mission_time_s": v.mission_time})
        return pd.DataFrame(rows)

    def uav_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"uav": u.name, "status": u.status.value, "battery_pct_simulated": round(u.battery, 1),
                              "x": u.position[0], "y": u.position[1], "altitude": u.position[2],
                              "comm_ok": u.comm_ok, "distance_flown_m": round(u.distance_flown, 1)} for u in self.net.uavs])

    def events_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)
