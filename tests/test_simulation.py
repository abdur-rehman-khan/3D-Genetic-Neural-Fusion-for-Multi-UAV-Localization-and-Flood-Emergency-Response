import numpy as np
import pandas as pd

from gennf.rescue_allocation import Victim, VictimStatus, allocate
from gennf.rssi_model import RSSIModel
from gennf.simulation import SimConfig, Simulation
from gennf.uav_network import UAVStatus, make_formation

FAST = dict(n_train=500, epochs=15, n_victims=4, max_time=200.0, rssi=RSSIModel(sigma_db=0.5))


def make_sim(**kw):
    return Simulation(SimConfig(**{**FAST, **kw}))


def test_default_has_ten_uavs():
    assert SimConfig().n_uavs == 10
    assert all(len(make_formation(f, 10, 200.0)) == 10 for f in ("random", "ring", "grid", "two_layer"))


def test_uavs_visibly_move():
    s = make_sim(); p0 = s.net.positions.copy()
    for _ in range(20): s.step()
    assert np.linalg.norm(s.net.positions - p0, axis=1).max() > 5.0
    assert len(s.history) == 21


def test_speed_limit_respected():
    s = make_sim()
    for _ in range(40): s.step()
    steps = np.array([np.linalg.norm(b["pos"] - a["pos"], axis=1).max() for a, b in zip(s.history, s.history[1:])])
    assert steps.max() <= s.cfg.speed_mps * s.cfg.dt + 1e-9


def test_end_to_end_mission_state_transitions():
    s = make_sim().run()
    assert s.done
    log = s.events_table()
    for kind in ("detected", "localized", "assigned"):
        assert kind in set(log.event)
    # transitions can happen within one step, so verify ordering on the event log rather than sampled statuses
    order = {"detected": 0, "localized": 1, "assigned": 2, "assisted": 3}
    for v in s.victims:
        seq = [order[e] for e in log[log.who == v.name].event if e in order]
        assert seq == sorted(seq) or "delivery_missed" in set(log[log.who == v.name].event)
        assert seq[:3] == [0, 1, 2]
    assert {"undetected", "assigned"} <= {st for h in s.history for st in h["v_status"]}
    assert any(v.status == VictimStatus.ASSISTED for v in s.victims)
    for v in s.victims:
        if v.status == VictimStatus.ASSISTED:
            assert v.mission_time is not None and v.mission_time > 0


def test_no_uav_serves_two_victims_simultaneously():
    s = make_sim(n_victims=8)
    while not s.done:
        s.step()
        ids = [u.victim_id for u in s.net.uavs if u.victim_id is not None]
        assert len(ids) == len(set(ids))


def test_allocation_policy_priority_and_exclusivity():
    class U:  # minimal UAV stub
        def __init__(self, pos): self.position = np.array(pos, float); self.battery = 90.0; self.available = True
    vs = []
    for i, (urg, x) in enumerate([(1, 10), (3, 20), (2, 30)]):
        v = Victim(i, np.array([x, 0, 0.0]), urg, "first-aid kit", float(i)); v.status = VictimStatus.LOCALIZED
        v.estimated_position = v.true_position.copy(); vs.append(v)
    pairs = allocate(vs, [U([0, 0, 30]), U([100, 0, 30])], now=10.0)
    assert [p[0].id for p in pairs] == [1, 2]                  # most urgent first; only 2 UAVs available
    assert len({id(p[1]) for p in pairs}) == 2


def test_groundtruth_never_passed_to_localizer():
    s = make_sim(); calls = []
    orig = s.localizer.localize
    def spy(uav_positions, rssi):
        calls.append((uav_positions, rssi)); return orig(uav_positions, rssi)
    s.localizer.localize = spy; s.run()
    truths = [v.true_position for v in s.victims]
    for uav, rssi in calls:
        assert uav.shape == (10, 3) and rssi.shape == (10,)
        assert not any(np.allclose(row, t) for row in uav for t in truths)


def test_metrics_come_from_simulation_state_and_csv_export(tmp_path):
    s = make_sim().run(); m = s.metrics(); t = s.victims_table()
    loc = t.dropna(subset=["error_m"])
    assert m["victims_localized"] == len(loc)
    assert np.isclose(m["mean_error_m"], loc.error_m.mean())
    est = loc[["est_x", "est_y", "est_z"]].to_numpy(); tru = loc[["true_x", "true_y", "true_z"]].to_numpy()
    assert np.isclose(m["rmse_3d_m"], np.sqrt(np.mean(np.sum((est - tru) ** 2, axis=1))))
    assert m["missions_completed"] == int((t.status == "assisted").sum())
    p = tmp_path / "victims.csv"; t.to_csv(p, index=False)
    back = pd.read_csv(p); assert list(back.columns) == list(t.columns) and len(back) == len(t)


def test_same_seed_same_outcome():
    a = make_sim(seed=7).run(); b = make_sim(seed=7).run()
    assert np.allclose(a.history[-1]["pos"], b.history[-1]["pos"]) and a.victims_table().equals(b.victims_table())


def test_changing_node_count():
    s = make_sim(n_uavs=8, n_victims=2)
    s.run(); assert len(s.net.uavs) == 8
