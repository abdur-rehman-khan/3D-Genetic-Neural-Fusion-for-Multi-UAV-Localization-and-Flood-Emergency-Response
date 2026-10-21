import inspect

import numpy as np

from gennf.genetic_optimizer import GAConfig, genetic_optimize
from gennf.legacy_original import run_original_pipeline
from gennf.localization import (GenNFLocalizer, evaluate, geometric_estimate, least_squares_localize,
                                pair_point, select_anchors, trilaterate_triple)
from gennf.neural_model import MLP
from gennf.rssi_model import RSSIModel, euclidean_distances
from conftest import random_layout


# ---------------------------------------------------------------- anchor selection
def test_select_anchors_maps_to_original_uav_ids():
    rssi = np.array([-90, -50, np.nan, -60, -55, -70.0, -52, -80, -85, -95])
    ids = select_anchors(rssi, 4)
    assert list(ids) == [1, 6, 4, 3]                       # strongest first, original indices
    assert len(set(ids)) == 4


def test_select_anchors_ties_are_unique_and_stable():
    rssi = np.full(10, -60.0)
    ids = select_anchors(rssi, 4)
    assert list(ids) == [0, 1, 2, 3] and len(set(ids)) == 4


def test_select_anchors_fewer_than_k():
    assert len(select_anchors(np.array([np.nan, -50, np.nan]), 4)) == 1


# ---------------------------------------------------------------- geometry
def test_pair_point_coincident_anchors_is_finite():
    a = np.array([5.0, 5, 5])
    assert np.all(np.isfinite(pair_point(a, a.copy(), 3.0, 4.0)))


def test_geometric_estimate_degenerate_inputs_finite():
    p = np.array([[1, 1, 1.0]] * 4)                          # four coincident anchors
    assert np.all(np.isfinite(geometric_estimate(p, np.array([2, 3, 4, 5.0]))))
    p = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0.0]])   # collinear
    for mode in ("pairwise", "trilateration"):
        assert np.all(np.isfinite(geometric_estimate(p, np.array([5, 6, 15, 25.0]), mode)))
    assert np.all(np.isfinite(geometric_estimate(p, np.zeros(4))))     # zero ranges


def test_trilateration_exact_without_noise():
    tgt = np.array([40.0, 60.0, 2.0])
    p = np.array([[0, 0, 30], [100, 0, 40], [0, 100, 35.0]])
    r = euclidean_distances(p, tgt)
    assert np.allclose(trilaterate_triple(p, r), tgt, atol=1e-6)


def test_geometric_estimate_output_shape():
    p = np.random.default_rng(0).uniform(0, 100, (4, 3))
    assert geometric_estimate(p, np.array([30, 40, 50, 60.0])).shape == (3,)
    assert geometric_estimate(p[:3], np.array([30, 40, 50.0])).shape == (3,)


# ---------------------------------------------------------------- GenNF inference
def test_output_contract_and_no_ground_truth_in_api(fast_localizer):
    sig = inspect.signature(fast_localizer.localize)
    assert list(sig.parameters) == ["uav_positions", "rssi"]           # no truth parameter
    rng = np.random.default_rng(0); uav = random_layout(rng); tgt = np.array([100, 100, 5.0])
    rssi = fast_localizer.rssi_model.measure(euclidean_distances(uav, tgt), rng)
    res = fast_localizer.localize(uav, rssi)
    assert res.estimated_position.shape == (3,) and res.status == "ok"
    assert len(res.anchor_ids) == 4 and len(set(res.anchor_ids)) == 4
    assert res.rssi.shape == (10,) and res.estimated_distances.shape == (10,)
    assert not hasattr(res, "true_position")
    rec = evaluate(res, tgt)
    assert np.isclose(rec.error_m, np.linalg.norm(res.estimated_position - tgt))


def test_estimate_independent_of_truth_given_same_rssi(fast_localizer):
    rng = np.random.default_rng(1); uav = random_layout(rng)
    rssi = fast_localizer.rssi_model.measure(euclidean_distances(uav, np.array([50, 50, 3.0])), rng)
    a = fast_localizer.localize(uav, rssi).estimated_position
    b = fast_localizer.localize(uav, rssi.copy()).estimated_position
    assert np.array_equal(a, b)


def test_insufficient_anchors_status(fast_localizer):
    rng = np.random.default_rng(2); uav = random_layout(rng)
    rssi = np.full(10, np.nan); rssi[:2] = [-60, -70]
    res = fast_localizer.localize(uav, rssi)
    assert res.status == "insufficient_anchors" and np.all(np.isnan(res.estimated_position))


def test_estimate_stays_inside_deployment_volume(fast_localizer):
    rng = np.random.default_rng(3)
    for _ in range(20):
        uav = random_layout(rng); tgt = rng.uniform((0, 0, 0), (200, 200, 35))
        res = fast_localizer.localize(uav, fast_localizer.rssi_model.measure(euclidean_distances(uav, tgt), rng))
        if res.status == "ok":
            assert np.all(res.estimated_position >= -1e-9) and np.all(res.estimated_position <= np.array([200, 200, 35]) + 1e-9)


def test_duplicate_uav_coordinates_do_not_crash(fast_localizer):
    uav = np.tile(np.array([[100.0, 100, 30]]), (10, 1))
    rssi = fast_localizer.rssi_model.measure(euclidean_distances(uav, np.array([90, 90, 1.0])), np.random.default_rng(0))
    assert np.all(np.isfinite(fast_localizer.localize(uav, rssi).estimated_position))


def test_reproducible_training_with_fixed_seed():
    from gennf.localization import GenNFConfig
    cfg = GenNFConfig(n_train=200, epochs=5, ga=GAConfig(population_size=10, generations=3), bounds_hi=(200, 200, 35), seed=11)
    a = GenNFLocalizer(RSSIModel(), cfg); a.fit(random_layout)
    b = GenNFLocalizer(RSSIModel(), cfg); b.fit(random_layout)
    assert np.allclose(a.net.get_flat(), b.net.get_flat())


def test_least_squares_baseline_recovers_noise_free_target():
    rng = np.random.default_rng(4); m = RSSIModel(sigma_db=0.0, comm_range=1e9); uav = random_layout(rng)
    tgt = np.array([80.0, 120, 10])
    res = least_squares_localize(uav, m.measure(euclidean_distances(uav, tgt), rng), m)
    assert np.linalg.norm(res.estimated_position - tgt) < 1e-2


# ---------------------------------------------------------------- GA / ANN building blocks
def test_ga_improves_and_is_seeded():
    cfg = GAConfig(population_size=30, generations=20)
    fit = lambda v: -float(np.sum((v - 3.0) ** 2))
    pop = np.random.default_rng(0).normal(0, 1, (30, 4))
    r1 = genetic_optimize(fit, pop, cfg, np.random.default_rng(5)); r2 = genetic_optimize(fit, pop, cfg, np.random.default_rng(5))
    assert r1.history[-1] >= r1.history[0] and np.array_equal(r1.best, r2.best)


def test_mlp_learns_linear_map_and_flat_roundtrip():
    rng = np.random.default_rng(0); x = rng.normal(size=(300, 3)); y = x @ np.array([[1, 0, 0], [0, 2, 0], [0, 0, -1.0]])
    net = MLP(3, (16,), 3, rng); before = net.mse(x, y); net.fit(x, y, epochs=150, rng=rng)
    assert net.mse(x, y) < 0.1 * before
    v = net.get_flat(); net.set_flat(v * 1.0); assert np.allclose(net.get_flat(), v) and len(v) == net.n_params


# ---------------------------------------------------------------- original-code leakage (documented finding)
def test_legacy_pipeline_leaks_ground_truth():
    """The MATLAB data flow reaches ~zero error regardless of RSSI noise because the GA fitness uses the
    true target. The corrected pipeline cannot do this (see test_estimate_independent_of_truth...)."""
    rng = np.random.default_rng(0); uav = rng.uniform(0, 10, (10, 3)); tgt = np.array([6.0, 3.0, 4.0])
    out = run_original_pipeline(uav, tgt, RSSIModel(sigma_db=20.0, comm_range=1e9), rng, epochs=60)
    assert out["uses_ground_truth"] is True
    assert out["error"] < 0.5                                  # tiny error despite 20 dB noise ...
    assert out["geometric_error"] > out["error"]               # ... because the RSSI-based estimate was ignored
