import numpy as np

from gennf.rssi_model import ORIGINAL_MATLAB, RSSIModel, euclidean_distances


def test_distance_uses_z_coordinate():
    # Regression for MATLAB line 27, which used mobileLoc(2) in the z term.
    d = euclidean_distances(np.array([[0, 0, 0.0]]), np.array([0, 3, 4.0]))
    assert d[0] == 5.0
    d2 = euclidean_distances(np.array([[0, 0, 0.0]]), np.array([0, 4, 3.0]))
    assert d2[0] == 5.0


def test_distance_basic_values():
    pts = np.array([[1, 2, 3], [4, 6, 3.0]])
    assert np.allclose(euclidean_distances(pts, np.array([1, 2, 3.0])), [0.0, 5.0])


def test_rssi_reference_and_slope():
    m = RSSIModel(p0_dbm=-40, n=2.0, d0=1.0, sigma_db=0.0)
    assert np.isclose(m.mean_rssi(1.0), -40.0)
    assert np.isclose(m.mean_rssi(10.0) - m.mean_rssi(1.0), -20.0)      # -10 n per decade


def test_inversion_round_trip():
    m = RSSIModel(sigma_db=0.0, n=3.1, p0_dbm=-35)
    d = np.array([2.0, 17.0, 120.0])
    assert np.allclose(m.distance_from_rssi(m.mean_rssi(d)), d)


def test_zero_distance_is_finite():
    m = RSSIModel(sigma_db=0.0)
    assert np.isfinite(m.mean_rssi(0.0))
    assert np.isfinite(m.distance_from_rssi(m.mean_rssi(0.0)))


def test_out_of_range_is_nan():
    m = RSSIModel(sigma_db=0.0, comm_range=50.0)
    r = m.measure(np.array([10.0, 60.0]))
    assert np.isfinite(r[0]) and np.isnan(r[1])


def test_noise_is_zero_mean_and_reproducible():
    m = RSSIModel(sigma_db=4.0)
    a = m.measure(np.full(5000, 20.0), np.random.default_rng(1))
    b = m.measure(np.full(5000, 20.0), np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert abs(np.mean(a) - m.mean_rssi(20.0)) < 0.3
    assert abs(np.std(a) - 4.0) < 0.3


def test_original_matlab_preset_parameters():
    assert (ORIGINAL_MATLAB.p0_dbm, ORIGINAL_MATLAB.n, ORIGINAL_MATLAB.d0, ORIGINAL_MATLAB.sigma_db) == (55.0, 1.0, 1.0, 7.0)
