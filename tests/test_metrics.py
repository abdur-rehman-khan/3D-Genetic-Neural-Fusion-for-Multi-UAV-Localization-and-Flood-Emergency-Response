import numpy as np
import pytest

from gennf import metrics as M


def test_position_errors_3d():
    e = M.position_errors(np.array([[3, 4, 0.0]]), np.array([[0, 0, 0.0]]))
    assert e[0] == 5.0


def test_rmse_is_3d_euclidean_not_per_axis():
    est = np.array([[3, 4, 0.0], [0, 0, 0.0]]); tru = np.zeros((2, 3))
    assert np.isclose(M.rmse_3d(est, tru), np.sqrt((25 + 0) / 2))


def test_mean_mae_and_success():
    est = np.array([[1, 1, 1.0], [2, 0, 0.0]]); tru = np.zeros((2, 3))
    assert np.isclose(M.mae_per_axis(est, tru), 5 / 6)
    assert np.isclose(M.mean_error(est, tru), (np.sqrt(3) + 2) / 2)
    assert M.success_rate(M.position_errors(est, tru), 1.8) == 0.5


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.position_errors(np.zeros((2, 3)), np.zeros((3, 3)))


def test_summarize_empty_is_nan_not_crash():
    s = M.summarize(np.zeros((0, 3)), np.zeros((0, 3)), 10.0)
    assert s["n"] == 0 and np.isnan(s["rmse_3d_m"])
