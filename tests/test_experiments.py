import numpy as np

from gennf.experiments import ExperimentConfig, run_trials, summarize_trials

FAST = dict(n_trials=15, n_train=400, epochs=10, ga_population=10, ga_generations=3)


def test_experiment_reproducible_and_summarised():
    a = run_trials(ExperimentConfig(seed=2, **FAST)); b = run_trials(ExperimentConfig(seed=2, **FAST))
    assert np.allclose(a.error_m.fillna(-1), b.error_m.fillna(-1))
    s = summarize_trials(a, 20.0)
    assert set(s.method) == {"gennf", "geometric_only", "ls_same_anchors", "ls_all_anchors"}
    assert (s.rmse_3d_m >= s.mean_error_m - 1e-9).all()        # RMSE >= mean error (Jensen)


def test_too_few_available_anchors_counts_as_failures():
    df = run_trials(ExperimentConfig(seed=1, n_available=3, **FAST))
    s = summarize_trials(df, 20.0).set_index("method")
    assert s.loc["ls_all_anchors", "n_failed"] == 15           # least squares needs >= 4 anchors


def test_fixed_geometry_deployment_runs():
    df = run_trials(ExperimentConfig(seed=1, deployment="ring", **FAST))
    assert df[df.method == "gennf"].error_m.notna().all()
