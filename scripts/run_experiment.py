"""Repeatable localization experiments.

  python scripts/run_experiment.py --config config/default.yaml --seeds 0 1 2 3 4 --noise-sweep --ablation
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from gennf.experiments import ExperimentConfig, noise_sweep, run_trials, summarize_trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--trials", type=int)
    ap.add_argument("--noise-sweep", action="store_true", help="error vs noise (0,1,2,4,6 dB)")
    ap.add_argument("--ablation", action="store_true", help="also run use_ga=False and geometry=pairwise variants")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    base = yaml.safe_load(Path(a.config).read_text())
    if a.trials: base["n_trials"] = a.trials
    out = Path(a.out); out.mkdir(exist_ok=True)
    variants = {"gennf_default": {}}
    if a.ablation:
        variants.update({"no_ga": {"use_ga": False}, "pairwise_geometry": {"geometry": "pairwise"}})
    all_sum, all_trials = [], []
    for name, over in variants.items():
        for seed in a.seeds:
            cfg = ExperimentConfig(**{**base, **over, "seed": seed}); df = run_trials(cfg)
            df["variant"], df["seed"] = name, seed; all_trials.append(df)
            s = summarize_trials(df, cfg.success_threshold_m); s["variant"], s["seed"] = name, seed; all_sum.append(s)
            print(f"[{name} seed={seed}] done")
    trials = pd.concat(all_trials); summary = pd.concat(all_sum)
    trials.to_csv(out / "trials.csv", index=False); summary.to_csv(out / "summary_per_seed.csv", index=False)
    pooled = summary.groupby(["variant", "method"])[["mean_error_m", "std_error_m", "rmse_3d_m", "mae_axis_m", "success_rate", "mean_runtime_ms"]].agg(["mean", "std"]).round(3)
    pooled.to_csv(out / "summary_pooled.csv"); print("\n=== mean (std across seeds) ===\n", pooled.to_string())
    if a.noise_sweep:
        sw = pd.concat([noise_sweep(ExperimentConfig(**{**base, "seed": s}), [0, 1, 2, 4, 6]).assign(seed=s) for s in a.seeds])
        sw.to_csv(out / "noise_sweep.csv", index=False)
        print("\n=== mean error (m) vs noise ===\n", sw.groupby(["noise_db", "method"]).error_m.mean().unstack().round(2).to_string())
    (out / "config_used.json").write_text(json.dumps({"base": base, "seeds": a.seeds, "variants": variants}, indent=2))


if __name__ == "__main__":
    main()
