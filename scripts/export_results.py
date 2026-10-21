"""Run the default 10-UAV flood-rescue simulation headlessly and export CSVs + the interactive 3D scene.

  python scripts/export_results.py --seed 42 --out results
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gennf.rssi_model import RSSIModel
from gennf.simulation import SimConfig, Simulation
from gennf.visualization import build_figure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--victims", type=int, default=6); ap.add_argument("--out", default="results")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    sim = Simulation(SimConfig(seed=a.seed, n_victims=a.victims, rssi=RSSIModel(sigma_db=a.noise))).run()
    sim.victims_table().to_csv(out / "victims.csv", index=False); sim.uav_table().to_csv(out / "uavs.csv", index=False)
    sim.events_table().to_csv(out / "events.csv", index=False)
    build_figure(sim).write_html(out / "scene.html", include_plotlyjs="cdn")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sim.metrics().items()})


if __name__ == "__main__":
    main()
