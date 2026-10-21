"""LEGACY: literal port of the DATA FLOW of the original MATLAB script (`gen + ANN.txt`).

!!! NOT A VALID LOCALIZER. !!!  Kept only so the leakage can be demonstrated and compared.

Reproduced faithfully (data flow):
  1. a geometric estimate is computed from RSSI ... and then never used again (as in MATLAB line 573);
  2. a GA evolves 3-vectors using the TRUE target as the fitness reference (MATLAB lines 448-515);
  3. an ANN (3 -> 32 -> ReLU -> 3) is trained on GA samples with the TRUE target as constant label;
  4. the 'estimate' starts at the first GA sample and is stepped toward ANN predictions (lines 573-592).
Because step 2 uses ground truth, the final error says nothing about RSSI localization quality: it is
(nearly) independent of RSSI noise. Numerical bugs of the MATLAB file (z-index, constant 'noise',
float equality) are NOT reproduced here; the RSSI stage uses the corrected model.
"""
from __future__ import annotations

import numpy as np

from .localization import geometric_estimate, select_anchors
from .neural_model import MLP
from .rssi_model import RSSIModel, euclidean_distances


def run_original_pipeline(uav_positions, true_target, rssi_model: RSSIModel, rng: np.random.Generator,
                          bounds_hi=(10.0, 10.0, 10.0), pop=100, gens=50, mut=0.05, epochs=300) -> dict:
    uav = np.asarray(uav_positions, float); target = np.asarray(true_target, float)
    hi = np.asarray(bounds_hi, float)
    rssi = rssi_model.measure(euclidean_distances(uav, target), rng, apply_range=False)
    ids = select_anchors(rssi, 4)
    geo = geometric_estimate(uav[ids], rssi_model.distance_from_rssi(rssi[ids]))   # computed, then ignored

    population = rng.random((pop, 3)) * hi                                           # create_individual
    for _ in range(gens):
        fit = -np.sum((population - target) ** 2, axis=1)                            # <-- GROUND TRUTH IN FITNESS
        population = population[np.argsort(-fit)]
        new = []
        for _i in range(pop):
            p1, p2 = population[rng.integers(0, pop // 2)], population[rng.integers(0, pop // 2)]
            cut = int(rng.integers(1, 4)); child = np.concatenate([p1[:cut], p2[cut:]])
            if rng.random() < mut:
                child[rng.integers(0, 3)] += (rng.random() - 0.5) * hi.max() / 10.0
            new.append(child)
        population = np.array(new)
    X = population; Y = np.tile(target, (pop, 1))                                    # <-- TRUE TARGET AS LABEL
    tr, te = slice(0, int(0.8 * pop)), slice(int(0.8 * pop), pop)
    net = MLP(3, (32,), 3, rng)
    net.fit(X[tr] / hi.max(), Y[tr] / hi.max(), epochs=epochs, rng=rng)
    pred = net.predict(X[te] / hi.max()) * hi.max()
    est = X[0].copy(); tol = np.array([0.01, 0.02, 0.05]) * hi.max() / 10.0
    for p in pred:                                                                   # lines 578-587
        for c in range(3):
            for _ in range(10_000):
                if abs(p[c] - est[c]) <= tol[c]:
                    break
                est[c] += 0.2 * (p[c] - est[c])
    return {"estimate": est, "geometric_estimate_unused": geo, "uses_ground_truth": True,
            "error": float(np.linalg.norm(est - target)),
            "geometric_error": float(np.linalg.norm(geo - target))}
