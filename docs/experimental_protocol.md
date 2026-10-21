# Experimental protocol

## Setup (defaults in `config/default.yaml`)
* Area 200 × 200 m; targets uniform in x,y and z ∈ [0, 35] m; 10 UAVs, per-trial random layout (x,y uniform, z ∈ [22, 50] m).
* RSSI: P0 = −40 dBm @ 1 m, n = 2.7, Gaussian σ = 1 dB (default), communication range 150 m (out-of-range links are missing).
* GenNF trained offline on 3000 simulated targets/layouts (GA 100×50, then Adam 150 epochs), test trials drawn from a
  different random stream. Success threshold 20 m (configurable).
* Methods: `gennf`, `geometric_only` (ablation), `ls_same_anchors` and `ls_all_anchors` (nonlinear least squares baselines).
  Trials where a method cannot produce an estimate (e.g. < 4 in-range anchors for least squares) are counted in `n_failed`
  and excluded from that method's error statistics, so methods are **not** averaged over identical trial subsets.

## Reproduce
```
python scripts/run_experiment.py --seeds 0 1 2 3 4 --trials 200 --noise-sweep --ablation --out results
```

## Example run (5 seeds × 200 trials, σ = 1 dB) — newly generated, not published results
Mean over seeds of per-seed statistics (std across seeds in brackets):

| Method | Mean error (m) | RMSE 3D (m) | Success ≤ 20 m |
|---|---|---|---|
| GenNF (corrected, trilateration geometry) | 17.1 (1.0) | 22.0 (2.4) | 0.75 (0.03) |
| GenNF, no GA initialisation | 17.9 (1.2) | 22.4 (2.0) | 0.71 (0.03) |
| GenNF, `pairwise` geometry | 21.7 (0.6) | 26.0 (1.1) | 0.59 (0.02) |
| Geometric stage only | 19.6 (1.2) | 26.3 (2.8) | 0.67 (0.04) |
| Least squares, same 4 anchors | 24.0 (0.5) | 35.8 (1.8) | 0.61 (0.03) |
| Least squares, all in-range anchors | 18.4 (1.2) | 26.5 (2.7) | 0.71 (0.02) |

Mean 3D error (m) vs noise σ (GenNF trained at σ = 1 dB, tested at each level):

| σ (dB) | GenNF | geometric only | LS all | LS same 4 |
|---|---|---|---|---|
| 0 | 12.9 | 6.3 | 5.6 | 11.1 |
| 1 | 17.1 | 19.6 | 18.4 | 24.0 |
| 2 | 24.9 | 30.4 | 25.8 | 30.5 |
| 4 | 37.0 | 45.4 | 38.1 | 41.7 |
| 6 | 44.9 | 55.2 | 51.5 | 47.7 |

**Reading these numbers honestly**
* In this simulator GenNF is comparable to or modestly better than the baselines for σ ≥ 1 dB, but **worse than its own
  geometric stage and than least squares at σ = 0** (the ANN adds error on clean data).
* The GA initialisation changes mean error by ≈0.7 m, about one seed-standard-deviation: **no clear benefit demonstrated**.
* Absolute errors (≈17 m on a 200 m area at 1 dB) are large; they are driven by using only the four strongest links.
* One scenario, one propagation model. No claim of general superiority is made.

## Published results
Not reproduced here: the paper's numbers could not be inspected. When you verify them, add a table clearly labelled
"published" and keep it separate from the tables above.
