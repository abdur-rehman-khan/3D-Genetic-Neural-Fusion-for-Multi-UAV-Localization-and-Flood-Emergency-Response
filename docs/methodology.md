# Methodology

> **Status of this document.** It describes what is implemented in this repository and how it relates to the
> MATLAB source `gen + ANN.txt`. The full text of the IEEE Access paper *RSSI Guided Localization Using Genetic
> Neural Fusion in Multi-UAV Network* could **not** be accessed while this code was written (only the abstract was
> available), so nothing here has been verified against the paper's equations, architecture, parameters or
> reported results. Please compare Section 2 against the paper before publishing claims.

## 1. Three clearly separated layers

| Layer | Code | Status |
|---|---|---|
| **Legacy** – data flow of the MATLAB script | `legacy_original.py` | Reproduced only to *demonstrate* a ground-truth leak. **Not a valid localizer.** |
| **GenNF (corrected)** – leak-free RSSI → position pipeline | `rssi_model.py`, `localization.py`, `genetic_optimizer.py`, `neural_model.py` | My reconstruction of GenNF from the MATLAB structure and the paper abstract. |
| **Flood-rescue extension** – environment, UAV motion, allocation, dashboard | `flood_environment.py`, `uav_network.py`, `rescue_allocation.py`, `simulation.py`, `visualization.py`, `app.py` | **New.** Not part of the published method. |

## 2. Audit of `gen + ANN.txt`

### 2.1 Fundamental issue: ground truth is an input to the "learned" stage
* GA fitness (`fitness_function`, lines 448-459) is `-‖X − Y‖²` with `Y = mobileLoc`, the **true target**. The GA
  therefore evolves candidate vectors toward the answer.
* The ANN is trained on the GA output (`X`) with the true target as constant label (`Y = repmat(Y_initial, …)`).
* The RSSI-derived geometric estimate (`XM, YM, ZM`) is never used afterwards: lines 573-575 overwrite it with `X(1,:)`,
  a GA sample, which is then stepped toward ANN predictions (lines 578-587).
* Consequence: the printed error measures GA convergence to a value it was given, not RSSI-based localization.
  `tests/test_localization.py::test_legacy_pipeline_leaks_ground_truth` shows ≈0 error even with 20 dB RSSI noise.
* Whether the paper's experiments share this structure is **unknown** to me; the abstract describes a GA + ANN that
  "adjusts weights", which is not what this file does (the GA generates data, it does not tune weights).

### 2.2 Numerical / implementation bugs (corrected in Python)
| # | MATLAB location | Problem | Python handling |
|---|---|---|---|
| 1 | l.27 | z-distance uses `mobileLoc(2)` instead of `(3)` | `euclidean_distances` uses x,y,z (regression test) |
| 2 | l.41 | "noise" is `+sigma`, a constant 7 dB offset → every range is scaled by 10^(-0.7)≈0.2 | zero-mean Gaussian noise, `RSSIModel.sigma_db` |
| 3 | l.32/31 | n = 1, P0 = 55 used as both "path loss" and received power | configurable `p0_dbm`, `n`; `ORIGINAL_MATLAB` preset for reference |
| 4 | l.53-64 | anchors recovered by float equality `C(k)==d2(i)` → duplicates/extra rows on ties (likely with integer coordinates) | `select_anchors` (stable argsort on RSSI, returns unique UAV IDs) |
| 5 | l.73-97 … | exact float equality tests (`Dab == C1+C2`) never fire | continuous rule in `pair_point` |
| 6 | l.82, 108, … | `abs(a-b) < D < (a+b)` is parsed by MATLAB as `(abs(a-b)<D) < (a+b)` | removed |
| 7 | l.77, 103 | `-` instead of `+` in the z term of the "separated spheres" point | corrected |
| 8 | l.106, 171, 238 | `R1(3,1)` used where `R1(3,2)`/other index intended; l.238 pasted from another pair; l.291, 356 test wrong `C(·)` | rewritten once, generically |
| 9 | l.84-91 … | division by `(R1(1,1)-R1(2,1))^2` → zero when two anchors share an x (≈10 % on a 1-10 integer grid) | no such division |
| 10 | l.5-12 | coincident anchors / target on an anchor possible (`randi` with replacement); `log10(0)` | `MIN_DISTANCE_M` guard; degenerate-geometry tests |
| 11 | l.606-608 | `Pe13` is a per-axis *mean squared* error printed as metres | `metrics.py` defines 3D RMSE, mean error, per-axis MAE explicitly |
| 12 | ANN | no input normalisation; 80/20 split of GA samples | inputs normalised; training on independent simulated targets |

## 3. GenNF as implemented (corrected, leak-free)

**Offline phase** (uses *simulated* labelled targets only):
1. Sample random UAV layouts and random targets; generate RSSI with the log-distance model.
2. Run the geometric stage on each sample → features; label = (true position − geometric estimate) / scale.
3. **Genetic algorithm** (population 100, 50 generations, mutation rate 0.05 — the MATLAB values — plus elitism and
   per-gene Gaussian mutation) searches the ANN's flat weight vector; fitness = −MSE on a training subset.
4. Adam fine-tunes the best individual. (`use_ga=False` gives the ablation.)

**Inference** (never receives ground truth — the signature is `localize(uav_positions, rssi)`):
1. `d̂ = d0·10^((P0 − RSSI)/(10 n))`.
2. Select the 4 strongest links (→ UAV IDs).
3. **Geometric stage**, structure kept from the MATLAB script: triples ABC, BCD, CDA, DAB; inside a triple the three
   pair estimates are combined with weights 1/(rᵢ+rⱼ); the four triples are combined with weights 1/(rᵢ+rⱼ+rₖ).
   Two per-triple rules are available:
   * `pairwise` — pair estimate is a point on the anchor–anchor line (segment point if spheres are disjoint,
     radical-plane foot otherwise). My simplification of MATLAB's branches (the intersecting-spheres formula A…E
     there is affected by bugs 5-9 and I could not establish its intent). **It is a biased estimator: ≈33 m error
     without noise in the benchmark scenario.**
   * `trilateration` — **corrected variant**: closed-form 3-sphere intersection (lower solution; targets are below
     the airborne anchors). Default for the simulator.
4. **ANN refinement** (3→32→ReLU→3 as in MATLAB when `feature_mode="geom3"`; `"extended"` adds anchor
   coordinates and ranges, 19 inputs): `x̂ = geometric estimate + scale·net(features)`, clipped to the deployment volume.

**Interpretation caveat.** "GA tunes ANN weights" follows the abstract's wording, not verified paper details.

## 4. Flood-rescue extension (new)
Procedural environment; UAV kinematics (constant speed, patrol around home positions, detour around hazard
cylinders); simulated battery; victim distress beacons at scheduled times; RSSI generated from the hidden true
position; localization by the GenNF module; **greedy allocation** (urgency, then nearest feasible idle UAV);
descent, delivery timer, return. A drop succeeds if the *estimate* lies within a configurable drop-zone radius of the
true victim (assumption); otherwise the victim is re-localized (max 3 attempts).
Ground truth is used only to synthesize RSSI and to evaluate (`tests/test_simulation.py::test_groundtruth_never_passed_to_localizer`).

## 5. Metrics
Errors are 3D Euclidean (m). **RMSE** = √mean(‖est−true‖²) (3D position RMSE, not per-coordinate). **MAE** = mean
|est−true| over targets *and* axes. Mean error and success rate (error ≤ threshold) are reported separately.

## 6. Limitations
Synthetic log-distance RSSI (no multipath/NLOS/body shadowing, no antenna pattern), no UAV dynamics beyond constant
speed, simulated battery, no real communication stack. **Synthetic results do not establish field performance.**
This is not a real emergency-response system.
