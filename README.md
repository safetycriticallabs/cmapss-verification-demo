# C-MAPSS Applied Verification Demonstration

Applied demonstration accompanying a journal manuscript in submission on a
failure-mode-driven requirements and verification framework for assuring
artificial intelligence in safety-critical systems. A deliberately ordinary
health-state classifier is trained on NASA C-MAPSS turbofan data and the
framework's verifications are executed against it; the object of study is the
verification evidence produced, not model performance.

Framework: https://doi.org/10.5281/zenodo.19024420

## Data

NASA C-MAPSS turbofan degradation simulation (Saxena et al., PHM 2008).
Public domain. Fetch:

    git clone --depth 1 https://github.com/edwardzjl/CMAPSSData /tmp/CMAPSSData

Task framing: 3-class health state (nominal RUL>100 / degraded 30<RUL<=100 /
critical RUL<=30) from 30-cycle windows over the 14 informative FD001 sensor
channels (selected by variance screen; matches the canonical literature set).
Model: MLP (128, 64), adam, batch 256, up to 60 epochs, seed 0; converges at
52 epochs. Unit-wise splits 60/20/20, seed 42; the sensor screen is computed on
the training partition. Every reported number is a measurement produced by the
scripts; the experiments added for the revision were declared in
`experiment-plan-2026-09-16.md` before they were run. `bash run_all.sh DATA`
regenerates everything in dependency order (M3 and M4 of E11 need PyTorch,
tested with torch 2.14.0 CPU).

## Experiments

| ID | Framework requirement | Script | Status |
|---|---|---|---|
| E0 | Baseline on compliant unit-wise partitions (AI-1.1) | `src/e0_e1_baseline_and_leakage.py` | done |
| E1 | Leakage counterfactual, row-level split (AI-1.1 violated) | same | done |
| E2 | Coverage of the declared envelope by the test partition (AI-3.3) | `src/e2_coverage.py` | done |
| E3 | Drift injection + PSI monitoring, four naive designs executed (AI-4.1/4.4) | `src/e3_drift.py` | done |
| E4 | OOD detection with ODD boundary validation (AI-6.1/6.2/6.3, AI-12.5) | `src/e4_ood.py` | done |
| E5 | Calibration ECE + temperature scaling (AI-8.3, AI-12.2) | `src/e5_calibration.py` | done |
| E6 | Cross-regime performance disparity on FD002 (AI-2.1) | `src/e6_disparity.py` | done |
| E7 | Input robustness: FGSM, Gaussian noise, stuck-at channel (AI-7.2, AI-3.4) | `src/e7_robustness.py` | done |
| E8 | Seed sweep and unit-level bootstrap intervals | `src/e8_seeds_bootstrap.py` | done |
| E9 | Threshold sensitivity of every determination | `src/e9_threshold_sensitivity.py` | done |
| E10 | Drift monitoring without oracle knowledge; magnitude sweep; KS and MMD comparators | `src/e10_drift_deployable.py` | done |
| E11 | Model-family sweep (MLP, deeper MLP, gradient-boosted trees, 1D-CNN, LSTM) | `src/e11_model_sweep.py` | done |

Results land in `results/*.json` (intermediate parts in `results/parts/`);
figures in `results/figures/`.

## Measured results

**E0/E1 (partitioning).** Unit-wise (honest) test: accuracy 0.8033 (unit-level
bootstrap 95% interval 0.768 to 0.836), macro-F1 0.8226, critical-class recall
0.9048. Row-level (leaked) split: 0.8305 (0.818 to 0.842). Measured leakage
inflation +2.72 pp at seed 0; over 15 runs (three partition seeds, five model
seeds) the gap is positive in every run, mean +2.36 pp, sd 0.96, range +0.45 to
+3.78. In each run any acceptance threshold between the two values is passed
by the leaked evaluation and failed by the honest one; the interval of
thresholds inverted in every run is 0.817 to 0.818, so the inversion is a
property of each evaluation pair, not of one fixed threshold.

**E3 (drift monitoring).** Four naive designs executed on the clean stream:
N1 whole-life raw baseline saturates (max PSI 4.25, life-phase composition);
N2 nominal-regime raw baseline without asset normalisation saturates (3.35,
engine-to-engine offsets); N3 per-asset 60-cycle windows with 8 bins alarm on
every one of 17 monitored assets and every window (sampling floor
(bins-1)/window = 0.117 exceeds the 0.10 alert); N4 quantile-bin PSI on the
integer-valued channel s17 degenerates (4.34 on a clean stream). Compliant
design (fleet-level monitoring of asset-normalised residuals, 13 continuous
channels, 500-cycle window, floor 0.018): clean-stream max PSI 0.124, with 7
of 64 evaluation ticks at or above the 0.10 alert and none at the 0.25
inhibit. On the injected 2-sigma ramp the first alert-level crossing (index
975, 0.58 sigma) coincides exactly with the clean stream's own excursion on
channel s9 and is not a detection; the injected series first departs from the
clean series at 0.92 sigma (attributable alert, +275 cycles) and crosses the
inhibit level at 1.17 sigma (+350 cycles). Classifier false-fault-indication
rate rises from 0.154 (pre-onset) to 0.698 (full bias).

**E4 (OOD detection, with ODD boundary validation).** FD003 holds two fault
modes and no per-unit label. Clustering per-unit degradation signatures
splits its 100 units into 57 whose signature matches FD001's HPC degradation
and 43 that do not (fan mode); the labels are fixed before detection and
retained in `results/e4_odd_boundary.json`. Thresholds are the 98th
percentile of validation scores; measured test FPR is 0.019 (Mahalanobis) and
0.025 (softmax). Softmax confidence (MSP) is inverted as an OOD score: AUROC
0.117 on the six unseen operating regimes (all 260 FD002 units) and 0.227 on
the fan-mode units; the network is more confident outside its ODD than inside
it. The class-conditional Mahalanobis detector (32-component PCA,
Ledoit-Wolf) passes on unseen regimes (TPR 1.000, AUROC 1.000) and narrowly
misses the representative criterion on the fan-mode units (TPR 0.962, AUROC
0.991; bootstrap interval 0.928 to 0.988 straddles the 0.97 criterion); on the
HPC-mode FD003 units, which are inside the declared ODD, it flags 4.4% (an
independent in-ODD fleet check against a 1.9% test FPR). The pooled set B
result of the original submission (TPR 0.442) was the mixture of these two
groups and is superseded.

**E5 (calibration).** Raw network ECE 0.160 (FAIL against <= 0.05; fitted
temperature 5.07). After temperature scaling on the validation partition: ECE
0.0295 (PASS on the point estimate; unit-level bootstrap interval 0.016 to
0.059 straddles the criterion), test accuracy unchanged at 0.8033. Stable
across five model seeds (ECE after 0.024 to 0.031).

**E2 (coverage).** The 20-engine test partition reaches 67% of the
training-occupied cells of a 20 x 20 grid over the first two principal
components (81% of training mass), FAIL against the representative 0.90;
99.1% of test windows lie inside the training Mahalanobis envelope.

**E6 (cross-regime disparity, FD002).** Per-regime test accuracy ranges over
4.1 pp and critical-class recall over 4.7 pp across the six regimes: PASS
against the representative 5 pp.

**E7 (robustness).** FGSM at epsilon 0.05 (standardised units) drops accuracy
from 0.803 to 0.419 (38 pp): FAIL against the representative 5 pp; Gaussian
noise of the same magnitude leaves accuracy at 0.803. The worst stuck-at
channel (s7) drops accuracy to 0.679.

**E9 (threshold sensitivity).** The fan-mode Mahalanobis determination passes
at 0.90 and 0.95 detection criteria at every false-positive budget and
reaches 0.97 only at a 10% budget; the calibration determination after
scaling passes at 0.03 and 0.05 for 10 to 30 bins and fails at 0.02; the
drift alert threshold lies inside the clean-stream range at 0.10 and outside
it at 0.125 or above with no change in attributable latency.

**E10 (drift without oracle knowledge).** With every detector's alert set at
its own clean-stream maximum, the oracle (regime-selected) design detects the
2-sigma ramp at 0.92 sigma (PSI), 1.00 (KS) and 0.83 (MMD). Neither
deployable variant produced a stable clean stream: the cycle-count variant
(cycles 21 to 60) has clean max PSI 0.36 and detects at 1.29 sigma (PSI) or
1.08 (KS); the whole-life variant saturates on the clean stream (max PSI 2.73)
and detects only at full bias. Health-state conditioning of the baseline is
what makes the compliant monitor work, and it is not available without a
health-state estimate in service.

**E11 (model-family sweep).** Leakage gap, softmax inversion and calibration
across five model classes (partition 42, seed 0):

| model | acc | leak gap | MSP AUROC A / B-fan | feature-Mahalanobis TPR A / B-fan | ECE before / after |
|---|---|---|---|---|---|
| M0 MLP(128,64) | 0.803 | +2.7 pp | 0.117 / 0.227 | 1.000 / 0.964 | 0.160 / 0.029 |
| M1 MLP(256,256,128) | 0.812 | +2.0 pp | 0.123 / 0.306 | 1.000 / 0.978 | 0.160 / 0.032 |
| M2 gradient-boosted trees | 0.799 | +4.5 pp | 0.508 / 0.515 | n/a | 0.132 / 0.024 |
| M3 1D-CNN | 0.728 | +14.8 pp | 0.003 / 0.055 | 1.000 / 1.000 | 0.150 / 0.042 |
| M4 LSTM | 0.770 | +14.9 pp | 0.911 / 0.746 | 0.958 / 0.958 | 0.166 / 0.058 |

The determinations are model-dependent: the sequence models leak an order of
magnitude more under row-level splitting; softmax is inverted for the
feed-forward and convolutional networks, uninformative for the trees, and
informative but insufficient for the LSTM; the LSTM's feature-space detector
fails the criterion on both sets and its calibration remediation leaves ECE
at 0.058, above the criterion.
