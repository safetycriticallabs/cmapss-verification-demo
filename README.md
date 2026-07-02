# C-MAPSS Applied Verification Demonstration

Applied demonstration for "A failure-mode-driven requirements and verification
framework for assuring artificial intelligence in safety-critical systems"
(RESS submission). A deliberately ordinary health-state classifier is trained
on NASA C-MAPSS turbofan data and the framework's verifications are executed
against it; the object of study is the verification evidence produced, not
model performance.

Framework: https://doi.org/10.5281/zenodo.19024420

## Data

NASA C-MAPSS turbofan degradation simulation (Saxena et al., PHM 2008).
Public domain. Fetch:

    git clone --depth 1 https://github.com/edwardzjl/CMAPSSData /tmp/CMAPSSData

Task framing: 3-class health state (nominal RUL>100 / degraded 30<RUL<=100 /
critical RUL<=30) from 30-cycle windows over the 14 informative FD001 sensor
channels (selected by variance screen; matches the canonical literature set).
Model: MLP (128, 64), adam, batch 256, up to 60 epochs, seed 0; converges at
52 epochs. Unit-wise splits 60/20/20, seed 42. All configuration pre-declared;
every reported number is a measurement.

## Experiments

| ID | Framework requirement | Script | Status |
|---|---|---|---|
| E0 | Baseline on compliant unit-wise partitions (AI-1.1) | `src/e0_e1_baseline_and_leakage.py` | done |
| E1 | Leakage counterfactual, row-level split (AI-1.1 violated) | same | done |
| E3 | Drift injection + PSI monitoring (AI-4.1/4.4) | `src/e3_drift.py` | done |
| E4 | OOD detection, FD002 regimes + FD003 fan fault (AI-6.1/6.2/6.3, AI-12.5) | `src/e4_ood.py` | done |
| E5 | Calibration ECE + temperature scaling (AI-8.3, AI-12.2) | `src/e5_calibration.py` | done |
| E2/E6 | Coverage quantification / cross-regime disparity | - | optional, not run |

Results land in `results/*.json`; figures in `results/figures/`.

## Measured results

**E0/E1 (partitioning).** Unit-wise (honest) test: accuracy 0.8033, macro-F1
0.8226, critical-class recall 0.9048. Row-level (leaked) split: 0.8305.
Measured leakage inflation +2.72 pp: any acceptance threshold between the two
values is passed by the leaked evaluation and failed by the honest one.

**E3 (drift monitoring).** Monitor qualification required four measured
design iterations, retained as findings: N1 whole-life pooled baseline
saturates (max PSI 7.89, life-phase composition); N2 nominal-regime pooled
baseline saturates (7.94, asset-to-asset sensor offsets); N3 per-asset
60-cycle windows false-alarm 100% (no-drift sampling floor (bins-1)/window =
0.117 exceeds the 0.10 alert threshold); N4 quantile-bin PSI degenerates on a
discrete channel (PSI 4.34 on integer-valued s17). Compliant design
(fleet-level monitoring of asset-normalised residuals, 13 continuous
channels, 500-cycle window, floor 0.018): clean stream max PSI 0.124, one
transient alert-level brush, no inhibit-level excursion; injected 2-sigma
ramp detected at +175 cycles (0.58 sigma) for alert and +350 cycles (1.17
sigma) for inhibit. Classifier false-fault-indication rate rises from 0.154
(pre-onset) to 0.698 (full bias): the operational consequence the inhibit
threshold pre-empts.

**E4 (OOD detection).** Softmax confidence (MSP) is worse than chance as an
OOD score: AUROC 0.117 on unseen operating regimes (FD002) and 0.352 on the
unseen fan-fault mode (FD003 late-life). The network is systematically MORE
confident outside its ODD, the measured form of the confidently-wrong
pathology, confirming AI-12.5's insufficiency assertion. The class-conditional
Mahalanobis detector (32-component PCA, Ledoit-Wolf) passes the representative
criterion (>=97% detection at <=2% FPR) on unseen regimes (TPR 1.000, AUROC
1.000) and FAILS it on the near-ODD fan-fault case (TPR 0.442, AUROC 0.767).
A system claiming that criterion with this detector is denied certification
on measured evidence; the near-ODD failure motivates AI-6.3 response
requirements and AI-5.5 independent validation.

**E5 (calibration).** Raw network ECE 0.160 (FAIL against <=0.05, fitted
temperature 5.07: grossly overconfident). After temperature scaling on the
validation partition: ECE 0.0295 (PASS), test accuracy unchanged at 0.8033.
Verification catches the inadequacy; the AI-12.2 remediation closes it.
