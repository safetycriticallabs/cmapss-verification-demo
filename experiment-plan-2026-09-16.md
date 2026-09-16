# Pre-registered experiment plan, 2026-09-16

This file declares the configuration, acceptance criteria and seeds of every
experiment added or corrected for the Scientific Reports revision of
"Evidence-based certification of artificial intelligence in safety-critical
systems", before any of them is run. It is committed before the scripts that
implement it. Every number in `results/` is produced by those scripts.

Conventions shared by all experiments: FD001 task framing, 30-cycle windows,
unit-level 60/20/20 partitions with seed 42, standardisation fitted on the
training partition, baseline model MLP(128, 64), seed 0, unchanged from the
submitted manuscript. Acceptance criteria marked "representative" are
project-defined values used to make determinations concrete; they are not
claimed to be derived from a hazard analysis of this component.

## P0. Corrections to existing experiments

**E3 naive monitor designs (N1 to N4), now executed rather than recorded.**
Each is a declared variant of the compliant monitor (500-cycle trailing window,
10 quantile bins, step 25, aggregate max over the 13 continuous channels, alert
0.10, inhibit 0.25), differing in exactly one element:
- N1: baseline = training-fleet whole-life raw channels; monitored = test-fleet
  whole-life raw stream. Reported: clean-stream max PSI.
- N2: baseline = training-fleet nominal-regime (RUL > 100) raw channels, no
  asset normalisation; monitored = test-fleet nominal-regime raw stream.
  Reported: clean-stream max PSI.
- N3: asset-normalised residuals as in the compliant design, but monitored per
  asset with 60-cycle windows and 8 quantile bins. Reported: fraction of test
  assets with at least one alert crossing on the clean stream, fraction of
  windows above alert, and the approximate sampling floor (bins - 1) / window.
- N4: the compliant design with quantile-bin PSI applied to the integer-valued
  channel s17. Reported: clean-stream max PSI on s17.
The compliant design itself is not re-tuned. Its clean-stream alert-level
crossing (max PSI 0.124) is reported as measured.

**E4 out-of-distribution verification.**
- Sensor screen computed on the training partition only. If the screened set
  differs from the 14 channels used so far, the new set is used and reported.
- Detection threshold = 98th percentile of in-distribution scores on the
  validation partition. The false-positive rate on the test partition is
  reported as measured beside the 2% design value.
- Set A = all 260 FD002 training units (previously the first 60).
- Set B = FD003 late-life windows (RUL <= 100), decomposed by the fault-mode
  signature clustering below into fan-mode and HPC-mode units. Results are
  reported for fan-mode units, HPC-mode units, and pooled.

**Fault-mode signature clustering (ODD boundary validation, AI-6.1).**
Per-unit signature = (mean over the last 20 cycles minus mean over the first
20 cycles) divided by the standard deviation over the first 20 cycles, per
screened channel. Signatures of all FD001 and FD003 units are standardised
together. KMeans with k = 2, n_init = 20, random_state = 0 is fitted on the
FD003 signatures. The cluster whose centroid is nearer (mean Euclidean
distance) to the FD001 unit signatures is labelled HPC-mode; the other
fan-mode. Labels are fixed before any detector is evaluated. The clustering
result and a figure are retained as the evidence artifact.

## P1. Statistics

- Seed sweep: partition seeds {42, 7, 11} x model seeds {0, 1, 2, 3, 4} for
  E0/E1 (15 paired runs); model seeds {0, 1, 2, 3, 4} at partition seed 42 for
  E4 softmax results and E5. Reported: mean, standard deviation, min, max.
- Bootstrap: unit-level resampling with replacement, B = 1000, seed 2026, for
  unit-level test accuracy, macro-F1 and critical-class recall; window-level
  resampling for the row-level (leaked) accuracy; unit-level resampling of the
  in-distribution and out-of-distribution units for OOD detection rate and
  AUROC; unit-level resampling for ECE before and after temperature scaling.
  95% percentile intervals.

## P2. Additional requirement areas (A2)

- E2 coverage (AI-3.3): 2D PCA fitted on training windows; a 20 x 20 grid over
  the 1st to 99th percentile of each axis on the training partition; grid
  coverage = cells occupied by at least one test window / cells occupied by at
  least one training window. Also: fraction of test windows whose input-space
  Mahalanobis score lies below the 99th percentile of training scores.
  Representative criterion: grid coverage >= 0.90.
- E6 cross-regime disparity (AI-2.1): model of the baseline architecture
  trained on FD002 with inputs = screened channels plus the three operating
  settings, unit-level 60/20/20 split, seed 42, model seed 0. Regimes
  identified by KMeans k = 6 on the operating settings. Per-regime test
  accuracy and critical-class recall; disparity = max minus min.
  Representative criterion: accuracy disparity <= 5 percentage points.
- E7 robustness (AI-7.2): FGSM on the baseline MLP with the gradient computed
  from the fitted weights, epsilon in {0.02, 0.05, 0.10, 0.20} in standardised
  input units (L-infinity), and Gaussian noise of the same magnitudes; single
  channel stuck at its training mean, every channel in turn. Reported: test
  accuracy under each. Representative criterion: accuracy degradation <= 5
  percentage points at epsilon = 0.05.

## P3. Threshold sensitivity (A3)

- OOD: detection rate at false-positive budgets {1, 2, 5, 10}% for each
  detector and set; pass region over detection criteria {0.90 to 0.99} and
  budgets {1 to 10}%.
- Calibration: ECE before and after scaling for {10, 15, 20, 30} bins; pass or
  fail against criteria {0.02, 0.03, 0.05, 0.10}.
- Leakage: the inverted band [min honest accuracy, max leaked accuracy] over
  the seed sweep.
- Drift: alert and inhibit latency as a function of thresholds (alert 0.05 to
  0.20, inhibit 0.15 to 0.50) on the injected runs of P4.

## P4. Drift monitoring without oracle knowledge (A4)

- Deployable variants of the compliant monitor, same window, bins, step and
  thresholds: (i) cycle-count rule: monitored stream = cycles 21 to 60 of each
  test engine (asset baseline = first 20 cycles), baseline = training-fleet
  residuals over the same cycle range; (ii) whole-life asset-normalised
  residuals against a training-fleet whole-life residual baseline. Neither uses
  remaining useful life.
- Injection-magnitude sweep: full bias in {0.5, 1.0, 1.5, 2.0, 3.0} sigma on
  the four declared channels, linear ramp. For streams of at least 2,000 rows,
  onset at index 800 and ramp over 600 cycles as before; for shorter streams,
  onset at 40% of the stream length and ramp over 30% of it.
- Comparator detectors on identical windows: two-sample Kolmogorov-Smirnov
  statistic (max over channels) and maximum mean discrepancy (RBF kernel,
  bandwidth by the median heuristic on the baseline, baseline subsampled to
  500 rows, seed 0). For every detector including PSI, an alert threshold equal
  to the clean-stream maximum (zero clean false alarms by construction) is used
  to compare detection latency; PSI is additionally reported at the declared
  0.10 and 0.25 thresholds.

## P5. Model-family sweep (A1)

Same partitions, windows and standardisation. Models: M0 MLP(128, 64) as
submitted; M1 MLP(256, 256, 128), same optimiser and budget; M2 histogram
gradient-boosted trees (max_iter 300, learning_rate 0.1, random_state 0); M3
one-dimensional CNN (conv 14 to 32, kernel 5, ReLU; conv 32 to 64, kernel 5,
ReLU; global average pooling; linear 64 to 3; Adam 1e-3, batch 256, 30 epochs,
seed 0); M4 LSTM (hidden 64, one layer; linear 64 to 3; same optimiser, batch,
epochs and seed). M3 and M4 require PyTorch; the version used is recorded in
the results. Per model: E0/E1 leakage gap; E4 softmax AUROC and detection rate
on set A and set B fan-mode; feature-space class-conditional Mahalanobis on
the penultimate layer for M0, M1, M3 and M4 (PCA to 32 components when the
layer is wider, shared Ledoit-Wolf covariance); E5 ECE before and after
temperature scaling fitted on the validation partition. The input-space
Mahalanobis detector is model-independent and reported once.

## Outputs

Each experiment writes `results/<id>.json` and any figure to
`results/figures/`. Software versions are those pinned in `requirements.txt`,
plus PyTorch for M3 and M4.
