#!/usr/bin/env bash
# Regenerates every result and figure from the scripts, in dependency order.
# Usage: bash run_all.sh /path/to/CMAPSSData [results_dir]
set -euo pipefail
DATA="${1:-/tmp/CMAPSSData}"
OUT="${2:-results}"
mkdir -p "$OUT/figures" "$OUT/parts"
S=src
python3 $S/e0_e1_baseline_and_leakage.py "$DATA" "$OUT"
python3 $S/e3_drift.py "$DATA" "$OUT"
python3 $S/e4_ood.py "$DATA" "$OUT"                      # writes e4_odd_boundary.json used below
python3 $S/e5_calibration.py "$DATA" "$OUT"
python3 $S/e2_coverage.py "$DATA" "$OUT"
python3 $S/e6_disparity.py "$DATA" "$OUT"
python3 $S/e7_robustness.py "$DATA" "$OUT"
for p in 42 7 11; do python3 $S/e8_seeds_bootstrap.py "$DATA" "$OUT" seeds_e0e1 $p; done
python3 $S/e8_seeds_bootstrap.py "$DATA" "$OUT" seeds_e4e5
python3 $S/e8_seeds_bootstrap.py "$DATA" "$OUT" bootstrap
python3 $S/e8_seeds_bootstrap.py "$DATA" "$OUT" merge
python3 $S/e9_threshold_sensitivity.py "$DATA" "$OUT"    # reads e8_seeds_bootstrap.json
for d in oracle cyclecount wholelife; do python3 $S/e10_drift_deployable.py "$DATA" "$OUT" $d; done
python3 $S/e10_drift_deployable.py "$DATA" "$OUT" merge
for m in M0 M1 M2 M3 M4; do python3 $S/e11_model_sweep.py "$DATA" "$OUT" $m; done   # M3, M4 need PyTorch
python3 $S/e11_model_sweep.py "$DATA" "$OUT" merge
echo "done: $OUT"
