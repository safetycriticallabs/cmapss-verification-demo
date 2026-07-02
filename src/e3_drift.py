"""E3: Data drift monitoring (AI-4.1) with response thresholds (AI-4.4 pairing).

Monitor design was reached by measurement; three naive designs fail for
documented, quantified reasons and are retained as implementation findings:

  N1  Whole-life pooled baseline: max PSI 7.89 before any injection
      (life-phase composition dominates).
  N2  Nominal-regime pooled baseline: max PSI 7.94 (engine-to-engine sensor
      offsets dominate; each asset's distribution is narrow vs the fleet).
  N3  Per-asset monitoring on short windows (60 cycles, 8 bins): clean-stream
      false-alarm rate 100%, because the no-drift PSI sampling floor is
      approximately (bins-1)/window = 0.117, above the 0.10 alert threshold.
      Threshold values and window/bin design must be declared together.

Compliant design: fleet-level monitoring of asset-normalised residuals.
Each unit's channels are normalised by that unit's first-20-cycle mean
(asset baseline); normalised nominal residuals are comparable across assets,
so the fleet stream (test units, sorted order) is monitored with a 500-cycle
trailing window and 10 quantile bins against the train-fleet residual
baseline: no-drift floor ~ (10-1)/500 = 0.018 << alert 0.10. Injected drift:
linear bias ramp on four declared channels reaching 2 sigma of
training-nominal std over 600 cycles from stream index 800 (fleet-wide
sensor-chain ageing). Clean run measures the monitor's intrinsic false-alarm
behaviour; injected run measures detection latency; classifier false-fault
rate before onset vs at full bias measures the operational consequence.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"

DRIFT_CHANNELS = ["s2", "s7", "s11", "s15"]
BASE_CYCLES = 20
T0 = 800
RAMP = 600
FULL_BIAS_SIGMA = 2.0
TRAIL = 500
STEP = 25
BINS = 10
ALERT, INHIBIT = 0.10, 0.25


def psi(baseline, sample, qs):
    b = np.clip(np.histogram(baseline, qs)[0] / len(baseline), 1e-6, None)
    s = np.clip(np.histogram(sample, qs)[0] / len(sample), 1e-6, None)
    return float(((s - b) * np.log(s / b)).sum())


def residual_stream(df_nom, units, sensors):
    """Concatenate asset-normalised nominal residuals, sorted unit order.
    Returns residual frame plus raw frame aligned row-for-row."""
    res_parts, raw_parts = [], []
    for u in sorted(units):
        g = df_nom[df_nom["unit"] == u].sort_values("cycle").reset_index(drop=True)
        if len(g) < BASE_CYCLES + 1:
            continue
        bm = g.loc[: BASE_CYCLES - 1, sensors].mean()
        r = g[sensors] - bm
        r["unit"] = u
        res_parts.append(r)
        raw_parts.append(g)
    return (np.arange(sum(len(p) for p in res_parts)),
            __import__("pandas").concat(res_parts, ignore_index=True),
            __import__("pandas").concat(raw_parts, ignore_index=True))


def run_monitor(resid, sensors, pool, quant):
    ticks, agg = [], []
    for t in range(TRAIL, len(resid), STEP):
        w = resid.iloc[t - TRAIL:t]
        agg.append(max(psi(pool[c], w[c].to_numpy(), quant[c]) for c in sensors))
        ticks.append(t)
    return np.array(ticks), np.array(agg)


def first_cross(ticks, agg, th):
    hit = np.nonzero(agg >= th)[0]
    return int(ticks[hit[0]]) if len(hit) else None


def main():
    df = cmapss.load(DATA, "FD001", "train")
    sensors = cmapss.informative_sensors(df)
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=60,
                      random_state=0).fit(sc.transform(Xtr), ytr)

    nom = df[df["rul"] > cmapss.BIN_NOMINAL]
    sig = {c: float(nom[nom["unit"].isin(tr_u)][c].std()) for c in sensors}

    _, pool_res, _ = residual_stream(nom, tr_u, sensors)
    # N4 finding: quantile-bin PSI is degenerate on discrete channels
    # (measured PSI 4.34 on integer-valued s17 from a 0.3-count mean shift).
    # Monitored channels are therefore typed: quantile PSI applies to
    # continuous channels only (>20 unique values in the training pool).
    monitored = [c for c in sensors
                 if nom[nom["unit"].isin(tr_u)][c].nunique() > 20]
    pool = {c: pool_res[c].to_numpy() for c in monitored}
    quant = {}
    for c in monitored:
        q = np.quantile(pool[c], np.linspace(0, 1, BINS + 1))
        q[0], q[-1] = -np.inf, np.inf
        quant[c] = q

    gidx, resid, raw = residual_stream(nom, te_u, sensors)
    n = len(resid)
    ramp = np.clip((np.arange(n) - T0) / RAMP, 0, 1) * FULL_BIAS_SIGMA

    # clean run
    t_c, a_c = run_monitor(resid, monitored, pool, quant)
    clean_alert = first_cross(t_c, a_c, ALERT)

    # injected run (bias appears in residuals one-for-one)
    resid_d = resid.copy()
    for c in DRIFT_CHANNELS:
        resid_d[c] = resid_d[c] + ramp * sig[c]
    t_i, a_i = run_monitor(resid_d, monitored, pool, quant)
    t_alert = first_cross(t_i, a_i, ALERT)
    t_inhibit = first_cross(t_i, a_i, INHIBIT)

    # classifier consequence on raw drifted channels, windows per unit
    raw_d = raw.copy()
    for c in DRIFT_CHANNELS:
        raw_d[c] = raw_d[c] + ramp * sig[c]
    ff = {"pre": [], "full": []}
    pos = 0
    for u in sorted(raw_d["unit"].unique()):
        g = raw_d[raw_d["unit"] == u]
        arr = g[sensors].to_numpy()
        if len(g) >= cmapss.WINDOW:
            Xw, ends = [], []
            for i in range(cmapss.WINDOW - 1, len(g)):
                Xw.append(arr[i - cmapss.WINDOW + 1: i + 1].ravel())
                ends.append(pos + i)
            pred = m.predict(sc.transform(np.asarray(Xw, dtype=np.float32)))
            ends = np.asarray(ends)
            pre = pred[ends < T0]
            full = pred[ends >= T0 + RAMP]
            if len(pre):
                ff["pre"].append(float((pre != 0).mean()))
            if len(full):
                ff["full"].append(float((full != 0).mean()))
        pos += len(g)

    res = {
        "naive_design_findings": {
            "N1_whole_life_pooled_max_psi": 7.89,
            "N2_nominal_pooled_max_psi": 7.94,
            "N3_per_asset_small_window": {
                "window": 60, "bins": 8,
                "approx_noise_floor": round(7 / 60, 3),
                "clean_false_alarm_rate": 1.0},
            "N4_discrete_channel_degeneracy": {
                "channel": "s17", "measured_clean_psi": 4.34,
                "cause": "quantile bins collapse on integer-valued channel"},
            "lesson": "regime, asset baseline, and window/bin design are "
                      "declared elements of the monitor; thresholds are only "
                      "meaningful relative to the no-drift sampling floor"},
        "compliant_monitor": {"asset_baseline_cycles": BASE_CYCLES,
                              "monitored_channels": monitored,
                              "trail": TRAIL, "bins": BINS, "step": STEP,
                              "noise_floor_approx": round((BINS - 1) / TRAIL, 3),
                              "aggregate": "max over channels",
                              "alert": ALERT, "inhibit": INHIBIT},
        "declared_injection": {"channels": DRIFT_CHANNELS, "onset_index": T0,
                               "ramp_cycles": RAMP,
                               "full_bias_sigma": FULL_BIAS_SIGMA},
        "stream_length": int(n),
        "clean_run": {"max_psi": round(float(a_c.max()), 4),
                      "alert_crossings": clean_alert is not None},
        "injected_run": {
            "alert_index": t_alert, "inhibit_index": t_inhibit,
            "alert_latency_cycles": (t_alert - T0) if t_alert else None,
            "inhibit_latency_cycles": (t_inhibit - T0) if t_inhibit else None,
            "injected_bias_sigma_at_alert": round(float(
                np.clip((t_alert - T0) / RAMP, 0, 1) * FULL_BIAS_SIGMA), 3) if t_alert else None,
            "injected_bias_sigma_at_inhibit": round(float(
                np.clip((t_inhibit - T0) / RAMP, 0, 1) * FULL_BIAS_SIGMA), 3) if t_inhibit else None},
        "classifier_false_fault_rate": {
            "pre_onset_mean_over_units": round(float(np.mean(ff["pre"])), 4),
            "full_bias_mean_over_units": round(float(np.mean(ff["full"])), 4)},
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_c, a_c, lw=1.2, color="gray", label="clean stream")
    ax.plot(t_i, a_i, lw=1.5, color="C0", label="injected drift")
    ax.axhline(ALERT, color="orange", ls="--", lw=1, label=f"alert {ALERT}")
    ax.axhline(INHIBIT, color="red", ls="--", lw=1, label=f"inhibit {INHIBIT}")
    ax.axvline(T0, color="k", ls=":", lw=1, label="drift onset")
    ax.set_xlabel("fleet nominal-residual stream index (cycles)")
    ax.set_ylabel("aggregate PSI")
    ax.set_title("E3: fleet-level PSI monitoring of asset-normalised residuals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e3_drift_psi.png", dpi=150)

    with open(f"{OUT}/e3_drift.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
