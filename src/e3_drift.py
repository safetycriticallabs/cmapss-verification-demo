"""E3: Data drift monitoring (AI-4.1) with response thresholds (AI-4.4 pairing).

Monitor design was reached by measurement. Four naive designs are executed
here as declared variants of the compliant monitor, each differing in exactly
one element, and their clean-stream behaviour is measured and retained as
implementation findings (see experiment-plan-2026-09-16.md, P0):

  N1  Whole-life pooled raw baseline (life-phase composition dominates).
  N2  Nominal-regime pooled raw baseline, no asset normalisation
      (engine-to-engine sensor offsets dominate).
  N3  Per-asset monitoring on short windows (60 cycles, 8 bins): the no-drift
      PSI sampling floor is approximately (bins-1)/window = 0.117, above the
      0.10 alert threshold, so clean streams alarm.
  N4  Quantile-bin PSI applied to the integer-valued channel s17: quantile
      edges collapse on a discrete channel and the statistic degenerates.

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


def make_quant(pool, channels, bins):
    quant = {}
    for c in channels:
        q = np.quantile(pool[c], np.linspace(0, 1, bins + 1))
        q[0], q[-1] = -np.inf, np.inf
        quant[c] = q
    return quant


def raw_stream(df_rows, units, sensors):
    """Concatenate raw (unnormalised) channels of the given rows, sorted unit
    order, cycle order."""
    parts = [df_rows[df_rows["unit"] == u].sort_values("cycle")[sensors]
             for u in sorted(units)]
    return __import__("pandas").concat(parts, ignore_index=True)


def naive_designs(df, nom, tr_u, te_u, sensors, monitored, pool_res, resid):
    """Execute N1 to N4 on the clean (uninjected) streams and return the
    measured findings. Each variant changes one element of the compliant
    design; window (TRAIL), bins (BINS) and step (STEP) are held unless the
    variant declares otherwise."""
    out = {}
    # N1: whole-life raw baseline and whole-life raw monitored stream.
    pool_n1 = {c: df[df["unit"].isin(tr_u)][c].to_numpy() for c in monitored}
    q_n1 = make_quant(pool_n1, monitored, BINS)
    _, a_n1 = run_monitor(raw_stream(df, te_u, sensors), monitored, pool_n1, q_n1)
    out["N1_whole_life_raw_baseline"] = {
        "clean_max_psi": round(float(a_n1.max()), 3),
        "cause": "life-phase composition of the pooled baseline"}
    # N2: nominal-regime raw baseline, no asset normalisation.
    pool_n2 = {c: nom[nom["unit"].isin(tr_u)][c].to_numpy() for c in monitored}
    q_n2 = make_quant(pool_n2, monitored, BINS)
    _, a_n2 = run_monitor(raw_stream(nom, te_u, sensors), monitored, pool_n2, q_n2)
    out["N2_nominal_raw_baseline"] = {
        "clean_max_psi": round(float(a_n2.max()), 3),
        "cause": "engine-to-engine sensor offsets, no asset baseline"}
    # N3: per-asset monitoring of asset-normalised residuals, 60-cycle
    # windows, 8 quantile bins, alert 0.10.
    win, bins3 = 60, 8
    pool_n3 = {c: pool_res[c].to_numpy() for c in monitored}
    q_n3 = make_quant(pool_n3, monitored, bins3)
    assets_alarm, n_assets, n_win, n_win_alert = 0, 0, 0, 0
    for u in sorted(resid["unit"].unique()):
        r = resid[resid["unit"] == u].reset_index(drop=True)
        if len(r) < win:
            continue
        n_assets += 1
        vals = []
        for t in range(win, len(r) + 1, STEP):
            w = r.iloc[t - win:t]
            vals.append(max(psi(pool_n3[c], w[c].to_numpy(), q_n3[c])
                            for c in monitored))
        vals = np.array(vals)
        n_win += len(vals)
        n_win_alert += int((vals >= ALERT).sum())
        assets_alarm += int((vals >= ALERT).any())
    out["N3_per_asset_short_window"] = {
        "window": win, "bins": bins3, "step": STEP,
        "approx_noise_floor": round((bins3 - 1) / win, 3),
        "assets_monitored": n_assets,
        "clean_false_alarm_rate_assets": round(assets_alarm / max(n_assets, 1), 3),
        "clean_false_alarm_rate_windows": round(n_win_alert / max(n_win, 1), 3),
        "cause": "no-drift sampling floor of the statistic exceeds the alert threshold"}
    # N4: compliant design applied to the integer-valued channel s17.
    if "s17" in sensors:
        pool_n4 = {"s17": pool_res["s17"].to_numpy()}
        q_n4 = make_quant(pool_n4, ["s17"], BINS)
        _, a_n4 = run_monitor(resid, ["s17"], pool_n4, q_n4)
        out["N4_discrete_channel_degeneracy"] = {
            "channel": "s17",
            "unique_values_in_training_pool": int(
                np.unique(np.round(pool_n4["s17"], 6)).size),
            "clean_max_psi": round(float(a_n4.max()), 3),
            "cause": "quantile bins collapse on an integer-valued channel"}
    out["lesson"] = ("regime, asset baseline, and window/bin design are "
                     "declared elements of the monitor; thresholds are only "
                     "meaningful relative to the no-drift sampling floor")
    return out


def main():
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=60,
                      random_state=0).fit(sc.transform(Xtr), ytr)

    nom = df[df["rul"] > cmapss.BIN_NOMINAL]
    sig = {c: float(nom[nom["unit"].isin(tr_u)][c].std()) for c in sensors}

    _, pool_res, _ = residual_stream(nom, tr_u, sensors)
    # N4 finding (measured below): quantile-bin PSI is degenerate on discrete
    # channels. Monitored channels are therefore typed: quantile PSI applies to
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
    # Attributable alert: the first tick at or above the alert level at which
    # the injected series differs from the clean series. A crossing at which
    # the two coincide is the clean stream's own excursion, not a detection.
    attributable = np.nonzero((a_i >= ALERT) & (a_i > a_c + 1e-9))[0]
    t_alert_attr = int(t_i[attributable[0]]) if len(attributable) else None
    t_depart = np.nonzero(a_i > a_c + 1e-9)[0]
    t_first_departure = int(t_i[t_depart[0]]) if len(t_depart) else None

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

    # N1 to N4, executed on the clean streams (experiment plan P0).
    naive = naive_designs(df, nom, tr_u, te_u, sensors, monitored, pool_res, resid)

    res = {
        "naive_design_findings": naive,
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
                      "alert_crossings": clean_alert is not None,
                      "first_alert_index": clean_alert,
                      "ticks_at_or_above_alert": int((a_c >= ALERT).sum()),
                      "ticks_total": int(len(a_c)),
                      "inhibit_crossings": first_cross(t_c, a_c, INHIBIT) is not None},
        "injected_run": {
            "alert_index": t_alert, "inhibit_index": t_inhibit,
            "alert_latency_cycles": (t_alert - T0) if t_alert else None,
            "inhibit_latency_cycles": (t_inhibit - T0) if t_inhibit else None,
            "injected_bias_sigma_at_alert": round(float(
                np.clip((t_alert - T0) / RAMP, 0, 1) * FULL_BIAS_SIGMA), 3) if t_alert else None,
            "injected_bias_sigma_at_inhibit": round(float(
                np.clip((t_inhibit - T0) / RAMP, 0, 1) * FULL_BIAS_SIGMA), 3) if t_inhibit else None,
            "first_index_injected_departs_from_clean": t_first_departure,
            "alert_index_attributable_to_injection": t_alert_attr,
            "alert_latency_attributable_cycles": (t_alert_attr - T0) if t_alert_attr else None,
            "injected_bias_sigma_at_attributable_alert": round(float(
                np.clip((t_alert_attr - T0) / RAMP, 0, 1) * FULL_BIAS_SIGMA), 3) if t_alert_attr else None,
            "note": "the first alert-level crossing of the injected run coincides with the "
                    "clean stream's own excursion when alert_index equals the clean run's "
                    "first_alert_index; use the attributable values for detection latency"},
        "classifier_false_fault_rate": {
            "pre_onset_mean_over_units": round(float(np.mean(ff["pre"])), 4),
            "full_bias_mean_over_units": round(float(np.mean(ff["full"])), 4)},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.35, 1]})
    for k, ax in enumerate(axes):
        ax.plot(t_c, a_c, lw=1.2, color="gray", label="clean stream")
        ax.plot(t_i, a_i, lw=1.5, color="C0", label="injected drift")
        ax.axhline(ALERT, color="orange", ls="--", lw=1, label=f"alert {ALERT}")
        ax.axhline(INHIBIT, color="red", ls="--", lw=1, label=f"inhibit {INHIBIT}")
        ax.axvline(T0, color="k", ls=":", lw=1, label="drift onset")
        if t_alert_attr:
            ax.axvline(t_alert_attr, color="C0", ls=":", lw=1, label="attributable alert")
        if t_inhibit:
            ax.axvline(t_inhibit, color="red", ls=":", lw=1, label="inhibit crossing")
        ax.set_xlabel("fleet nominal-residual stream index (cycles)")
        ax.text(-0.08, 1.04, "ab"[k], transform=ax.transAxes, fontsize=12, fontweight="bold")
    axes[0].set_yscale("log"); axes[0].set_ylabel("aggregate PSI (log scale)")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].set_xlim(T0 - 100, T0 + 500); axes[1].set_ylim(0, 0.4)
    axes[1].set_ylabel("aggregate PSI")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e3_drift_psi.png", dpi=200)

    with open(f"{OUT}/e3_drift.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
