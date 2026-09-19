"""E10: Drift monitoring without oracle knowledge, injection-magnitude sweep,
and comparator detectors (experiment plan P4).

Designs (same asset baseline of 20 cycles, 10 quantile bins, step 25, alert
0.10, inhibit 0.25 for PSI):
  oracle      the compliant E3 monitor: nominal-regime (RUL > 100) residual
              stream, which uses remaining useful life to select the regime
              and is therefore not deployable as written.
  cyclecount  deployable variant (i): cycles 21 to 60 of each test engine,
              baseline = training-fleet residuals over the same cycles. No
              remaining-useful-life knowledge.
  wholelife   deployable variant (ii): whole-life asset-normalised residuals
              against a training-fleet whole-life residual baseline.
Declared deviation from the plan for `cyclecount`: the stream is 800 rows,
so a 500-cycle trailing window would leave 13 evaluation ticks and no
pre-onset period; the window is scaled to 25% of the stream (200 cycles;
sampling floor (10-1)/200 = 0.045, below the 0.10 alert). It is recorded in
the results as a deviation.

Injection: linear bias ramp on the four declared channels to a full bias in
{0.5, 1.0, 1.5, 2.0, 3.0} sigma; onset 800 and ramp 600 for streams of at
least 2,000 rows, else onset at 40% and ramp over 30% of the stream.

Detectors on identical windows: PSI (max over channels), two-sample KS
statistic (max over channels), and RBF-kernel MMD^2 (bandwidth by the median
heuristic on a 500-row baseline subsample, seed 0). For every detector the
alert threshold used for latency comparison is the clean-stream maximum
(zero clean false alarms by construction); PSI is also reported at the
declared 0.10 and 0.25 thresholds with the attributable-alert rule of E3.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss
import e3_drift as e3

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(f"{OUT}/parts", exist_ok=True)
DESIGN = sys.argv[3] if len(sys.argv) > 3 else "merge"
MAGS = [0.5, 1.0, 1.5, 2.0, 3.0]
BASE = e3.BASE_CYCLES
MMD_N, MMD_SEED = 500, 0


def resid_rows(rows, units, sensors):
    parts = []
    for u in sorted(units):
        g = rows[rows["unit"] == u].sort_values("cycle").reset_index(drop=True)
        if len(g) < BASE + 1:
            continue
        r = g[sensors] - g.loc[:BASE - 1, sensors].mean()
        r["unit"] = u
        r["cycle"] = g["cycle"].to_numpy()
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def streams(design, df, tr_u, te_u, sensors):
    if design == "oracle":
        nom = df[df["rul"] > cmapss.BIN_NOMINAL]
        base_rows, mon_rows = nom, nom
        keep = lambda r: r
    elif design == "cyclecount":
        base_rows, mon_rows = df, df
        keep = lambda r: r[(r["cycle"] > BASE) & (r["cycle"] <= 60)].reset_index(drop=True)
    elif design == "wholelife":
        base_rows, mon_rows = df, df
        keep = lambda r: r
    else:
        raise ValueError(design)
    pool = keep(resid_rows(base_rows, tr_u, sensors))
    mon = keep(resid_rows(mon_rows, te_u, sensors))
    return pool, mon


def run_detectors(mon, monitored, pool, quant, trail, step, mmd_sub, bw):
    ticks, out = [], {"PSI": [], "KS": [], "MMD": []}
    P = pool[monitored].to_numpy()
    for t in range(trail, len(mon), step):
        W = mon.iloc[t - trail:t]
        out["PSI"].append(max(e3.psi(pool[c].to_numpy(), W[c].to_numpy(), quant[c]) for c in monitored))
        out["KS"].append(max(ks_2samp(pool[c].to_numpy(), W[c].to_numpy()).statistic for c in monitored))
        out["MMD"].append(mmd2(mmd_sub, W[monitored].to_numpy(), bw))
        ticks.append(t)
    return np.array(ticks), {k: np.array(v) for k, v in out.items()}


def mmd2(X, Y, bw):
    def k(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d / (2 * bw ** 2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def main_design(design):
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    nom = df[df["rul"] > cmapss.BIN_NOMINAL]
    sig = {c: float(nom[nom["unit"].isin(tr_u)][c].std()) for c in sensors}
    monitored = [c for c in sensors if nom[nom["unit"].isin(tr_u)][c].nunique() > 20]
    pool, mon = streams(design, df, tr_u, te_u, sensors)
    n = len(mon)
    trail = e3.TRAIL if n >= 2000 else int(round(0.25 * n))
    t0, ramp_len = (e3.T0, e3.RAMP) if n >= 2000 else (int(0.4 * n), int(0.3 * n))
    quant = e3.make_quant({c: pool[c].to_numpy() for c in monitored}, monitored, e3.BINS)
    rng = np.random.default_rng(MMD_SEED)
    sub = pool[monitored].to_numpy()[rng.choice(len(pool), size=min(MMD_N, len(pool)), replace=False)]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    bw = float(np.median(d[np.triu_indices(len(sub), 1)]))
    ticks, clean = run_detectors(mon, monitored, pool, quant, trail, e3.STEP, sub, bw)
    res = {"design": design, "stream_rows": int(n), "units": int(mon["unit"].nunique()),
           "trail": trail, "step": e3.STEP, "bins": e3.BINS, "onset": t0, "ramp": ramp_len,
           "deviation_from_plan": ("trailing window scaled to 25% of an 800-row stream"
                                   if design == "cyclecount" else None),
           "noise_floor_approx": round((e3.BINS - 1) / trail, 3),
           "mmd_bandwidth": round(bw, 4), "ticks": int(len(ticks)),
           "clean": {k: {"max": round(float(v.max()), 4),
                         "psi_ticks_at_or_above_alert": int((v >= e3.ALERT).sum()) if k == "PSI" else None,
                         "psi_ticks_at_or_above_inhibit": int((v >= e3.INHIBIT).sum()) if k == "PSI" else None}
                     for k, v in clean.items()},
           "injected": {}}
    thr_clean = {k: float(v.max()) for k, v in clean.items()}
    series = {"clean": clean}
    for mag in MAGS:
        ramp = np.clip((np.arange(n) - t0) / ramp_len, 0, 1) * mag
        mon_d = mon.copy()
        for c in e3.DRIFT_CHANNELS:
            mon_d[c] = mon_d[c] + ramp * sig[c]
        _, inj = run_detectors(mon_d, monitored, pool, quant, trail, e3.STEP, sub, bw)
        series[str(mag)] = inj
        row = {}
        for k in ("PSI", "KS", "MMD"):
            hit = np.nonzero((inj[k] > thr_clean[k]) & (inj[k] > clean[k] + 1e-12))[0]
            t_det = int(ticks[hit[0]]) if len(hit) else None
            row[k] = {"clean_max_threshold": round(thr_clean[k], 4),
                      "latency_cycles": (t_det - t0) if t_det is not None else None,
                      "bias_sigma_at_detection": round(float(np.clip((t_det - t0) / ramp_len, 0, 1) * mag), 3)
                      if t_det is not None else None}
        attr = np.nonzero((inj["PSI"] >= e3.ALERT) & (inj["PSI"] > clean["PSI"] + 1e-9))[0]
        t_a = int(ticks[attr[0]]) if len(attr) else None
        t_i = e3.first_cross(ticks, inj["PSI"], e3.INHIBIT)
        row["PSI_declared_thresholds"] = {
            "attributable_alert_latency_cycles": (t_a - t0) if t_a is not None else None,
            "bias_sigma_at_attributable_alert": round(float(np.clip((t_a - t0) / ramp_len, 0, 1) * mag), 3) if t_a is not None else None,
            "inhibit_latency_cycles": (t_i - t0) if t_i is not None else None,
            "bias_sigma_at_inhibit": round(float(np.clip((t_i - t0) / ramp_len, 0, 1) * mag), 3) if t_i is not None else None}
        res["injected"][str(mag)] = row
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, k in zip(ax, ("PSI", "KS", "MMD")):
        a.plot(ticks, clean[k], color="gray", lw=1.2, label="clean")
        for mag in MAGS:
            a.plot(ticks, series[str(mag)][k], lw=1, label=f"{mag} sigma")
        a.axhline(thr_clean[k], color="k", ls="--", lw=0.8, label="clean max")
        a.axvline(t0, color="k", ls=":", lw=0.8)
        a.set_yscale("log"); a.set_title(f"{design}: {k}"); a.set_xlabel("stream index")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e10_drift_{design}.png", dpi=150)
    json.dump(res, open(f"{OUT}/parts/e10_part_{design}.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


def merge():
    res = {d: json.load(open(f"{OUT}/parts/e10_part_{d}.json")) for d in ("oracle", "cyclecount", "wholelife")}
    json.dump(res, open(f"{OUT}/e10_drift_deployable.json", "w"), indent=2)
    for d, r in res.items():
        print(d, "clean PSI max", r["clean"]["PSI"]["max"], "| 2.0 sigma:",
              {k: v.get("bias_sigma_at_detection") for k, v in r["injected"]["2.0"].items() if k != "PSI_declared_thresholds"},
              "| declared PSI:", r["injected"]["2.0"]["PSI_declared_thresholds"])


if __name__ == "__main__":
    if DESIGN == "merge":
        merge()
    else:
        main_design(DESIGN)
