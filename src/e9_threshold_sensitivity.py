"""E9: Threshold sensitivity of the determinations (experiment plan P3).

Every acceptance criterion in this demonstration is a representative,
project-defined value. This script shows how each determination moves across
a declared range of criteria, so that the sensitivity of a pass or fail to the
chosen number is itself evidence.
  OOD: detection rate at false-positive budgets {1, 2, 5, 10}% (thresholds
       from the validation partition) for each detector and set; pass region
       over detection criteria 0.90 to 0.99 and budgets 1 to 10%.
  Calibration: ECE before and after temperature scaling for {10, 15, 20, 30}
       bins; pass or fail against criteria {0.02, 0.03, 0.05, 0.10}.
  Leakage: the range of honest and leaked accuracies over the seed sweep
       (reads results/e8_seeds_bootstrap.json).
  Drift: alert and inhibit latency versus threshold values, on the injected
       run of E3 (alert 0.05 to 0.20, inhibit 0.15 to 0.50).
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss
import e3_drift as e3

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
FPR_BUDGETS = [0.01, 0.02, 0.05, 0.10]
TPR_CRITERIA = [0.90, 0.95, 0.97, 0.99]
ECE_BINS = [10, 15, 20, 30]
ECE_CRITERIA = [0.02, 0.03, 0.05, 0.10]
ALERTS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
INHIBITS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def ece(conf, correct, bins):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.sum():
            e += (mk.sum() / len(conf)) * abs(correct[mk].mean() - conf[mk].mean())
    return float(e)


def main():
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xva, yva, _, _ = cmapss.windows(df, va_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    Str = sc.transform(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=60,
                      random_state=0).fit(Str, ytr)
    res = {}

    # ---- OOD ----
    pca = PCA(n_components=32, random_state=0).fit(Str)
    Ztr = pca.transform(Str)
    means = {c: Ztr[ytr == c].mean(axis=0) for c in np.unique(ytr)}
    prec = LedoitWolf().fit(np.vstack([Ztr[ytr == c] - means[c] for c in means])).precision_

    def mah(X):
        Z = pca.transform(sc.transform(X)); d = []
        for c, mu in means.items():
            r = Z - mu; d.append(np.einsum("ij,jk,ik->i", r, prec, r))
        return np.sqrt(np.min(np.stack(d, axis=1), axis=1))
    msp = lambda X: -m.predict_proba(sc.transform(X)).max(axis=1)
    d2 = cmapss.load(DATA, "FD002", "train")
    X2, _, _, _ = cmapss.windows(d2, d2["unit"].unique(), sensors)
    d3 = cmapss.load(DATA, "FD003", "train")
    labels = json.load(open(f"{OUT}/e4_odd_boundary.json"))["labels"]
    d3l = d3[d3["rul"] <= 100]
    d3l = d3l.assign(cycle=d3l.groupby("unit").cumcount() + 1)
    X3, _, u3, _ = cmapss.windows(d3l, d3l["unit"].unique(), sensors)
    X3f = X3[np.array([labels[str(int(u))] == "fan" for u in u3])]
    res["ood"] = {}
    for name, fn in (("MSP", msp), ("MAH", mah)):
        s_val, s_te = fn(Xva), fn(Xte)
        for tag, Xo in (("A_fd002_regimes", X2), ("B_fd003_fan_mode", X3f)):
            s = fn(Xo)
            row = {}
            for b in FPR_BUDGETS:
                thr = np.quantile(s_val, 1 - b)
                tpr = float((s > thr).mean())
                row[f"budget_{b:.2f}"] = {"tpr": round(tpr, 4),
                                          "fpr_measured_on_test": round(float((s_te > thr).mean()), 4),
                                          "pass_at": [c for c in TPR_CRITERIA if tpr >= c]}
            res["ood"][f"{name}_{tag}"] = row

    # ---- Calibration ----
    lva = np.log(np.clip(m.predict_proba(sc.transform(Xva)), 1e-12, 1))
    lte = np.log(np.clip(m.predict_proba(sc.transform(Xte)), 1e-12, 1))
    nll = lambda T: -np.log(np.clip(softmax(lva / T, axis=1)[np.arange(len(yva)), yva], 1e-12, 1)).mean()
    T = float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)
    res["calibration"] = {"temperature": round(T, 3)}
    for tag, temp in (("before", 1.0), ("after", T)):
        p = softmax(lte / temp, axis=1); conf = p.max(1); corr = (p.argmax(1) == yte).astype(float)
        res["calibration"][tag] = {}
        for nb in ECE_BINS:
            e = ece(conf, corr, nb)
            res["calibration"][tag][f"bins_{nb}"] = {"ece": round(e, 4),
                                                     "pass_at": [c for c in ECE_CRITERIA if e <= c]}

    # ---- Leakage band from the seed sweep ----
    try:
        sw = json.load(open(f"{OUT}/e8_seeds_bootstrap.json"))["seed_sweep_e0e1"]
        runs = sw["runs"]
        res["leakage"] = {"runs": len(runs),
                          "honest_range": [min(r["unit_level_acc"] for r in runs), max(r["unit_level_acc"] for r in runs)],
                          "leaked_range": [min(r["row_level_acc"] for r in runs), max(r["row_level_acc"] for r in runs)],
                          "per_run_inversion_band": [[r["unit_level_acc"], r["row_level_acc"]] for r in runs],
                          "thresholds_inverted_in_every_run": [round(max(r["unit_level_acc"] for r in runs), 4),
                                                               round(min(r["row_level_acc"] for r in runs), 4)],
                          "note": "a threshold in the last interval is passed by the leaked evaluation and failed by the honest one in every run; if the interval is empty no single threshold inverts every run"}
    except FileNotFoundError:
        res["leakage"] = "run e8 first"

    # ---- Drift latency vs thresholds (E3 injected run, 2 sigma) ----
    nom = df[df["rul"] > cmapss.BIN_NOMINAL]
    sig = {c: float(nom[nom["unit"].isin(tr_u)][c].std()) for c in sensors}
    _, pool_res, _ = e3.residual_stream(nom, tr_u, sensors)
    monitored = [c for c in sensors if nom[nom["unit"].isin(tr_u)][c].nunique() > 20]
    pool = {c: pool_res[c].to_numpy() for c in monitored}
    quant = e3.make_quant(pool, monitored, e3.BINS)
    _, resid, _ = e3.residual_stream(nom, te_u, sensors)
    n = len(resid)
    ramp = np.clip((np.arange(n) - e3.T0) / e3.RAMP, 0, 1) * e3.FULL_BIAS_SIGMA
    t_c, a_c = e3.run_monitor(resid, monitored, pool, quant)
    resid_d = resid.copy()
    for c in e3.DRIFT_CHANNELS:
        resid_d[c] = resid_d[c] + ramp * sig[c]
    t_i, a_i = e3.run_monitor(resid_d, monitored, pool, quant)
    sig_at = lambda t: round(float(np.clip((t - e3.T0) / e3.RAMP, 0, 1) * e3.FULL_BIAS_SIGMA), 3) if t else None
    res["drift"] = {"clean_max_psi": round(float(a_c.max()), 4), "alert": {}, "inhibit": {}}
    for th in ALERTS:
        attr = np.nonzero((a_i >= th) & (a_i > a_c + 1e-9))[0]
        t_attr = int(t_i[attr[0]]) if len(attr) else None
        res["drift"]["alert"][str(th)] = {"clean_false_alarm_ticks": int((a_c >= th).sum()),
                                          "attributable_latency_cycles": (t_attr - e3.T0) if t_attr else None,
                                          "bias_sigma_at_attributable_alert": sig_at(t_attr)}
    for th in INHIBITS:
        t = e3.first_cross(t_i, a_i, th)
        res["drift"]["inhibit"][str(th)] = {"clean_false_alarm_ticks": int((a_c >= th).sum()),
                                            "latency_cycles": (t - e3.T0) if t else None,
                                            "bias_sigma_at_inhibit": sig_at(t)}

    # figure: OOD TPR vs budget and calibration ECE vs bins
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for key, row in res["ood"].items():
        ax[0].plot(FPR_BUDGETS, [row[f"budget_{b:.2f}"]["tpr"] for b in FPR_BUDGETS], marker="o", label=key)
    ax[0].axhline(0.97, color="k", ls="--", lw=1, label="0.97 criterion")
    ax[0].set_xscale("log"); ax[0].set_xlabel("false-positive budget"); ax[0].set_ylabel("detection rate")
    ax[0].legend(fontsize=7); ax[0].set_title("OOD determination vs budget")
    for tag in ("before", "after"):
        ax[1].plot(ECE_BINS, [res["calibration"][tag][f"bins_{nb}"]["ece"] for nb in ECE_BINS], marker="o", label=f"ECE {tag}")
    for c in ECE_CRITERIA:
        ax[1].axhline(c, color="gray", ls=":", lw=0.8)
    ax[1].set_xlabel("ECE bins"); ax[1].set_ylabel("ECE"); ax[1].legend(fontsize=8); ax[1].set_title("calibration determination vs binning")
    fig.suptitle("E9: threshold sensitivity")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e9_threshold_sensitivity.png", dpi=150)
    json.dump(res, open(f"{OUT}/e9_threshold_sensitivity.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "leakage"}, indent=1)[:6000])
    print("leakage:", json.dumps({k: v for k, v in res["leakage"].items() if k != "per_run_inversion_band"}))


if __name__ == "__main__":
    main()
