"""E4: Out-of-distribution detection (AI-6.1/6.2/6.3; AI-12.5), with the
ODD boundary validation step added for the revision (experiment plan P0).

The declared ODD is FD001: single operating condition, HPC-degradation fault
mode. Out-of-ODD sets:

  OOD-A: all FD002 training units (six unseen operating regimes, same fault
         mode).
  OOD-B: FD003 late-life windows (RUL <= 100). FD003 contains two fault
         modes, HPC degradation (inside the declared ODD by physics) and fan
         degradation (outside it), and the dataset does not label which unit
         has which. Set B is therefore decomposed by a fault-mode signature
         clustering, fixed before any detector is evaluated, into fan-mode
         units (the true out-of-ODD set) and HPC-mode units (an independent
         in-ODD fleet). Results are reported for both groups and pooled.
         Early-life FD003 windows are excluded because a healthy engine is
         genuinely in-distribution.

Detectors, per AI-12.5's assertion that softmax confidence alone is
insufficient for neural networks:

  MSP: maximum softmax probability (baseline).
  MAH: minimum class-conditional Mahalanobis distance in a 32-component PCA
       space fitted on training windows (shared covariance, Ledoit-Wolf).

Operating point per the representative criterion: the detection threshold is
the 98th percentile of in-distribution scores on the VALIDATION partition
(design false-positive rate 2%); the false-positive rate measured on the test
partition is reported beside it. Detection rate (TPR) and AUROC are reported
per set. AI-6.1 evidence: the training distribution characterisation
(PCA + class Gaussians) and the ODD boundary validation are both artifacts.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
FPR_TARGET = 0.02
TPR_CRITERION = 0.97
SIG_CYCLES = 20


def unit_signatures(df, sensors, n=SIG_CYCLES):
    """Per-unit degradation signature: (mean of the last n cycles minus mean
    of the first n cycles) / std of the first n cycles, per channel."""
    rows, units = [], sorted(df["unit"].unique())
    for u in units:
        g = df[df["unit"] == u].sort_values("cycle")
        first, last = g[sensors].iloc[:n], g[sensors].iloc[-n:]
        rows.append(((last.mean() - first.mean()) / (first.std() + 1e-9)).to_numpy())
    return np.asarray(rows), units


def fault_mode_labels(d1, d3, sensors):
    """ODD boundary validation: label each FD003 unit HPC-mode or fan-mode by
    clustering degradation signatures (k = 2). The cluster nearer to the
    FD001 (HPC-only) signatures is HPC-mode. Fixed before detection."""
    S1, _ = unit_signatures(d1, sensors)
    S3, u3 = unit_signatures(d3, sensors)
    sc = StandardScaler().fit(np.vstack([S1, S3]))
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(sc.transform(S3))
    d = [np.linalg.norm(sc.transform(S1) - km.cluster_centers_[k], axis=1).mean()
         for k in range(2)]
    hpc_cluster = int(np.argmin(d))
    labels = {int(u): ("HPC" if l == hpc_cluster else "fan")
              for u, l in zip(u3, km.labels_)}
    diff = S3[km.labels_ != hpc_cluster].mean(0) - S3[km.labels_ == hpc_cluster].mean(0)
    order = np.argsort(-np.abs(diff))
    art = {"method": "KMeans k=2, n_init=20, random_state=0 on standardised "
                     "per-unit signatures (last-20 minus first-20 cycle mean, "
                     "over first-20 std), FD001 and FD003 standardised together",
           "fd003_units": len(u3),
           "hpc_mode_units": int(sum(v == "HPC" for v in labels.values())),
           "fan_mode_units": int(sum(v == "fan" for v in labels.values())),
           "mean_distance_fd001_to_cluster_centres": [round(float(x), 3) for x in d],
           "largest_signature_differences_fan_minus_hpc":
               [(sensors[i], round(float(diff[i]), 2)) for i in order[:6]],
           "labels": labels}
    # figure: signatures in the first two principal components
    pca = PCA(n_components=2, random_state=0).fit(sc.transform(np.vstack([S1, S3])))
    Z1, Z3 = pca.transform(sc.transform(S1)), pca.transform(sc.transform(S3))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(Z1[:, 0], Z1[:, 1], s=18, marker="x", color="gray", label="FD001 units (HPC only)")
    m = km.labels_ == hpc_cluster
    ax.scatter(Z3[m, 0], Z3[m, 1], s=22, color="C0", label="FD003 units, HPC-mode cluster")
    ax.scatter(Z3[~m, 0], Z3[~m, 1], s=22, color="C3", label="FD003 units, fan-mode cluster")
    ax.set_xlabel("signature PC1"); ax.set_ylabel("signature PC2")
    ax.set_title("E4: ODD boundary validation, FD003 fault-mode signatures")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e4_fd003_signatures.png", dpi=150)
    return labels, art


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

    # --- AI-6.1: training distribution characterisation ---
    pca = PCA(n_components=32, random_state=0).fit(Str)
    Ztr = pca.transform(Str)
    means = {c: Ztr[ytr == c].mean(axis=0) for c in np.unique(ytr)}
    lw = LedoitWolf().fit(
        np.vstack([Ztr[ytr == c] - means[c] for c in means]))
    prec = lw.precision_

    def mahalanobis(X):
        Z = pca.transform(sc.transform(X))
        d = []
        for c, mu in means.items():
            r = Z - mu
            d.append(np.einsum("ij,jk,ik->i", r, prec, r))
        return np.sqrt(np.min(np.stack(d, axis=1), axis=1))

    def msp(X):
        return m.predict_proba(sc.transform(X)).max(axis=1)

    # --- OOD sets ---
    d2 = cmapss.load(DATA, "FD002", "train")
    X2, _, u2, _ = cmapss.windows(d2, d2["unit"].unique(), sensors)
    d3 = cmapss.load(DATA, "FD003", "train")
    labels, boundary = fault_mode_labels(df, d3, sensors)
    d3l = d3[d3["rul"] <= 100]
    d3l = d3l.assign(cycle=d3l.groupby("unit").cumcount() + 1)
    X3, _, u3, _ = cmapss.windows(d3l, d3l["unit"].unique(), sensors)
    is_fan = np.array([labels[int(u)] == "fan" for u in u3])
    sets = {"A_fd002_regimes": (X2, u2),
            "B_fd003_fan_mode": (X3[is_fan], u3[is_fan]),
            "B_fd003_hpc_mode": (X3[~is_fan], u3[~is_fan]),
            "B_fd003_pooled": (X3, u3)}

    res = {"declared_odd": "FD001: single operating condition, HPC fault mode",
           "characterisation": {"pca_components": 32,
                                "covariance": "shared, Ledoit-Wolf",
                                "class_gaussians": len(means)},
           "criterion": f"detection >= {TPR_CRITERION:.0%} at FPR <= {FPR_TARGET:.0%} (representative)",
           "threshold_rule": "98th percentile of in-distribution scores on the validation partition",
           "n_id_val": int(len(Xva)), "n_id_test": int(len(Xte)),
           "n_sets": {k: {"windows": int(len(v[0])), "units": int(len(np.unique(v[1])))}
                      for k, v in sets.items()},
           "odd_boundary_validation": {k: v for k, v in boundary.items() if k != "labels"}}

    scores = {"MSP": (lambda X: -msp(X)), "MAH": mahalanobis}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, fn) in zip(axes, scores.items()):
        s_val, s_id = fn(Xva), fn(Xte)
        thr = float(np.quantile(s_val, 1 - FPR_TARGET))
        fpr_test = float((s_id > thr).mean())
        res[f"{name}_threshold"] = round(thr, 6)
        res[f"{name}_fpr_measured_on_test"] = round(fpr_test, 4)
        oods = {}
        for tag, (Xo, _) in sets.items():
            s_ood = fn(Xo)
            oods[tag] = s_ood
            auroc = roc_auc_score(
                np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))],
                np.r_[s_id, s_ood])
            tpr = float((s_ood > thr).mean())
            res[f"{name}_{tag}"] = {"auroc": round(float(auroc), 4),
                                    "tpr_at_threshold": round(tpr, 4),
                                    "pass_97pct": bool(tpr >= TPR_CRITERION)}
        allv = np.concatenate([s_id] + [oods[k] for k in oods if k != "B_fd003_pooled"])
        if name == "MAH":
            bins = np.geomspace(max(allv.min(), 1e-1), allv.max(), 70)
            ax.set_xscale("log")
        else:
            bins = np.linspace(allv.min(), -0.32, 70)
        ax.hist(s_id, bins=bins, alpha=0.5, density=True, label="ID (FD001 test)", color="gray")
        for tag, col in (("A_fd002_regimes", "C0"), ("B_fd003_fan_mode", "C3"),
                         ("B_fd003_hpc_mode", "C2")):
            ax.hist(oods[tag], bins=bins, alpha=0.45, density=True, label=tag, color=col)
        ax.axvline(thr, color="k", ls="--", lw=1, label="2% FPR threshold (validation)")
        ax.set_title(name)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.suptitle("E4: OOD score distributions")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e4_ood_scores.png", dpi=150)

    with open(f"{OUT}/e4_ood.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(f"{OUT}/e4_odd_boundary.json", "w") as f:
        json.dump(boundary, f, indent=2)
    print(json.dumps({k: v for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
