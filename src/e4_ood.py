"""E4: Out-of-distribution detection (AI-6.1/6.2/6.3; AI-12.5).

The declared ODD is FD001: single operating condition, HPC-degradation fault
mode. Two real out-of-ODD sets are evaluated:

  OOD-A: FD002 windows (six unseen operating regimes; same fault mode).
  OOD-B: FD003 late-life windows, RUL <= 100 (unseen fan-degradation fault
         mode present in the subset; early-life FD003 windows are excluded
         because a healthy engine is genuinely in-distribution).

Detectors compared, per AI-12.5's assertion that softmax confidence alone is
insufficient for neural networks:

  MSP: maximum softmax probability (baseline).
  MAH: minimum class-conditional Mahalanobis distance in a 32-component PCA
       space fitted on training windows (shared covariance, Ledoit-Wolf).

Operating point per the representative criterion: threshold set on the
in-distribution test partition at 2% false-positive rate; detection rate
(TPR) reported per OOD set, alongside AUROC. AI-6.1 evidence: the training
distribution characterisation (PCA + class Gaussians) is itself an artifact.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def main():
    df = cmapss.load(DATA, "FD001", "train")
    sensors = cmapss.informative_sensors(df)
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)
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

    # --- OOD sets (same sensor channels; FD002/FD003 share the schema) ---
    d2 = cmapss.load(DATA, "FD002", "train")
    X2, _, _, _ = cmapss.windows(d2, d2["unit"].unique()[:60], sensors)
    d3 = cmapss.load(DATA, "FD003", "train")
    d3l = d3[d3["rul"] <= 100]
    # rebuild per-unit windows on late-life rows only
    X3, _, _, _ = cmapss.windows(
        d3l.assign(cycle=d3l.groupby("unit").cumcount() + 1),
        d3l["unit"].unique(), sensors)

    res = {"declared_odd": "FD001: single operating condition, HPC fault mode",
           "characterisation": {"pca_components": 32,
                                "covariance": "shared, Ledoit-Wolf",
                                "class_gaussians": len(means)},
           "criterion": f"detection >= 97% at FPR <= {FPR_TARGET:.0%} (representative)",
           "n_id_test": int(len(Xte)), "n_ood_A_fd002": int(len(X2)),
           "n_ood_B_fd003_late": int(len(X3))}

    scores = {"MSP": (lambda X: -msp(X)), "MAH": mahalanobis}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, fn) in zip(axes, scores.items()):
        s_id = fn(Xte)
        thr = np.quantile(s_id, 1 - FPR_TARGET)
        oods = {}
        for tag, Xo in [("A_fd002_regimes", X2), ("B_fd003_fanfault", X3)]:
            s_ood = fn(Xo)
            oods[tag] = s_ood
            auroc = roc_auc_score(
                np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))],
                np.r_[s_id, s_ood])
            tpr = float((s_ood > thr).mean())
            res[f"{name}_{tag}"] = {"auroc": round(float(auroc), 4),
                                    "tpr_at_2pct_fpr": round(tpr, 4),
                                    "pass_97pct": bool(tpr >= 0.97)}
        allv = np.concatenate([s_id] + list(oods.values()))
        if name == "MAH":  # heavy right tail: log-spaced bins, log axis
            bins = np.geomspace(max(allv.min(), 1e-1), allv.max(), 70)
            ax.set_xscale("log")
        else:
            bins = np.linspace(allv.min(), -0.32, 70)
        for tag, s_ood in oods.items():
            ax.hist(s_ood, bins=bins, alpha=0.5, density=True, label=tag)
        ax.hist(s_id, bins=bins, alpha=0.5, density=True, label="ID (FD001 test)")
        ax.axvline(thr, color="k", ls="--", lw=1, label="2% FPR threshold")
        ax.set_title(name)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.suptitle("E4: OOD score distributions")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e4_ood_scores.png", dpi=150)

    with open(f"{OUT}/e4_ood.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
