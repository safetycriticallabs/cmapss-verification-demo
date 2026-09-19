"""E2: Test coverage of the declared operating envelope (AI-3.3, operational
input distribution testing). Experiment plan P2.

Two declared coverage measures of the test partition against the training
partition, both computed in the input space of the model:
  grid coverage: a 20 x 20 grid over the first two principal components
      (PCA fitted on training windows; axes span the 1st to 99th percentile of
      the training scores). Coverage = cells occupied by at least one test
      window / cells occupied by at least one training window. Also the
      fraction of training mass (windows) that lies in cells the test set
      reaches.
  envelope coverage: fraction of test windows whose class-conditional
      Mahalanobis score (E4 characterisation) lies below the 99th percentile
      of training scores.
Representative criterion: grid coverage >= 0.90.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
GRID = 20
CRITERION = 0.90


def main():
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)

    pca2 = PCA(n_components=2, random_state=0).fit(Str)
    Ztr, Zte = pca2.transform(Str), pca2.transform(Ste)
    lo, hi = np.percentile(Ztr, 1, axis=0), np.percentile(Ztr, 99, axis=0)
    edges = [np.linspace(lo[k], hi[k], GRID + 1) for k in range(2)]
    Htr, _, _ = np.histogram2d(Ztr[:, 0], Ztr[:, 1], bins=edges)
    Hte, _, _ = np.histogram2d(Zte[:, 0], Zte[:, 1], bins=edges)
    occ_tr, occ_te = Htr > 0, Hte > 0
    grid_cov = float((occ_tr & occ_te).sum() / occ_tr.sum())
    mass_cov = float(Htr[occ_te].sum() / Htr.sum())
    per_class = {}
    for c, name in enumerate(("nominal", "degraded", "critical")):
        Hc_tr, _, _ = np.histogram2d(Ztr[ytr == c, 0], Ztr[ytr == c, 1], bins=edges)
        Hc_te, _, _ = np.histogram2d(Zte[yte == c, 0], Zte[yte == c, 1], bins=edges)
        per_class[name] = round(float(((Hc_tr > 0) & (Hc_te > 0)).sum() / max((Hc_tr > 0).sum(), 1)), 4)

    pca = PCA(n_components=32, random_state=0).fit(Str)
    Z32tr = pca.transform(Str)
    means = {c: Z32tr[ytr == c].mean(axis=0) for c in np.unique(ytr)}
    prec = LedoitWolf().fit(np.vstack([Z32tr[ytr == c] - means[c] for c in means])).precision_

    def mah(S):
        Z = pca.transform(S); d = []
        for c, mu in means.items():
            r = Z - mu; d.append(np.einsum("ij,jk,ik->i", r, prec, r))
        return np.sqrt(np.min(np.stack(d, axis=1), axis=1))
    s_tr, s_te = mah(Str), mah(Ste)
    env99 = float(np.percentile(s_tr, 99))
    env_cov = float((s_te <= env99).mean())

    res = {"requirement": "AI-3.3 operational input distribution testing",
           "grid": {"size": GRID, "axes": "PC1, PC2 of standardised training windows",
                    "range": "1st to 99th percentile of training scores",
                    "training_cells_occupied": int(occ_tr.sum()),
                    "test_cells_occupied_within_training": int((occ_tr & occ_te).sum()),
                    "test_cells_outside_training_cells": int((occ_te & ~occ_tr).sum()),
                    "grid_coverage": round(grid_cov, 4),
                    "training_mass_covered": round(mass_cov, 4),
                    "per_class_grid_coverage": per_class},
           "envelope": {"training_99th_percentile_mahalanobis": round(env99, 4),
                        "test_fraction_within_envelope": round(env_cov, 4)},
           "criterion": f"grid coverage >= {CRITERION} (representative)",
           "pass": bool(grid_cov >= CRITERION)}

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].imshow(np.log1p(Htr.T), origin="lower", cmap="Greys",
                 extent=[lo[0], hi[0], lo[1], hi[1]], aspect="auto")
    ax[0].set_title("training windows (log count)")
    cov_map = np.where(occ_tr, np.where(occ_te, 1.0, 0.0), np.nan)
    ax[1].imshow(cov_map.T, origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
                 extent=[lo[0], hi[0], lo[1], hi[1]], aspect="auto")
    ax[1].set_title(f"training cells reached by test (coverage {grid_cov:.2f})")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.suptitle("E2: coverage of the declared envelope by the test partition")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e2_coverage.png", dpi=150)
    json.dump(res, open(f"{OUT}/e2_coverage.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
