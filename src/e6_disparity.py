"""E6: Cross-regime performance disparity (AI-2.1, bias-bounded baseline
performance). Experiment plan P2.

A model of the baseline architecture is trained on FD002, whose engines run
under six operating regimes that change from cycle to cycle. Inputs are the
screened sensor channels plus the three operating settings, so the regime is
observable to the model. Regimes are identified by KMeans (k = 6) on the
operating settings. Each window is assigned the regime of its final cycle.
Reported: per-regime test accuracy and critical-class recall; disparity =
max minus min across regimes. Representative criterion: accuracy disparity
<= 5 percentage points.
"""

import json
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
CRITERION_PP = 5.0
OPS = ["op1", "op2", "op3"]


def main():
    df = cmapss.load(DATA, "FD002", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    feats = sensors + OPS
    km = KMeans(n_clusters=6, n_init=20, random_state=0).fit(
        df.loc[df["unit"].isin(tr_u), OPS].to_numpy())
    df = df.assign(regime=km.predict(df[OPS].to_numpy()))
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, feats)
    Xte, yte, ute, cte = cmapss.windows(df, te_u, feats)
    # regime of each test window = regime of its final cycle
    key = df.set_index(["unit", "cycle"])["regime"]
    rte = np.array([key.loc[(u, c)] for u, c in zip(ute, cte)])
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=60,
                      random_state=0).fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    per = {}
    for r in range(6):
        mk = rte == r
        crit = mk & (yte == 2)
        per[f"regime_{r}"] = {
            "n_windows": int(mk.sum()),
            "op_settings_centre": [round(float(v), 3) for v in km.cluster_centers_[r]],
            "accuracy": round(float(accuracy_score(yte[mk], pred[mk])), 4),
            "critical_recall": round(float((pred[crit] == 2).mean()), 4) if crit.any() else None}
    accs = [v["accuracy"] for v in per.values()]
    recs = [v["critical_recall"] for v in per.values() if v["critical_recall"] is not None]
    res = {"requirement": "AI-2.1 bias-bounded baseline performance (cross-regime disparity)",
           "dataset": "FD002 (six operating regimes), unit-level 60/20/20 split seed 42, model seed 0",
           "inputs": feats, "n_train_windows": int(len(Xtr)), "n_test_windows": int(len(Xte)),
           "overall_test_accuracy": round(float(accuracy_score(yte, pred)), 4),
           "per_regime": per,
           "accuracy_disparity_pp": round(100 * (max(accs) - min(accs)), 2),
           "critical_recall_disparity_pp": round(100 * (max(recs) - min(recs)), 2),
           "criterion": f"accuracy disparity <= {CRITERION_PP} pp (representative)",
           "pass": bool(100 * (max(accs) - min(accs)) <= CRITERION_PP)}
    json.dump(res, open(f"{OUT}/e6_disparity.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
