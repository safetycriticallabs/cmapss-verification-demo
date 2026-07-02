"""E0: baseline model on unit-wise partitions (framework-compliant, AI-1.1).
E1: leakage counterfactual, random row-level split (AI-1.1 violated).

Same architecture, same training budget, same seed. The measured gap between
the two test accuracies is the failure AI-1 exists to catch: inflated
evaluation caused by fault-signature leakage across partitions.
"""

import json
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"


def make_model():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
        max_iter=60, random_state=0,
    )


def evaluate(model, X, y):
    p = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
        "per_class_recall": [round(float(r), 4) for r in
                             confusion_matrix(y, p, normalize="true").diagonal()],
        "confusion_matrix": confusion_matrix(y, p).tolist(),
        "n": int(len(y)),
    }


def main():
    t0 = time.time()
    df = cmapss.load(DATA, "FD001", "train")
    sensors = cmapss.informative_sensors(df)
    results = {"dataset": "FD001", "window": cmapss.WINDOW,
               "sensors_kept": sensors, "n_sensors": len(sensors),
               "bins": {"nominal": f"RUL>{cmapss.BIN_NOMINAL}",
                        "degraded": f"{cmapss.BIN_CRITICAL}<RUL<={cmapss.BIN_NOMINAL}",
                        "critical": f"RUL<={cmapss.BIN_CRITICAL}"}}

    # ---- E0: unit-wise partitioning (AI-1.1 compliant) ----
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    results["units"] = {"train": len(tr_u), "val": len(va_u), "test": len(te_u)}
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xva, yva, _, _ = cmapss.windows(df, va_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    scaler = StandardScaler().fit(Xtr)
    m0 = make_model().fit(scaler.transform(Xtr), ytr)
    results["E0_unit_split"] = {
        "val": evaluate(m0, scaler.transform(Xva), yva),
        "test": evaluate(m0, scaler.transform(Xte), yte),
        "class_counts_train": np.bincount(ytr).tolist(),
    }

    # ---- E1: row-level split (AI-1.1 violated; leakage counterfactual) ----
    Xa, ya, _, _ = cmapss.windows(df, df["unit"].unique(), sensors)
    itr, iva, ite = cmapss.row_level_split(Xa, ya)
    scaler1 = StandardScaler().fit(Xa[itr])
    m1 = make_model().fit(scaler1.transform(Xa[itr]), ya[itr])
    results["E1_row_split_leakage"] = {
        "val": evaluate(m1, scaler1.transform(Xa[iva]), ya[iva]),
        "test": evaluate(m1, scaler1.transform(Xa[ite]), ya[ite]),
    }

    gap = (results["E1_row_split_leakage"]["test"]["accuracy"]
           - results["E0_unit_split"]["test"]["accuracy"])
    results["leakage_inflation_pp"] = round(100 * gap, 2)
    results["runtime_s"] = round(time.time() - t0, 1)

    with open(f"{OUT}/e0_e1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
