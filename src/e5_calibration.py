"""E5: Confidence calibration (AI-8.3 / AI-12.2).

Measures expected calibration error (ECE, 15 bins) of the baseline model
before and after temperature scaling. Temperature is fitted on the validation
partition only (NLL minimisation); results are reported on the test partition
against the representative acceptance criterion ECE <= 0.05. Logits are
recovered as log-probabilities, a standard recalibration family when the
classifier exposes only softmax outputs.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    detail = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        e += (m.sum() / n) * gap
        detail.append((float((lo + hi) / 2), float(conf[m].mean()),
                       float(correct[m].mean()), int(m.sum())))
    return float(e), detail


def main():
    df = cmapss.load(DATA, "FD001", "train")
    sensors = cmapss.informative_sensors(df)
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xva, yva, _, _ = cmapss.windows(df, va_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=60,
                      random_state=0).fit(sc.transform(Xtr), ytr)

    logit_va = np.log(np.clip(m.predict_proba(sc.transform(Xva)), 1e-12, 1))
    logit_te = np.log(np.clip(m.predict_proba(sc.transform(Xte)), 1e-12, 1))

    def nll(T):
        p = softmax(logit_va / T, axis=1)
        return -np.log(np.clip(p[np.arange(len(yva)), yva], 1e-12, 1)).mean()

    T = float(minimize_scalar(nll, bounds=(0.05, 10.0),
                              method="bounded").x)

    res = {"criterion": "ECE <= 0.05 (representative)", "bins": 15,
           "temperature_fitted_on_val": round(T, 3)}
    figs = {}
    for tag, temp in [("before", 1.0), ("after", T)]:
        p = softmax(logit_te / temp, axis=1)
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == yte).astype(float)
        e, detail = ece(conf, correct)
        res[f"ece_{tag}"] = round(e, 4)
        res[f"pass_{tag}"] = bool(e <= 0.05)
        figs[tag] = detail
    res["test_accuracy_unchanged"] = round(
        float((softmax(logit_te / T, axis=1).argmax(1) == yte).mean()), 4)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (tag, detail) in zip(axes, figs.items()):
        cx = [d[1] for d in detail]
        ay = [d[2] for d in detail]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.bar(cx, ay, width=1 / 15, edgecolor="k", alpha=0.7)
        ax.set_title(f"{tag} (ECE={res['ece_' + tag]:.3f})")
        ax.set_xlabel("confidence")
    axes[0].set_ylabel("empirical accuracy")
    fig.suptitle("Reliability diagrams, test partition (E5)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/e5_reliability.png", dpi=150)

    with open(f"{OUT}/e5_calibration.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
