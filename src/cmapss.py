"""C-MAPSS data utilities for the applied verification demonstration.

Data: NASA C-MAPSS turbofan degradation simulation (Saxena et al., PHM 2008).
Task framing: health-state classification (nominal / degraded / critical) from
sliding windows of sensor channels, mirroring an FDD-style component.

The model and pipeline are deliberately ordinary; the object of study is the
verification evidence the framework produces, not model performance.
"""

import numpy as np
import pandas as pd

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle", "op1", "op2", "op3"] + SENSOR_COLS

# Health-state label bins on remaining useful life (RUL), in cycles.
# Rationale: RUL > 100 treated as nominal (consistent with the piecewise-linear
# RUL cap widely used on C-MAPSS); RUL <= 30 treated as critical (representative
# maintenance-alert horizon). Documented as project-defined per Section 1.5 of
# the framework.
BIN_NOMINAL = 100
BIN_CRITICAL = 30

WINDOW = 30  # cycles per input window
SEED = 42


def load(path, subset="FD001", split="train"):
    df = pd.read_csv(
        f"{path}/{split}_{subset}.txt", sep=r"\s+", header=None, names=COLS
    )
    rul = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["rul"] = rul
    df["label"] = np.where(
        rul > BIN_NOMINAL, 0, np.where(rul > BIN_CRITICAL, 1, 2)
    )
    return df


def informative_sensors(df, threshold=1e-5):
    """Drop near-constant channels (documented, data-driven selection)."""
    keep = []
    for c in SENSOR_COLS:
        s = df[c].std()
        m = abs(df[c].mean())
        if s > threshold and (m == 0 or s / max(m, 1e-12) > 1e-6):
            if df[c].nunique() > 2:
                keep.append(c)
    return keep


def split_units(units, seed=SEED, fracs=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    units = np.array(sorted(units))
    rng.shuffle(units)
    n = len(units)
    a = int(round(fracs[0] * n))
    b = a + int(round(fracs[1] * n))
    return units[:a], units[a:b], units[b:]


def windows(df, units, sensors, window=WINDOW):
    """Sliding windows per unit; label = health state at window end."""
    X, y, uid, cyc = [], [], [], []
    for u in units:
        g = df[df["unit"] == u].sort_values("cycle")
        arr = g[sensors].to_numpy()
        labels = g["label"].to_numpy()
        cycles = g["cycle"].to_numpy()
        for i in range(window - 1, len(g)):
            X.append(arr[i - window + 1 : i + 1].ravel())
            y.append(labels[i])
            uid.append(u)
            cyc.append(cycles[i])
    return (np.asarray(X, dtype=np.float32), np.asarray(y),
            np.asarray(uid), np.asarray(cyc))


def row_level_split(X, y, seed=SEED, fracs=(0.6, 0.2, 0.2)):
    """The leakage counterfactual: split windows at random, ignoring units."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    a = int(round(fracs[0] * len(X)))
    b = a + int(round(fracs[1] * len(X)))
    return idx[:a], idx[a:b], idx[b:]
