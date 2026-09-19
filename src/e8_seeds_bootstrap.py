"""E8: Seed sweep and unit-level bootstrap intervals (experiment plan P1).

Parts (run separately; each writes a partial file, `merge` combines them):
  seeds_e0e1 <pseed>  E0/E1 for one partition seed x model seeds 0..4
  seeds_e4e5          E4 softmax and E5 calibration for model seeds 0..4
  bootstrap           unit-level bootstrap intervals (B = 1000, seed 2026)
  merge               combine partial files into e8_seeds_bootstrap.json
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(f"{OUT}/parts", exist_ok=True)
PART = sys.argv[3] if len(sys.argv) > 3 else "merge"
ARG = sys.argv[4] if len(sys.argv) > 4 else None
MODEL_SEEDS = [0, 1, 2, 3, 4]
B, BOOT_SEED = 1000, 2026


def model(seed):
    return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                         solver="adam", alpha=1e-4, batch_size=256,
                         learning_rate_init=1e-3, max_iter=60, random_state=seed)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def fit_temperature(logit_va, yva):
    def nll(T):
        p = softmax(logit_va / T, axis=1)
        return -np.log(np.clip(p[np.arange(len(yva)), yva], 1e-12, 1)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def summary(v):
    v = np.asarray(v, dtype=float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "n": int(len(v))}


def base(pseed=42):
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique(), seed=pseed)
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    return df, tr_u, va_u, te_u, sensors


def part_seeds_e0e1(pseed):
    df, tr_u, va_u, te_u, sensors = base(pseed)
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    Xa, ya, _, _ = cmapss.windows(df, df["unit"].unique(), sensors)
    itr, iva, ite = cmapss.row_level_split(Xa, ya, seed=pseed)
    sc, sc1 = StandardScaler().fit(Xtr), StandardScaler().fit(Xa[itr])
    rows = []
    for ms in MODEL_SEEDS:
        a0 = float((model(ms).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte)) == yte).mean())
        a1 = float((model(ms).fit(sc1.transform(Xa[itr]), ya[itr]).predict(sc1.transform(Xa[ite])) == ya[ite]).mean())
        rows.append({"partition_seed": pseed, "model_seed": ms, "unit_level_acc": round(a0, 4),
                     "row_level_acc": round(a1, 4), "gap_pp": round(100 * (a1 - a0), 2)})
        print(rows[-1], flush=True)
    json.dump(rows, open(f"{OUT}/parts/e8_part_e0e1_{pseed}.json", "w"), indent=1)


def part_seeds_e4e5():
    df, tr_u, va_u, te_u, sensors = base(42)
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xva, yva, _, _ = cmapss.windows(df, va_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    d2 = cmapss.load(DATA, "FD002", "train")
    X2, _, _, _ = cmapss.windows(d2, d2["unit"].unique(), sensors)
    d3 = cmapss.load(DATA, "FD003", "train")
    labels = json.load(open(f"{OUT}/e4_odd_boundary.json"))["labels"]
    d3l = d3[d3["rul"] <= 100]
    d3l = d3l.assign(cycle=d3l.groupby("unit").cumcount() + 1)
    X3, _, u3, _ = cmapss.windows(d3l, d3l["unit"].unique(), sensors)
    X3f = X3[np.array([labels[str(int(u))] == "fan" for u in u3])]
    rows = []
    for ms in MODEL_SEEDS:
        m = model(ms).fit(sc.transform(Xtr), ytr)
        msp = lambda X: -m.predict_proba(sc.transform(X)).max(axis=1)
        s_val, s_id = msp(Xva), msp(Xte)
        thr = np.quantile(s_val, 0.98)
        r = {"model_seed": ms, "test_acc": round(float((m.predict(sc.transform(Xte)) == yte).mean()), 4)}
        for tag, Xo in (("A_fd002", X2), ("B_fd003_fan", X3f)):
            s = msp(Xo)
            r[f"msp_auroc_{tag}"] = round(float(roc_auc_score(np.r_[np.zeros(len(s_id)), np.ones(len(s))], np.r_[s_id, s])), 4)
            r[f"msp_tpr_{tag}"] = round(float((s > thr).mean()), 4)
        lva = np.log(np.clip(m.predict_proba(sc.transform(Xva)), 1e-12, 1))
        lte = np.log(np.clip(m.predict_proba(sc.transform(Xte)), 1e-12, 1))
        T = fit_temperature(lva, yva)
        for tag, temp in (("before", 1.0), ("after", T)):
            p = softmax(lte / temp, axis=1)
            r[f"ece_{tag}"] = round(ece(p.max(1), (p.argmax(1) == yte).astype(float)), 4)
        r["temperature"] = round(T, 3)
        rows.append(r); print(r, flush=True)
    json.dump(rows, open(f"{OUT}/parts/e8_part_e4e5.json", "w"), indent=1)


def part_bootstrap():
    rng = np.random.default_rng(BOOT_SEED)
    df, tr_u, va_u, te_u, sensors = base(42)
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xva, yva, _, _ = cmapss.windows(df, va_u, sensors)
    Xte, yte, ute, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    Str = sc.transform(Xtr)
    m = model(0).fit(Str, ytr)
    pred = m.predict(sc.transform(Xte))
    units = np.unique(ute)
    idx_by_unit = {u: np.nonzero(ute == u)[0] for u in units}

    def unit_resample(units_arr, idx_map):
        pick = rng.choice(units_arr, size=len(units_arr), replace=True)
        return np.concatenate([idx_map[u] for u in pick])

    out = {"B": B, "seed": BOOT_SEED, "method": "unit-level resampling with replacement; 95% percentile intervals"}
    acc, f1, rec = [], [], []
    for _ in range(B):
        i = unit_resample(units, idx_by_unit)
        acc.append(accuracy_score(yte[i], pred[i]))
        f1.append(f1_score(yte[i], pred[i], average="macro"))
        crit = yte[i] == 2
        rec.append(float((pred[i][crit] == 2).mean()) if crit.any() else np.nan)
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    out["unit_level_test"] = {"accuracy": round(float((pred == yte).mean()), 4), "accuracy_ci": ci(acc),
                              "macro_f1_ci": ci(f1), "critical_recall_ci": ci(rec)}
    # row-level (leaked) accuracy: window-level bootstrap
    Xa, ya, _, _ = cmapss.windows(df, df["unit"].unique(), sensors)
    itr, iva, ite = cmapss.row_level_split(Xa, ya)
    sc1 = StandardScaler().fit(Xa[itr])
    p1 = model(0).fit(sc1.transform(Xa[itr]), ya[itr]).predict(sc1.transform(Xa[ite]))
    corr1 = (p1 == ya[ite]).astype(float)
    acc1 = [corr1[rng.integers(0, len(corr1), len(corr1))].mean() for _ in range(B)]
    out["row_level_test"] = {"accuracy": round(float(corr1.mean()), 4), "accuracy_ci": ci(acc1)}
    # OOD: fixed thresholds from validation (as in E4), resample ID and OOD units
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
    X2, _, u2, _ = cmapss.windows(d2, d2["unit"].unique(), sensors)
    d3 = cmapss.load(DATA, "FD003", "train")
    labels = json.load(open(f"{OUT}/e4_odd_boundary.json"))["labels"]
    d3l = d3[d3["rul"] <= 100]
    d3l = d3l.assign(cycle=d3l.groupby("unit").cumcount() + 1)
    X3, _, u3, _ = cmapss.windows(d3l, d3l["unit"].unique(), sensors)
    fan = np.array([labels[str(int(u))] == "fan" for u in u3])
    sets = {"A_fd002_regimes": (X2, u2), "B_fd003_fan_mode": (X3[fan], u3[fan]),
            "B_fd003_hpc_mode": (X3[~fan], u3[~fan])}
    out["ood"] = {}
    for name, fn in (("MSP", msp), ("MAH", mah)):
        s_val, s_id = fn(Xva), fn(Xte)
        thr = np.quantile(s_val, 0.98)
        for tag, (Xo, uo) in sets.items():
            s = fn(Xo)
            ou = np.unique(uo); omap = {u: np.nonzero(uo == u)[0] for u in ou}
            tprs, aurocs = [], []
            for _ in range(B):
                i = unit_resample(units, idx_by_unit); j = unit_resample(ou, omap)
                tprs.append(float((s[j] > thr).mean()))
                aurocs.append(roc_auc_score(np.r_[np.zeros(len(i)), np.ones(len(j))], np.r_[s_id[i], s[j]]))
            out["ood"][f"{name}_{tag}"] = {"tpr": round(float((s > thr).mean()), 4), "tpr_ci": ci(tprs),
                                           "auroc": round(float(roc_auc_score(np.r_[np.zeros(len(s_id)), np.ones(len(s))], np.r_[s_id, s])), 4),
                                           "auroc_ci": ci(aurocs)}
            print(name, tag, out["ood"][f"{name}_{tag}"], flush=True)
    # ECE before/after, unit-level resampling of the test partition
    lva = np.log(np.clip(m.predict_proba(sc.transform(Xva)), 1e-12, 1))
    lte = np.log(np.clip(m.predict_proba(sc.transform(Xte)), 1e-12, 1))
    T = fit_temperature(lva, yva)
    out["calibration"] = {"temperature": round(T, 3)}
    for tag, temp in (("before", 1.0), ("after", T)):
        p = softmax(lte / temp, axis=1); conf = p.max(1); corr = (p.argmax(1) == yte).astype(float)
        vals = [ece(conf[i], corr[i]) for i in (unit_resample(units, idx_by_unit) for _ in range(B))]
        out["calibration"][f"ece_{tag}"] = round(ece(conf, corr), 4)
        out["calibration"][f"ece_{tag}_ci"] = ci(vals)
    json.dump(out, open(f"{OUT}/parts/e8_part_bootstrap.json", "w"), indent=1)
    print(json.dumps(out["unit_level_test"]), json.dumps(out["row_level_test"]), json.dumps(out["calibration"]))


def part_merge():
    rows = []
    for f in sorted(glob.glob(f"{OUT}/parts/e8_part_e0e1_*.json")):
        rows += json.load(open(f))
    e45 = json.load(open(f"{OUT}/parts/e8_part_e4e5.json"))
    boot = json.load(open(f"{OUT}/parts/e8_part_bootstrap.json"))
    res = {"seed_sweep_e0e1": {"runs": rows,
                               "gap_pp": summary([r["gap_pp"] for r in rows]),
                               "unit_level_acc": summary([r["unit_level_acc"] for r in rows]),
                               "row_level_acc": summary([r["row_level_acc"] for r in rows]),
                               "gap_always_positive": bool(min(r["gap_pp"] for r in rows) > 0),
                               "inverted_band": [round(min(r["unit_level_acc"] for r in rows), 4),
                                                 round(max(r["row_level_acc"] for r in rows), 4)]},
           "seed_sweep_e4e5": {"runs": e45,
                               **{k: summary([r[k] for r in e45]) for k in e45[0] if k != "model_seed"}},
           "bootstrap": boot}
    json.dump(res, open(f"{OUT}/e8_seeds_bootstrap.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res["seed_sweep_e0e1"].items() if k != "runs"}, indent=1))
    print(json.dumps({k: v for k, v in res["seed_sweep_e4e5"].items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    if PART == "seeds_e0e1":
        part_seeds_e0e1(int(ARG))
    elif PART == "seeds_e4e5":
        part_seeds_e4e5()
    elif PART == "bootstrap":
        part_bootstrap()
    else:
        part_merge()
