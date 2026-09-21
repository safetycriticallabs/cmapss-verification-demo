"""E12: Evidence sufficiency for the two determinations whose bootstrap
interval contains the criterion (fan-mode detection, ECE after scaling).

Asks what fleet size would be needed for the interval, at the observed
point estimate and between-engine variance, to exclude the criterion.
  Detection: per-engine detection rates over the fan-mode units; the 95%
      half-width of the pooled rate is approximated as 1.96 * sd / sqrt(n)
      for n out-of-envelope engines (an approximation; the E8 bootstrap
      also resamples the in-envelope units that set the threshold).
  Calibration: the test-partition ECE after scaling, with k engines
      resampled with replacement from the 20 test engines (B = 600, seed
      2026), for k in {20, 40, 80, 160}: a resampling extrapolation that
      keeps the observed per-engine behaviour and only changes the count.
Both are approximations of how the intervals scale, not predictions of what
new engines would show.
"""

import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
TPR_CRITERION, ECE_CRITERION = 0.97, 0.05
B, SEED = 600, 2026


def ece(conf, corr, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.sum():
            e += mk.sum() / len(conf) * abs(corr[mk].mean() - conf[mk].mean())
    return float(e)


def main():
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xva, yva, _, _ = cmapss.windows(df, va_u, sensors)
    Xte, yte, ute, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr); Str = sc.transform(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                      batch_size=256, learning_rate_init=1e-3, max_iter=60, random_state=0).fit(Str, ytr)
    pca = PCA(n_components=32, random_state=0).fit(Str); Ztr = pca.transform(Str)
    means = {c: Ztr[ytr == c].mean(0) for c in np.unique(ytr)}
    prec = LedoitWolf().fit(np.vstack([Ztr[ytr == c] - means[c] for c in means])).precision_

    def mah(X):
        Z = pca.transform(sc.transform(X)); d = []
        for c, mu in means.items():
            r = Z - mu; d.append(np.einsum("ij,jk,ik->i", r, prec, r))
        return np.sqrt(np.min(np.stack(d, 1), 1))
    thr = float(np.quantile(mah(Xva), 0.98))
    labels = json.load(open(f"{OUT}/e4_odd_boundary.json"))["labels"]
    d3 = cmapss.load(DATA, "FD003", "train")
    d3l = d3[d3["rul"] <= 100]; d3l = d3l.assign(cycle=d3l.groupby("unit").cumcount() + 1)
    X3, _, u3, _ = cmapss.windows(d3l, d3l["unit"].unique(), sensors)
    fan = [u for u in np.unique(u3) if labels[str(int(u))] == "fan"]
    s = mah(X3)
    per = np.array([(s[u3 == u] > thr).mean() for u in fan])
    pooled = float((s[np.isin(u3, fan)] > thr).mean()); sd = float(per.std(ddof=1))
    det = {"fan_mode_engines": len(fan), "pooled_tpr": round(pooled, 4), "per_engine_tpr_sd": round(sd, 4),
           "per_engine_tpr_min": round(float(per.min()), 3), "criterion": TPR_CRITERION, "scaling": {}}
    n = len(fan)
    while n <= 2048:
        hw = 1.96 * sd / np.sqrt(n)
        det["scaling"][str(n)] = {"half_width": round(float(hw), 4), "upper": round(pooled + hw, 4),
                                  "excludes_criterion": bool(pooled + hw < TPR_CRITERION)}
        n *= 2
    det["engines_for_exclusion_at_observed_estimate"] = next(
        (int(k) for k, v in det["scaling"].items() if v["excludes_criterion"]), None)

    lva = np.log(np.clip(m.predict_proba(sc.transform(Xva)), 1e-12, 1))
    lte = np.log(np.clip(m.predict_proba(sc.transform(Xte)), 1e-12, 1))
    nll = lambda T: -np.log(np.clip(softmax(lva / T, 1)[np.arange(len(yva)), yva], 1e-12, 1)).mean()
    T = float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)
    p = softmax(lte / T, 1); conf = p.max(1); corr = (p.argmax(1) == yte).astype(float)
    rng = np.random.default_rng(SEED); units = np.unique(ute); idx = {u: np.nonzero(ute == u)[0] for u in units}
    cal = {"test_engines": int(len(units)), "ece_after": round(ece(conf, corr), 4), "criterion": ECE_CRITERION,
           "resampling_extrapolation": {}}
    for k in (20, 40, 80, 160):
        vals = []
        for _ in range(B):
            pick = rng.choice(units, size=k, replace=True)
            i = np.concatenate([idx[u] for u in pick]); vals.append(ece(conf[i], corr[i]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        cal["resampling_extrapolation"][str(k)] = {"interval": [round(float(lo), 4), round(float(hi), 4)],
                                                   "excludes_criterion": bool(hi < ECE_CRITERION)}
    cal["engines_for_exclusion"] = next((int(k) for k, v in cal["resampling_extrapolation"].items() if v["excludes_criterion"]), None)
    res = {"detection_fan_mode": det, "calibration_after_scaling": cal,
           "note": "approximations of interval scaling at the observed point estimates; not predictions for new engines"}
    json.dump(res, open(f"{OUT}/e12_evidence_sufficiency.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
