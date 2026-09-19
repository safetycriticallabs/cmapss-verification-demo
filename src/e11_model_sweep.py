"""E11: Model-family sweep (experiment plan P5).

The same verifications executed against five model classes on the identical
FD001 task, partitions, windows and standardisation:
  M0  MLP(128, 64), the submitted model (scikit-learn)
  M1  MLP(256, 256, 128), same optimiser and budget (scikit-learn)
  M2  histogram gradient-boosted trees (scikit-learn)
  M3  one-dimensional CNN (PyTorch)
  M4  LSTM (PyTorch)
Per model: E0/E1 leakage gap (partition seed 42, model seed 0); E4 softmax
(maximum class probability) AUROC and detection rate on set A (all FD002
units) and set B fan-mode units, threshold at the 98th percentile of
validation scores; feature-space class-conditional Mahalanobis on the
penultimate layer for the neural models (PCA to 32 components when wider,
shared Ledoit-Wolf covariance); E5 ECE before and after temperature scaling
fitted on the validation partition. The input-space Mahalanobis detector is
model-independent (see E4) and not repeated here.

Run one model per invocation: `python src/e11_model_sweep.py DATA OUT M3`,
then `merge`.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(f"{OUT}/parts", exist_ok=True)
WHICH = sys.argv[3] if len(sys.argv) > 3 else "merge"
EPOCHS, BATCH, LR, SEED = 30, 256, 1e-3, 0
FPR_TARGET, TPR_CRITERION = 0.02, 0.97


# ---------------------------------------------------------------- models ---
class SkModel:
    def __init__(self, est, n_ch):
        self.est, self.n_ch = est, n_ch

    def fit(self, X, y):
        self.est.fit(X, y); return self

    def proba(self, X):
        return self.est.predict_proba(X)

    def features(self, X):
        if not isinstance(self.est, MLPClassifier):
            return None
        h = X
        for W, b in zip(self.est.coefs_[:-1], self.est.intercepts_[:-1]):
            h = np.maximum(h @ W + b, 0)
        return h


class TorchModel:
    def __init__(self, kind, n_ch, window):
        import torch, torch.nn as nn
        torch.manual_seed(SEED)
        self.torch, self.kind, self.n_ch, self.window = torch, kind, n_ch, window
        if kind == "cnn":
            self.body = nn.Sequential(
                nn.Conv1d(n_ch, 32, 5, padding=2), nn.ReLU(),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten())
        else:
            self.body = nn.LSTM(n_ch, 64, batch_first=True)
        self.head = nn.Linear(64, 3)

    def _feat(self, xb):
        if self.kind == "cnn":
            return self.body(xb.transpose(1, 2))          # (N, C, L)
        out, _ = self.body(xb)                             # (N, L, H)
        return out[:, -1, :]

    def _tensor(self, X):
        return self.torch.tensor(X.reshape(len(X), self.window, self.n_ch), dtype=self.torch.float32)

    def fit(self, X, y):
        torch, nn = self.torch, self.torch.nn
        params = list(self.body.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=LR)
        Xt, yt = self._tensor(X), torch.tensor(y, dtype=torch.long)
        g = torch.Generator().manual_seed(SEED)
        self.body.train(); self.head.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                loss = nn.functional.cross_entropy(self.head(self._feat(Xt[idx])), yt[idx])
                loss.backward(); opt.step()
        self.body.eval(); self.head.eval()
        return self

    def _batched(self, X, fn):
        outs = []
        with self.torch.no_grad():
            Xt = self._tensor(X)
            for i in range(0, len(Xt), 4096):
                outs.append(fn(Xt[i:i + 4096]).numpy())
        return np.concatenate(outs)

    def proba(self, X):
        return self._batched(X, lambda xb: self.torch.softmax(self.head(self._feat(xb)), dim=1))

    def features(self, X):
        return self._batched(X, self._feat)


def make(which, n_ch, window):
    if which == "M0":
        return SkModel(MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4,
                                     batch_size=256, learning_rate_init=1e-3, max_iter=60, random_state=SEED), n_ch)
    if which == "M1":
        return SkModel(MLPClassifier(hidden_layer_sizes=(256, 256, 128), activation="relu", solver="adam", alpha=1e-4,
                                     batch_size=256, learning_rate_init=1e-3, max_iter=60, random_state=SEED), n_ch)
    if which == "M2":
        return SkModel(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=False,
                                                      random_state=SEED), n_ch)
    if which == "M3":
        return TorchModel("cnn", n_ch, window)
    if which == "M4":
        return TorchModel("lstm", n_ch, window)
    raise ValueError(which)


# ---------------------------------------------------------- verifications ---
def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def temperature(logit_va, yva):
    nll = lambda T: -np.log(np.clip(softmax(logit_va / T, axis=1)[np.arange(len(yva)), yva], 1e-12, 1)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def feature_mahalanobis(Ftr, ytr):
    pca = PCA(n_components=min(32, Ftr.shape[1]), random_state=0).fit(Ftr)
    Z = pca.transform(Ftr)
    means = {c: Z[ytr == c].mean(axis=0) for c in np.unique(ytr)}
    prec = LedoitWolf().fit(np.vstack([Z[ytr == c] - means[c] for c in means])).precision_

    def score(F):
        Zf = pca.transform(F); d = []
        for c, mu in means.items():
            r = Zf - mu; d.append(np.einsum("ij,jk,ik->i", r, prec, r))
        return np.sqrt(np.min(np.stack(d, axis=1), axis=1))
    return score


def run(which):
    t0 = time.time()
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    n_ch, window = len(sensors), cmapss.WINDOW
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xva, yva, _, _ = cmapss.windows(df, va_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    Str, Sva, Ste = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    m = make(which, n_ch, window).fit(Str, ytr)
    pte = m.proba(Ste)
    acc = float((pte.argmax(1) == yte).mean())
    res = {"model": which, "test_accuracy": round(acc, 4)}
    # E1 leakage counterfactual
    Xa, ya, _, _ = cmapss.windows(df, df["unit"].unique(), sensors)
    itr, iva, ite = cmapss.row_level_split(Xa, ya)
    sc1 = StandardScaler().fit(Xa[itr])
    m1 = make(which, n_ch, window).fit(sc1.transform(Xa[itr]), ya[itr])
    acc1 = float((m1.proba(sc1.transform(Xa[ite])).argmax(1) == ya[ite]).mean())
    res["leakage"] = {"unit_level_acc": round(acc, 4), "row_level_acc": round(acc1, 4),
                      "gap_pp": round(100 * (acc1 - acc), 2)}
    # E4 sets
    d2 = cmapss.load(DATA, "FD002", "train")
    X2, _, _, _ = cmapss.windows(d2, d2["unit"].unique(), sensors)
    d3 = cmapss.load(DATA, "FD003", "train")
    labels = json.load(open(f"{OUT}/e4_odd_boundary.json"))["labels"]
    d3l = d3[d3["rul"] <= 100]
    d3l = d3l.assign(cycle=d3l.groupby("unit").cumcount() + 1)
    X3, _, u3, _ = cmapss.windows(d3l, d3l["unit"].unique(), sensors)
    X3f = X3[np.array([labels[str(int(u))] == "fan" for u in u3])]
    S2, S3f = sc.transform(X2), sc.transform(X3f)
    detectors = {"MSP": lambda S: -m.proba(S).max(axis=1)}
    Ftr = m.features(Str)
    if Ftr is not None:
        fm = feature_mahalanobis(Ftr, ytr)
        detectors["feature_MAH"] = lambda S: fm(m.features(S))
        res["feature_dim"] = int(Ftr.shape[1])
    res["ood"] = {}
    for name, fn in detectors.items():
        s_val, s_id = fn(Sva), fn(Ste)
        thr = np.quantile(s_val, 1 - FPR_TARGET)
        res["ood"][name] = {"fpr_measured_on_test": round(float((s_id > thr).mean()), 4)}
        for tag, S in (("A_fd002_regimes", S2), ("B_fd003_fan_mode", S3f)):
            s = fn(S)
            tpr = float((s > thr).mean())
            res["ood"][name][tag] = {"auroc": round(float(roc_auc_score(np.r_[np.zeros(len(s_id)), np.ones(len(s))], np.r_[s_id, s])), 4),
                                     "tpr": round(tpr, 4), "pass_97pct": bool(tpr >= TPR_CRITERION)}
    # E5 calibration
    lva = np.log(np.clip(m.proba(Sva), 1e-12, 1)); lte = np.log(np.clip(pte, 1e-12, 1))
    T = temperature(lva, yva)
    res["calibration"] = {"temperature": round(T, 3)}
    for tag, temp in (("before", 1.0), ("after", T)):
        p = softmax(lte / temp, axis=1)
        e = ece(p.max(1), (p.argmax(1) == yte).astype(float))
        res["calibration"][f"ece_{tag}"] = round(e, 4)
        res["calibration"][f"pass_{tag}"] = bool(e <= 0.05)
    if which in ("M3", "M4"):
        import torch
        res["software"] = {"torch": torch.__version__}
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/parts/e11_part_{which}.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


def merge():
    out = {w: json.load(open(f"{OUT}/parts/e11_part_{w}.json")) for w in ("M0", "M1", "M2", "M3", "M4")}
    json.dump(out, open(f"{OUT}/e11_model_sweep.json", "w"), indent=2)
    print(f"{'model':5s} {'acc':>6s} {'gap':>6s} {'MSP A':>7s} {'MSP Bf':>7s} {'fMAH A':>7s} {'fMAH Bf':>8s} {'ECE b':>6s} {'ECE a':>6s} {'T':>5s}")
    for w, r in out.items():
        fm = r["ood"].get("feature_MAH", {})
        print(f"{w:5s} {r['test_accuracy']:6.3f} {r['leakage']['gap_pp']:+6.2f} "
              f"{r['ood']['MSP']['A_fd002_regimes']['auroc']:7.3f} {r['ood']['MSP']['B_fd003_fan_mode']['auroc']:7.3f} "
              f"{fm.get('A_fd002_regimes', {}).get('tpr', float('nan')):7.3f} {fm.get('B_fd003_fan_mode', {}).get('tpr', float('nan')):8.3f} "
              f"{r['calibration']['ece_before']:6.3f} {r['calibration']['ece_after']:6.3f} {r['calibration']['temperature']:5.2f}")


if __name__ == "__main__":
    if WHICH == "merge":
        merge()
    else:
        run(WHICH)
