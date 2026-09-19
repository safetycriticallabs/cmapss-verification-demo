"""E7: Input robustness (AI-7.2, adversarial input protection; AI-3.4
boundary and edge cases). Experiment plan P2.

Three declared perturbation families on the baseline MLP, all in the
standardised input space the model consumes:
  FGSM: x + eps * sign(grad_x CE(f(x), y)), with the gradient computed from
        the fitted weights (ReLU forward pass, softmax output), for eps in
        {0.02, 0.05, 0.10, 0.20} (L-infinity, standardised units).
  Gaussian noise: x + N(0, eps^2) at the same magnitudes.
  Stuck-at channel: every lag of one sensor channel replaced by its training
        mean (zero in standardised units), each channel in turn.
Reported: test accuracy under each perturbation. Representative criterion:
accuracy degradation <= 5 percentage points at eps = 0.05 (FGSM).
"""

import json
import sys

import numpy as np
from scipy.special import softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cmapss

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/CMAPSSData"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
EPS = [0.02, 0.05, 0.10, 0.20]
CRITERION_PP = 5.0
NOISE_SEED = 0


def forward(m, X):
    """Manual forward pass; returns activations for the backward pass."""
    acts, pre, h = [X], [], X
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        z = h @ W + b
        pre.append(z)
        h = np.maximum(z, 0)
        acts.append(h)
    logits = h @ m.coefs_[-1] + m.intercepts_[-1]
    return acts, pre, softmax(logits, axis=1)


def input_gradient(m, X, y):
    acts, pre, p = forward(m, X)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    g = (p - onehot) / len(y)                       # dL/dlogits
    g = g @ m.coefs_[-1].T                          # dL/dh_last
    for k in range(len(pre) - 1, -1, -1):
        g = g * (pre[k] > 0)                        # through ReLU
        g = g @ m.coefs_[k].T                       # dL/dh_{k-1}
    return g, p


def main():
    df = cmapss.load(DATA, "FD001", "train")
    tr_u, va_u, te_u = cmapss.split_units(df["unit"].unique())
    sensors = cmapss.informative_sensors(df[df["unit"].isin(tr_u)])
    Xtr, ytr, _, _ = cmapss.windows(df, tr_u, sensors)
    Xte, yte, _, _ = cmapss.windows(df, te_u, sensors)
    sc = StandardScaler().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                      solver="adam", alpha=1e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=60,
                      random_state=0).fit(Str, ytr)
    # implementation check: manual forward pass must reproduce predict_proba
    _, _, p_manual = forward(m, Ste)
    check = float(np.abs(p_manual - m.predict_proba(Ste)).max())
    clean = float((m.predict(Ste) == yte).mean())
    grad, _ = input_gradient(m, Ste, yte)
    rng = np.random.default_rng(NOISE_SEED)
    res = {"requirement": "AI-7.2 adversarial input protection; AI-3.4 boundary and edge cases",
           "forward_pass_check_max_abs_diff": check, "clean_accuracy": round(clean, 4),
           "fgsm": {}, "gaussian_noise": {}, "stuck_at_channel": {}}
    for eps in EPS:
        adv = Ste + eps * np.sign(grad)
        noi = Ste + rng.normal(0, eps, Ste.shape)
        res["fgsm"][str(eps)] = round(float((m.predict(adv) == yte).mean()), 4)
        res["gaussian_noise"][str(eps)] = round(float((m.predict(noi) == yte).mean()), 4)
    n_lag = cmapss.WINDOW
    for j, c in enumerate(sensors):
        X = Ste.copy()
        X[:, j::len(sensors)] = 0.0   # window is flattened lag-major: (lag, channel)
        res["stuck_at_channel"][c] = round(float((m.predict(X) == yte).mean()), 4)
    worst = min(res["stuck_at_channel"], key=res["stuck_at_channel"].get)
    deg = 100 * (clean - res["fgsm"]["0.05"])
    res["fgsm_degradation_pp_at_0.05"] = round(deg, 2)
    res["worst_stuck_channel"] = {"channel": worst, "accuracy": res["stuck_at_channel"][worst]}
    res["criterion"] = f"FGSM accuracy degradation <= {CRITERION_PP} pp at eps = 0.05 (representative)"
    res["pass"] = bool(deg <= CRITERION_PP)
    json.dump(res, open(f"{OUT}/e7_robustness.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
