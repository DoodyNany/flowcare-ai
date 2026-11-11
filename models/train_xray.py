"""
Flow Care - Chest X-ray pneumonia WARNING model (never a diagnosis).

A small convolutional neural network (CNN) written and trained from scratch in JAX.
Data: PneumoniaMNIST (MedMNIST v2, CC BY 4.0) - 5,856 real, de-identified pediatric
chest X-rays (Kermany et al., 2018), resized to 128x128 (64x64 also supported).
Official split: train 4,708 / validation 524 / test 624 (test never used for training
or tuning).

Safety design: the threshold is tuned on the VALIDATION set for high sensitivity
(catch >= 95% of pneumonia cases), because the AI only raises a flag for a physician
to review - a missed case is worse than an extra review.
The network ends in global-average-pooling, so it can show a heat-map of WHERE it
looked (class activation map) to help the physician check it.
"""
import json, time
import numpy as np
import jax, jax.numpy as jnp
import optax
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
import sys
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 128
d = np.load(ROOT / "xray_data" / f"pneumoniamnist_{SIZE}.npz")
prep = lambda a: (a.astype(np.float32) / 255.0)[..., None]
Xtr, ytr = prep(d["train_images"]), d["train_labels"][:, 0].astype(np.float32)
Xva, yva = prep(d["val_images"]), d["val_labels"][:, 0].astype(np.float32)
Xte, yte = prep(d["test_images"]), d["test_labels"][:, 0].astype(np.float32)
MEAN, STD = float(Xtr.mean()), float(Xtr.std())
norm = lambda x: (x - MEAN) / STD

CH = [1, 16, 32, 64, 64] if SIZE == 64 else [1, 16, 32, 64, 64, 96]
NL = len(CH) - 1
key = jax.random.PRNGKey(0)


def init(key):
    p = {}
    for i in range(NL):
        key, k = jax.random.split(key)
        fan = 9 * CH[i]
        p[f"w{i}"] = jax.random.normal(k, (3, 3, CH[i], CH[i + 1])) * np.sqrt(2 / fan)
        p[f"b{i}"] = jnp.zeros(CH[i + 1])
    key, k = jax.random.split(key)
    p["wd"] = jax.random.normal(k, (CH[NL],)) * 0.05
    p["bd"] = jnp.array(0.0)
    return p


def features(p, x):
    for i in range(NL):
        x = jax.lax.conv_general_dilated(x, p[f"w{i}"], (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
        x = jax.nn.relu(x + p[f"b{i}"])
        if i < NL - 1:
            x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    return x  # (N, 8, 8, C)


def logits(p, x, drop_key=None):
    f = features(p, x).mean(axis=(1, 2))
    if drop_key is not None:
        f = f * jax.random.bernoulli(drop_key, 0.7, f.shape) / 0.7
    return f @ p["wd"] + p["bd"]


pos_w = float((1 - ytr.mean()) / ytr.mean())   # re-balance (74% of train is pneumonia)


def loss_fn(p, x, y, k):
    z = logits(p, x, k)
    w = jnp.where(y == 1, pos_w, 1.0)
    l = optax.sigmoid_binary_cross_entropy(z, y)
    return (w * l).mean() + 1e-4 * sum(jnp.sum(p[f"w{i}"] ** 2) for i in range(NL))


EPOCHS, BS = (30 if SIZE == 64 else 22), 64
steps = EPOCHS * (len(Xtr) // BS)
opt = optax.adamw(optax.cosine_decay_schedule(2e-3, steps), weight_decay=1e-4)
params = init(key)
state = opt.init(params)


@jax.jit
def step(p, s, x, y, k):
    l, g = jax.value_and_grad(loss_fn)(p, x, y, k)
    u, s = opt.update(g, s, p)
    return optax.apply_updates(p, u), s, l


predict = jax.jit(lambda p, x: jax.nn.sigmoid(logits(p, x)))


def augment(rng, x):
    # random shift up to 4 px, small brightness/contrast change
    n = len(x); out = np.empty_like(x)
    m = SIZE // 16
    pad = np.pad(x, ((0, 0), (m, m), (m, m), (0, 0)), mode="edge")
    dx, dy = rng.integers(0, 2 * m + 1, n), rng.integers(0, 2 * m + 1, n)
    for i in range(n):
        out[i] = pad[i, dy[i]:dy[i] + SIZE, dx[i]:dx[i] + SIZE]
    c = rng.uniform(0.85, 1.15, (n, 1, 1, 1)); b = rng.uniform(-0.06, 0.06, (n, 1, 1, 1))
    return np.clip(out * c + b, 0, 1)


rng = np.random.default_rng(0)
best, best_p = -1, None
t0 = time.time()
for ep in range(EPOCHS):
    perm = rng.permutation(len(Xtr))
    for b in range(len(Xtr) // BS):
        ix = perm[b * BS:(b + 1) * BS]
        key, k = jax.random.split(key)
        params, state, l = step(params, state, norm(augment(rng, Xtr[ix])), ytr[ix], k)
    pv = np.array(predict(params, norm(Xva)))
    auc = roc_auc_score(yva, pv)
    if auc > best:
        best, best_p = auc, jax.tree_util.tree_map(np.array, params)
    print(f"epoch {ep+1:2d} loss {float(l):.3f} val AUC {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

params = best_p
pv = np.array(predict(params, norm(Xva)))
# threshold: highest cut that still flags >= 95% of pneumonia in VALIDATION
cands = np.sort(pv[yva == 1])
thr = float(cands[int(np.floor(0.05 * len(cands)))])
pt = np.array(predict(params, norm(Xte)))
tn, fp, fn, tp = confusion_matrix(yte, pt >= thr).ravel()
res = {
    "dataset": f"PneumoniaMNIST {SIZE}x{SIZE} (MedMNIST v2), official test set, never seen in training",
    "test_images": int(len(yte)), "test_pneumonia": int(yte.sum()), "test_normal": int((1 - yte).sum()),
    "auc": round(float(roc_auc_score(yte, pt)), 3),
    "threshold": round(thr, 3),
    "sensitivity_pct": round(100 * tp / (tp + fn), 1),
    "specificity_pct": round(100 * tn / (tn + fp), 1),
    "flags_correct_pct": round(100 * tp / (tp + fp), 1),
    "missed_cases": int(fn), "false_flags": int(fp),
    "accuracy_pct": round(100 * (tp + tn) / len(yte), 1),
    "val_auc": round(float(best), 3),
}
print(json.dumps(res, indent=2))
OUT = ROOT / "export"
json.dump(res, open(OUT / "xray_results.json", "w"), indent=2)
exp = {"mean": MEAN, "std": STD, "threshold": thr, "channels": CH, "size": SIZE}
for k_, v in params.items():
    exp[k_] = {"shape": list(np.shape(v)), "data": np.round(np.ravel(v), 5).tolist()}
json.dump(exp, open(OUT / "xray_model.json", "w"), separators=(",", ":"))

# a few TEST images for the demo (de-identified public data) + reference outputs for JS check
pick = list(np.where(yte == 1)[0][:7]) + list(np.where(yte == 0)[0][:5])
json.dump({"images": [d["test_images"][i].ravel().tolist() for i in pick],
           "labels": [int(yte[i]) for i in pick],
           "ref_prob": [round(float(pt[i]), 5) for i in pick]},
          open(OUT / "xray_samples.json", "w"), separators=(",", ":"))
