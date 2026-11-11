"""
Flow Care - Chest CT lung-nodule WARNING model (never a diagnosis).

A small 3D convolutional neural network (3D CNN), written and trained from scratch in JAX.
Data: NoduleMNIST3D (MedMNIST v2, CC BY 4.0), built from the LIDC-IDRI chest CT
collection: 1,633 real, de-identified 3D CT crops (28x28x28 voxels) centred on a lung
nodule. Label: radiologists' malignancy rating, benign (0) vs suspicious for malignancy (1).
Official split: train 1,158 / validation 165 / test 310 (test never used for training or tuning).

Safety design: threshold tuned on VALIDATION to catch >= 90% of suspicious nodules,
because the AI only puts the scan higher on the radiologist's list.
Global-average-pooling at the end gives a 3D heat-map (class activation map) of where
the network looked, shown slice by slice in the dashboard.
"""
import json, time
import numpy as np
import jax, jax.numpy as jnp
import optax
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
d = np.load(ROOT / "xray_data" / "nodulemnist3d.npz")
prep = lambda a: (a.astype(np.float32) / 255.0)[..., None]          # (N, D, H, W, 1)
Xtr, ytr = prep(d["train_images"]), d["train_labels"][:, 0].astype(np.float32)
Xva, yva = prep(d["val_images"]), d["val_labels"][:, 0].astype(np.float32)
Xte, yte = prep(d["test_images"]), d["test_labels"][:, 0].astype(np.float32)
print("shapes", Xtr.shape, "suspicious share train/val/test",
      ytr.mean().round(3), yva.mean().round(3), yte.mean().round(3))
MEAN, STD = float(Xtr.mean()), float(Xtr.std())
norm = lambda x: (x - MEAN) / STD

CH = [1, 16, 32, 64]          # 28 -> pool 14 -> pool 7 -> conv -> GAP
NL = len(CH) - 1
DN = ("NDHWC", "DHWIO", "NDHWC")


def init(key):
    p = {}
    for i in range(NL):
        key, k = jax.random.split(key)
        p[f"w{i}"] = jax.random.normal(k, (3, 3, 3, CH[i], CH[i + 1])) * np.sqrt(2 / (27 * CH[i]))
        p[f"b{i}"] = jnp.zeros(CH[i + 1])
    key, k = jax.random.split(key)
    p["wd"] = jax.random.normal(k, (CH[NL],)) * 0.05
    p["bd"] = jnp.array(0.0)
    return p


def features(p, x):
    for i in range(NL):
        x = jax.lax.conv_general_dilated(x, p[f"w{i}"], (1, 1, 1), "SAME", dimension_numbers=DN)
        x = jax.nn.relu(x + p[f"b{i}"])
        if i < NL - 1:
            x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 2, 1), (1, 2, 2, 2, 1), "VALID")
    return x  # (N, 7, 7, 7, 64)


def logits(p, x, drop_key=None):
    f = features(p, x).mean(axis=(1, 2, 3))
    if drop_key is not None:
        f = f * jax.random.bernoulli(drop_key, 0.6, f.shape) / 0.6
    return f @ p["wd"] + p["bd"]


pos_w = float((1 - ytr.mean()) / ytr.mean())


def loss_fn(p, x, y, k):
    z = logits(p, x, k)
    w = jnp.where(y == 1, pos_w, 1.0)
    return (w * optax.sigmoid_binary_cross_entropy(z, y)).mean() + \
        3e-4 * sum(jnp.sum(p[f"w{i}"] ** 2) for i in range(NL))


EPOCHS, BS = 60, 32
steps = EPOCHS * (len(Xtr) // BS)
opt = optax.adamw(optax.cosine_decay_schedule(1.5e-3, steps), weight_decay=1e-4)
key = jax.random.PRNGKey(1)
params = init(key)
state = opt.init(params)


@jax.jit
def step(p, s, x, y, k):
    l, g = jax.value_and_grad(loss_fn)(p, x, y, k)
    u, s = opt.update(g, s, p)
    return optax.apply_updates(p, u), s, l


predict = jax.jit(lambda p, x: jax.nn.sigmoid(logits(p, x)))


def augment(rng, x):
    # random flips on all three axes + 90-degree rotation in the axial plane + small shift
    x = x.copy()
    for ax in (1, 2, 3):
        f = rng.random(len(x)) < 0.5
        x[f] = np.flip(x[f], axis=ax)
    r = rng.random(len(x)) < 0.5
    x[r] = np.rot90(x[r], axes=(2, 3))
    s = rng.integers(-2, 3, 3)
    x = np.roll(x, tuple(s), axis=(1, 2, 3))
    c = rng.uniform(0.9, 1.1, (len(x), 1, 1, 1, 1))
    return np.clip(x * c, 0, 1)


rng = np.random.default_rng(0)
best, best_p, t0 = -1, None, time.time()
for ep in range(EPOCHS):
    perm = rng.permutation(len(Xtr))
    for b in range(len(Xtr) // BS):
        ix = perm[b * BS:(b + 1) * BS]
        key, k = jax.random.split(key)
        params, state, l = step(params, state, norm(augment(rng, Xtr[ix])), ytr[ix], k)
    pv = np.array(predict(params, norm(Xva)))
    auc = roc_auc_score(yva, pv)
    if auc > best and ep >= 5:
        best, best_p = auc, jax.tree_util.tree_map(np.array, params)
    if ep % 5 == 4 or ep == EPOCHS - 1:
        print(f"epoch {ep+1:2d} loss {float(l):.3f} val AUC {auc:.4f} best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)

params = best_p
pv = np.array(predict(params, norm(Xva)))
cands = np.sort(pv[yva == 1])
thr = float(cands[int(np.floor(0.10 * len(cands)))])         # >= 90% sensitivity on validation
pt = np.array(predict(params, norm(Xte)))
tn, fp, fn, tp = confusion_matrix(yte, pt >= thr).ravel()
res = {
    "dataset": "NoduleMNIST3D 28x28x28 (MedMNIST v2, from LIDC-IDRI chest CT), official test set",
    "test_scans": int(len(yte)), "test_suspicious": int(yte.sum()), "test_benign": int((1 - yte).sum()),
    "auc": round(float(roc_auc_score(yte, pt)), 3), "threshold": round(thr, 3),
    "sensitivity_pct": round(100 * tp / (tp + fn), 1), "specificity_pct": round(100 * tn / (tn + fp), 1),
    "flags_correct_pct": round(100 * tp / (tp + fp), 1), "missed_cases": int(fn), "false_flags": int(fp),
    "accuracy_pct": round(100 * (tp + tn) / len(yte), 1), "val_auc": round(float(best), 3),
    "params": int(sum(np.size(v) for v in params.values())),
}
print(json.dumps(res, indent=2))
OUT = ROOT / "export"
json.dump(res, open(OUT / "ct_results.json", "w"), indent=2)
exp = {"mean": MEAN, "std": STD, "threshold": thr, "channels": CH, "size": 28}
for k_, v in params.items():
    exp[k_] = {"shape": list(np.shape(v)), "data": np.round(np.ravel(v), 5).tolist()}
json.dump(exp, open(OUT / "ct_model.json", "w"), separators=(",", ":"))
pick = list(np.where(yte == 1)[0][:5]) + list(np.where(yte == 0)[0][:5])
json.dump({"scans": [d["test_images"][i].ravel().tolist() for i in pick],
           "labels": [int(yte[i]) for i in pick], "ref_prob": [round(float(pt[i]), 5) for i in pick]},
          open(OUT / "ct_samples.json", "w"), separators=(",", ":"))
