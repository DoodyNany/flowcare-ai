"""Pick a realistic demo reading list (mostly normal, a few pneumonia, one false flag) from the TEST set."""
import json, numpy as np, jax, jax.numpy as jnp
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
X = json.load(open(ROOT / "export" / "xray_model.json"))
d = np.load(ROOT / "xray_data" / f"pneumoniamnist_{X['size']}.npz")
P = {k: np.array(v["data"], np.float32).reshape(v["shape"]) for k, v in X.items() if isinstance(v, dict)}
NL = len(X["channels"]) - 1
def prob(x):
    x = ((x.astype(np.float32) / 255 - X["mean"]) / X["std"])[..., None]
    for i in range(NL):
        x = jax.lax.conv_general_dilated(x, P[f"w{i}"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
        x = jax.nn.relu(x + P[f"b{i}"])
        if i < NL - 1: x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    return np.array(jax.nn.sigmoid(x.mean(axis=(1, 2)) @ P["wd"] + P["bd"][()]))
im, y = d["test_images"], d["test_labels"][:, 0]
p = np.concatenate([prob(im[i:i + 64]) for i in range(0, len(im), 64)])
t = X["threshold"]
pn = [i for i in np.where((y == 1) & (p >= t))[0][:4]]                      # 4 pneumonia, flagged
nn = [i for i in np.where((y == 0) & (p < t - 0.12))[0][:6]]                 # 6 clearly normal
bl = [i for i in np.where((y == 0) & (p < t) & (p >= t - 0.12))[0][:1]]      # 1 borderline normal
ff = [i for i in np.where((y == 0) & (p >= t))[0][:1]]                      # 1 false flag (AI wrong)
pick = pn + nn + bl + ff
print("picked", len(pick), "probs", np.round(p[pick], 2), "labels", y[pick])
json.dump({"images": [im[i].ravel().tolist() for i in pick], "labels": [int(y[i]) for i in pick],
           "ref_prob": [round(float(p[i]), 5) for i in pick]},
          open(ROOT / "export" / "xray_samples.json", "w"), separators=(",", ":"))
