"""
Flow Care - Patient deterioration risk AI (warn-only).

Predicts the chance that a ward patient deteriorates in the next 24 hours
(ICU transfer / rapid-response call) from vital signs + 3 routine labs.

DATA: SYNTHETIC ward patients (no real patient data). Physiology ranges and the
outcome relationship are modelled on published early-warning literature, so the
accuracy numbers show the method works, NOT how it would perform on real patients.

Baseline for comparison: NEWS2 (Royal College of Physicians, 2017), the score
hospitals already use.  Model: logistic regression (every factor is explainable).
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

OUT = Path(__file__).resolve().parent.parent / "export"
rng = np.random.default_rng(7)
N = 60000

# ---- synthetic patients: hidden "illness severity" drives everything ----
age = np.clip(rng.normal(58, 18, N), 18, 98)
sev = np.clip(rng.gamma(1.4, 0.7, N) + 0.012 * (age - 58), 0, None)        # hidden, 0 = well
sepsis = rng.random(N) < 0.10 + 0.08 * np.tanh(sev)        # some severity is infection-driven
hr = 78 + 9 * sev + 14 * sepsis + rng.normal(0, 11, N)
rr = 15 + 2.2 * sev + 3 * sepsis + rng.normal(0, 2.5, N)
sbp = 128 - 7 * sev - 14 * sepsis * (sev > 1.2) + rng.normal(0, 15, N)
temp = 36.9 + 0.9 * sepsis + 0.1 * sev + rng.normal(0, 0.45, N) - 0.6 * (sepsis & (sev > 2.5))
spo2 = np.clip(97.5 - 1.6 * sev + rng.normal(0, 1.4, N), 75, 100)
on_o2 = (spo2 < 93) | (rng.random(N) < 0.05 + 0.05 * sev)
spo2 = np.where(on_o2, np.clip(spo2 + 3, 75, 100), spo2)   # oxygen hides low saturation
confused = rng.random(N) < 1 / (1 + np.exp(-(sev - 3.2) * 1.6)) + 0.02 * (age > 80)
lactate = np.clip(1.1 + 0.35 * sev ** 1.3 + 1.2 * sepsis * (sev > 1.5) + rng.normal(0, 0.4, N), 0.4, 15)
wbc = np.clip(8 + 5 * sepsis + 0.6 * sev + rng.normal(0, 2.5, N), 0.5, 40)
creat = np.clip(85 + 22 * sev + 0.6 * (age - 58) + rng.normal(0, 25, N), 30, 900)

logit = (-5.0 + 1.05 * sev + 0.9 * sepsis * (lactate > 2.5) + 0.018 * (age - 58)
         + 0.6 * confused + 0.5 * on_o2 + rng.normal(0, 0.5, N))
y = rng.random(N) < 1 / (1 + np.exp(-logit))
print(f"deterioration rate: {y.mean():.1%}")

raw = dict(age=age, hr=hr, rr=rr, sbp=sbp, temp=temp, spo2=spo2, on_o2=on_o2.astype(float),
           confused=confused.astype(float), lactate=lactate, wbc=wbc, creat=creat)


def news2(r):
    s = np.zeros(len(r["hr"]))
    rr_, sp, sb, h, t = r["rr"], r["spo2"], r["sbp"], r["hr"], r["temp"]
    s += np.select([rr_ <= 8, rr_ <= 11, rr_ <= 20, rr_ <= 24], [3, 1, 0, 2], 3)
    s += np.select([sp <= 91, sp <= 93, sp <= 95], [3, 2, 1], 0)
    s += 2 * r["on_o2"]
    s += np.select([sb <= 90, sb <= 100, sb <= 110, sb <= 219], [3, 2, 1, 0], 3)
    s += np.select([h <= 40, h <= 50, h <= 90, h <= 110, h <= 130], [3, 1, 0, 1, 2], 3)
    s += 3 * r["confused"]
    s += np.select([t <= 35, t <= 36, t <= 38, t <= 39], [3, 1, 0, 1], 2)
    return s


# explainable engineered features: "how far outside normal" for each sign
FEATS = {
    "Age (per 10 y over 50)":   lambda r: np.maximum(0, r["age"] - 50) / 10,
    "Fast heart rate":          lambda r: np.maximum(0, r["hr"] - 90) / 10,
    "Slow heart rate":          lambda r: np.maximum(0, 50 - r["hr"]) / 10,
    "Fast breathing":           lambda r: np.maximum(0, r["rr"] - 20) / 2,
    "Low blood pressure":       lambda r: np.maximum(0, 110 - r["sbp"]) / 10,
    "Low oxygen saturation":    lambda r: np.maximum(0, 96 - r["spo2"]),
    "On supplemental oxygen":   lambda r: r["on_o2"],
    "Fever":                    lambda r: np.maximum(0, r["temp"] - 38),
    "Low temperature":          lambda r: np.maximum(0, 36 - r["temp"]),
    "New confusion":            lambda r: r["confused"],
    "High lactate":             lambda r: np.maximum(0, r["lactate"] - 2),
    "Abnormal white cells":     lambda r: np.maximum(0, r["wbc"] - 11) / 2 + np.maximum(0, 4 - r["wbc"]) / 2,
    "High creatinine":          lambda r: np.maximum(0, r["creat"] - 110) / 50,
}
X = np.column_stack([f(raw) for f in FEATS.values()])
idx = rng.permutation(N); tr, te = idx[: int(.75 * N)], idx[int(.75 * N):]
m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
p = m.predict_proba(X[te])[:, 1]
nw = news2(raw)[te]

auc_ai, auc_news = roc_auc_score(y[te], p), roc_auc_score(y[te], nw)
# compare at the same alert workload as NEWS2 >= 5 ("urgent response" trigger)
alert_news = nw >= 5
rate = alert_news.mean()
cut = np.quantile(p, 1 - rate)
alert_ai = p >= cut
res = {
    "data": "synthetic ward patients (no real data)",
    "test_patients": int(len(te)), "deterioration_rate_pct": round(100 * y[te].mean(), 1),
    "auc_ai": round(auc_ai, 3), "auc_news2": round(auc_news, 3),
    "same_alert_rate_pct": round(100 * rate, 1),
    "caught_ai_pct": round(100 * recall_score(y[te], alert_ai), 1),
    "caught_news2_pct": round(100 * recall_score(y[te], alert_news), 1),
    "alert_correct_ai_pct": round(100 * precision_score(y[te], alert_ai), 1),
    "alert_correct_news2_pct": round(100 * precision_score(y[te], alert_news), 1),
    "ai_threshold_prob": round(float(cut), 3),
}
print(json.dumps(res, indent=2))
json.dump({"features": list(FEATS), "coef": np.round(m.coef_[0], 4).tolist(),
           "intercept": round(float(m.intercept_[0]), 4), "alert_threshold": round(float(cut), 3),
           "watch_threshold": round(float(np.quantile(p, 0.80)), 3)},
          open(OUT / "risk_model.json", "w"))
json.dump(res, open(OUT / "risk_results.json", "w"), indent=2)
for n, c in zip(FEATS, m.coef_[0]):
    print(f"{n:28s} {c:+.2f}")
