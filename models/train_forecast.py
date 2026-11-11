"""
Flow Care - Forecasting AI (trained from scratch on our simulated hospital).

Model A: each department's census 8 hours ahead (gradient-boosted decision trees).
Model B: daily ED visits 1-7 days ahead, using the Islamic calendar (Hajj/Ramadan planner).

Train: Feb 2023 - Dec 2025.  Test: all of 2026 (never seen in training).
Models are exported as plain JSON trees so the dashboard runs them in the browser
with no server and no outside AI service.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "export"
OUT.mkdir(exist_ok=True)
BEDS = {"ED": 46, "IM": 98, "SURG": 58, "OBGYN": 36, "ICU": 24}
HORIZON = 8
TEST_START = pd.Timestamp("2026-01-01")

df = pd.read_csv(DATA / "hourly_census.csv", parse_dates=["time"])
cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["date"]).set_index("date")
day = df.time.dt.normalize()
for c in ["ramadan", "hajj", "eid"]:
    df[c] = cal.loc[day, c].values
df["hour"] = df.time.dt.hour
df["dow"] = df.time.dt.dayofweek
df["winter"] = df.time.dt.month.isin([11, 12, 1, 2]).astype(int)

HOURLY_FEATURES = ["now", "lag1", "lag3", "lag6", "lag24", "lag168", "arr3", "arr24",
                   "hour", "dow", "ramadan", "hajj", "eid", "winter"]


def hourly_frame(d):
    c, a = df[d], df[f"{d}_arr"]
    X = pd.DataFrame({
        "now": c, "lag1": c.shift(1), "lag3": c.shift(3), "lag6": c.shift(6),
        "lag24": c.shift(24), "lag168": c.shift(168),
        "arr3": a.rolling(3).sum(), "arr24": a.rolling(24).sum(),
        "hour": df.hour, "dow": df.dow, "ramadan": df.ramadan, "hajj": df.hajj,
        "eid": df.eid, "winter": df.winter,
    })
    y = c.shift(-HORIZON) - c          # predict the CHANGE over the next 8 hours
    return X, y


def export_gbr(m, feats):
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "f": t.feature.tolist(), "t": np.round(t.threshold, 4).tolist(),
            "l": t.children_left.tolist(), "r": t.children_right.tolist(),
            "v": np.round(t.value[:, 0, 0] * m.learning_rate, 5).tolist(),
        })
    return {"features": feats, "init": float(m.init_.constant_[0][0]), "trees": trees}


results, export = {}, {"horizon": HORIZON, "beds": BEDS, "depts": {}}
for d in BEDS:
    X, y = hourly_frame(d)
    ok = X.notna().all(axis=1) & y.notna()
    tr = ok & (df.time < TEST_START)
    te = ok & (df.time >= TEST_START)
    m = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.08,
                                  subsample=0.8, random_state=0)
    m.fit(X[tr], y[tr])
    pred = X.loc[te, "now"] + m.predict(X[te])
    truth = X.loc[te, "now"] + y[te]
    mae_ai = float(np.mean(np.abs(pred - truth)))
    mae_base = float(np.mean(np.abs(X.loc[te, "now"] - truth)))   # "same as now" guess

    # Early warning: an EVENT = the dept goes from below to >=95% full.
    # AI ALERT at hour t = forecast(t+8) >= 95% of beds while currently below.
    thr = 0.95 * BEDS[d]
    full_all = (df[d] >= thr).values
    alert_all = np.zeros(len(df), bool)
    alert_all[np.where(te)[0]] = (pred.values >= thr) & (X.loc[te, "now"].values < thr)
    test_idx = np.where(df.time >= TEST_START)[0]
    events = [i for i in test_idx[1:] if full_all[i] and not full_all[i - 1]]
    lead = []
    for i in events:
        win = [k for k in range(i - HORIZON, i - 1) if alert_all[k]]   # 2..8 h before
        lead.append(i - win[0] if win else 0)
    lead = np.array(lead)
    warned = lead >= 2
    # false alarms: alerts with no event in the following 10 hours
    al = np.where(alert_all)[0]
    fa = np.mean([not full_all[k + 1:k + 11].any() for k in al]) if len(al) else 0.0
    results[d] = {
        "mae_ai_beds": round(mae_ai, 2), "mae_simple_guess_beds": round(mae_base, 2),
        "improvement_pct": round(100 * (1 - mae_ai / mae_base), 1),
        "fill_up_events_2026": len(events),
        "warned_2h_plus_pct": round(100 * warned.mean(), 1) if len(events) else None,
        "median_warning_hours": float(np.median(lead[warned])) if warned.any() else None,
        "false_alarm_pct": round(100 * fa, 1),
    }
    export["depts"][d] = export_gbr(m, HOURLY_FEATURES)
    import joblib; joblib.dump(m, OUT / f"_sk_{d}.pkl")
    print(d, results[d])

# ---------------- Model B: 7-day ED planner ----------------
daily = pd.read_csv(DATA / "daily_ed.csv", parse_dates=["time"]).rename(columns={"time": "date"})
daily = daily.set_index("date")
calj = cal.reindex(daily.index)
rows = []
v = daily.ed_visits
for k in range(1, 8):
    base = pd.DataFrame(index=daily.index)
    base["k"] = k
    base["mean7"] = v.rolling(7).mean()             # known on the forecast day
    base["mean28"] = v.rolling(28).mean()
    tgt = daily.index + pd.Timedelta(days=k)
    base["target_date"] = tgt
    tc = cal.reindex(tgt)
    base["dow"] = tgt.dayofweek
    base["winter"] = tgt.month.isin([11, 12, 1, 2]).astype(int)
    base["ramadan"], base["hajj"], base["eid"] = tc.ramadan.values, tc.hajj.values, tc.eid.values
    base["h_day_dh"] = np.where(tc.h_month.values == 12, tc.h_day.values, 0)  # day within Dhul-Hijjah
    base["y"] = v.reindex(tgt).values
    rows.append(base)
B = pd.concat(rows).dropna()
DAILY_FULL = ["k", "mean7", "mean28", "dow", "winter", "ramadan", "hajj", "eid", "h_day_dh"]
DAILY_NOHIJRI = ["k", "mean7", "mean28", "dow", "winter"]
trB, teB = B.target_date < TEST_START, B.target_date >= TEST_START
mB = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.06, random_state=0)
mB.fit(B.loc[trB, DAILY_FULL], B.loc[trB, "y"])
mB0 = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.06, random_state=0)
mB0.fit(B.loc[trB, DAILY_NOHIJRI], B.loc[trB, "y"])
hajj26 = teB & (B.target_date.isin(cal.index[cal.hajj == 1]))
ram26 = teB & (B.target_date.isin(cal.index[cal.ramadan == 1]))
week7 = (B.k == 7)


def mae(m, f, mask):
    return round(float(np.mean(np.abs(m.predict(B.loc[mask, f]) - B.loc[mask, "y"]))), 1)


results["ed_7day_planner"] = {
    "all_2026_7days_ahead_mae_visits": mae(mB, DAILY_FULL, teB & week7),
    "hajj_2026_7days_ahead_mae_with_islamic_calendar": mae(mB, DAILY_FULL, hajj26 & week7),
    "hajj_2026_7days_ahead_mae_without_islamic_calendar": mae(mB0, DAILY_NOHIJRI, hajj26 & week7),
    "ramadan_2026_7days_ahead_mae_with": mae(mB, DAILY_FULL, ram26 & week7),
    "ramadan_2026_7days_ahead_mae_without": mae(mB0, DAILY_NOHIJRI, ram26 & week7),
    "hajj_2026_avg_daily_visits": round(float(B.loc[hajj26 & week7, "y"].mean()), 1),
}
print(results["ed_7day_planner"])
export["ed_daily"] = export_gbr(mB, DAILY_FULL)
import joblib; joblib.dump(mB, OUT / "_sk_daily.pkl")

# data the dashboard replays (2026 only) + calendar flags
d26 = df[df.time >= "2025-12-24"]
export["replay"] = {
    "start": str(d26.time.iloc[0]),
    "census": {d: d26[d].astype(int).tolist() for d in BEDS},
    "arrivals": {d: d26[f"{d}_arr"].astype(int).tolist() for d in BEDS},
}
c26 = cal[cal.index >= "2025-11-01"]
export["calendar"] = {"start": str(c26.index[0].date()),
                      "ramadan": c26.ramadan.tolist(), "hajj": c26.hajj.tolist(),
                      "eid": c26.eid.tolist(),
                      "h_day_dh": np.where(c26.h_month == 12, c26.h_day, 0).tolist(),
                      "h_month": c26.h_month.tolist(), "h_day": c26.h_day.tolist()}
dd = daily[daily.index >= "2025-11-01"].ed_visits
export["daily_ed"] = {"start": str(dd.index[0].date()), "visits": dd.astype(int).tolist()}

json.dump(export, open(OUT / "forecast_model.json", "w"), separators=(",", ":"))
json.dump(results, open(OUT / "forecast_results.json", "w"), indent=2)
print("saved", (OUT / "forecast_model.json").stat().st_size // 1024, "KB")
