"""
Flow Care - Hospital simulation (SYNTHETIC DATA, no real patients).

Simulates hourly patient flow 2023-01-01 .. 2026-12-31 for a Jeddah hospital:
ED, Internal Medicine (IM), Surgery, OB/GYN, ICU.

Realism built in (from published patterns):
  * daily rhythm: ED arrivals peak late morning/evening; ward discharges happen 10:00-17:00
  * weekly rhythm: Saudi weekend = Friday/Saturday
  * Islamic calendar (Umm al-Qura): Ramadan shifts ED visits to night,
    Hajj season (Dhul-Hijjah 1-15) raises ED/IM demand in Jeddah, Eid spikes
  * winter respiratory season raises IM/ICU admissions
  * wards: length of stay is random (log-normal); a patient is "ready" after LOS
    but only leaves during the discharge window -> realistic bed blocking

Outputs: data/hourly_census.csv, data/daily_ed.csv
"""
import numpy as np
import pandas as pd
from hijridate import Gregorian
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

START, END = "2023-01-01", "2026-12-31 23:00"
hours = pd.date_range(START, END, freq="h")
H = len(hours)

DEPTS = {
    # name: (beds, mean arrivals/hour, mean LOS hours, LOS sigma, discharge window?)
    "ED":    (46, 6.2, 4.0, 0.6, False),
    "IM":    (98, 0.62, 4.0 * 24, 0.55, True),
    "SURG":  (58, 0.50, 3.0 * 24, 0.60, True),
    "OBGYN": (36, 0.40, 2.2 * 24, 0.40, True),
    "ICU":   (24, 0.11, 4.5 * 24, 0.65, True),
}

# ---------- calendar features (shared with the model code) ----------
days = pd.date_range(START, "2026-12-31", freq="D")
hij = {d: Gregorian(d.year, d.month, d.day).to_hijri() for d in days}
cal = pd.DataFrame({
    "date": days,
    "h_month": [hij[d].month for d in days],
    "h_day": [hij[d].day for d in days],
})
cal["ramadan"] = (cal.h_month == 9).astype(int)
cal["hajj"] = ((cal.h_month == 12) & (cal.h_day <= 15)).astype(int)
cal["eid"] = (((cal.h_month == 10) & (cal.h_day <= 3)) |
              ((cal.h_month == 12) & (cal.h_day >= 10) & (cal.h_day <= 13))).astype(int)
cal.to_csv(OUT / "calendar.csv", index=False)
cal_idx = cal.set_index("date")

day_of = hours.normalize()
hr = hours.hour.values
dow = hours.dayofweek.values          # Mon=0 .. Fri=4, Sat=5
ram = cal_idx.loc[day_of, "ramadan"].values
hajj = cal_idx.loc[day_of, "hajj"].values
eid = cal_idx.loc[day_of, "eid"].values
month = hours.month.values
years_from_start = (hours - hours[0]).days.values / 365.25

# ED hourly profile (normal vs Ramadan, where activity moves to night)
ed_norm = np.array([.55,.45,.38,.33,.30,.32,.45,.70,1.0,1.25,1.4,1.45,
                    1.4,1.35,1.3,1.3,1.35,1.4,1.45,1.45,1.35,1.15,.9,.7])
ed_ram = np.array([1.2,1.3,1.25,1.1,.8,.5,.35,.35,.45,.6,.75,.85,
                   .9,.9,.9,.9,.85,.7,.6,1.1,1.5,1.6,1.5,1.35])
ed_norm /= ed_norm.mean(); ed_ram /= ed_ram.mean()
adm_profile = np.clip(ed_norm ** 0.6, 0, None); adm_profile /= adm_profile.mean()

winter = np.isin(month, [11, 12, 1, 2]).astype(float)
weekend = np.isin(dow, [4, 5]).astype(float)
growth = 1 + 0.03 * years_from_start           # 3% demand growth per year

census = {}
arrivals_rec = {}
for name, (beds, lam, los_mu, los_sig, window) in DEPTS.items():
    if name == "ED":
        prof = np.where(ram == 1, ed_ram[hr], ed_norm[hr])
        mult = (1 + 0.55 * hajj + 0.35 * eid + 0.10 * winter + 0.08 * weekend - 0.05 * ram)
    elif name == "IM":
        prof = adm_profile[hr]
        mult = (1 + 0.35 * hajj + 0.25 * winter + 0.10 * eid - 0.12 * weekend)
    elif name == "SURG":   # elective surgery mostly Sun-Thu, reduced in Hajj/Eid
        prof = np.where((hr >= 7) & (hr <= 17), 1.9, 0.35)
        mult = (1 - 0.65 * weekend - 0.30 * eid - 0.15 * hajj + 0.10 * (1 - weekend))
    elif name == "OBGYN":
        prof = np.ones(H)
        mult = (1 + 0.05 * winter - 0.04 * ram)
    else:  # ICU
        prof = np.ones(H)
        mult = (1 + 0.40 * winter + 0.45 * hajj + 0.15 * eid)
    rate = lam * prof * np.clip(mult, 0.05, None) * growth
    arr = RNG.poisson(rate)
    arrivals_rec[name] = arr

    departures = np.zeros(H + 24 * 60, dtype=np.int64)
    tot = int(arr.sum())
    los = RNG.lognormal(np.log(los_mu), los_sig, tot)
    arr_hours = np.repeat(np.arange(H), arr)
    ready = arr_hours + np.ceil(los).astype(int)
    if window:
        # discharges only 10:00-17:00, and slower at weekends (+ one extra day 40% of time)
        rh = hours[np.clip(ready, 0, H - 1)].hour.values
        shift = np.where(rh < 10, 10 - rh, np.where(rh > 17, 24 - rh + 10 + RNG.integers(0, 4, tot), 0))
        ready = ready + shift + RNG.integers(0, 3, tot)
        rd = hours[np.clip(ready, 0, H - 1)].dayofweek.values
        wk = np.isin(rd, [4, 5]) & (RNG.random(tot) < 0.4)
        ready = ready + wk * 24
    np.add.at(departures, ready, 1)
    occ = np.cumsum(arr) - np.cumsum(departures[:H])
    census[name] = occ

df = pd.DataFrame({"time": hours})
for name in DEPTS:
    df[name] = census[name]
    df[f"{name}_arr"] = arrivals_rec[name]
# drop first 30 days (warm-up while wards fill)
df = df[df.time >= "2023-02-01"].reset_index(drop=True)
df.to_csv(OUT / "hourly_census.csv", index=False)

daily = df.set_index("time")["ED_arr"].resample("D").sum().rename("ed_visits").reset_index()
daily.to_csv(OUT / "daily_ed.csv", index=False)

print(df[list(DEPTS)].describe().round(1))
print("beds:", {k: v[0] for k, v in DEPTS.items()})
print(daily.describe().round(1))
