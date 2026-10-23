"""Report figures + numbers. Writes PNGs to report/figures and a JSON of stats."""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import (PARAMS, clean, drop_bad_labels, features, fit, load,
                      metrics, predict)

FIG = Path("report/figures"); FIG.mkdir(parents=True, exist_ok=True)
TEAL = "#064A56"

train_raw, valid_raw, december_raw, template = load()
train_all = clean(train_raw)[0]
train = drop_bad_labels(train_all)
valid, fills = clean(valid_raw)

stats = {"raw_rows": int(len(train_raw)), "train_rows": int(len(train)),
         "dropped": int(len(train_all) - len(train)), "valid_rows": int(len(valid))}

# --- 1. price vs distance
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
ax.scatter(train.distance, train.posted_rate, s=2, alpha=.12, color=TEAL, lw=0,
           label=f"kept ({len(train):,})")
bad = train_all[(train_all.posted_rate / train_all.distance).gt(6)]
low = train_all[(train_all.posted_rate / train_all.distance).lt(1)]
ax.scatter(bad.distance, bad.posted_rate, s=14, color="#C0392B",
           label=f"above $6/mile ({len(bad)})")
ax.scatter(low.distance, low.posted_rate, s=14, color="#E67E22",
           label=f"below $1/mile ({len(low)})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Distance (miles, log)"); ax.set_ylabel("Posted rate ($, log)")
ax.set_title("Rate is almost a pure function of distance", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIG / "rate_vs_distance.png"); plt.close(fig)

# --- 2. $/mile distribution
rpm = train_all.posted_rate / train_all.distance
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
ax.hist(rpm.clip(0, 8), bins=120, color=TEAL, alpha=.85)
ax.axvline(6, color="#C0392B", ls="--", lw=1.2)
ax.set_xlabel("Rate per mile ($)"); ax.set_ylabel("Loads")
ax.set_title("0.56% of labels sit above $6/mile — treated as label errors",
             loc="left", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "rpm_hist.png"); plt.close(fig)

# --- 3. holdout actual vs predicted + residual spread
lo = pd.Timestamp("2025-10-01")
tr, va = train[train.date < lo], train[train.date >= lo]
cats = {c: tr[c].astype("category").cat.categories for c in ("equipment", "pickup", "delivery")}
model = lgb.LGBMRegressor(n_estimators=3000, **PARAMS)
model.fit(features(tr), np.log(tr.posted_rate),
          eval_set=[(features(va, cats), np.log(va.posted_rate))],
          eval_metric="l1", callbacks=[lgb.early_stopping(150, verbose=False)])
p = np.exp(model.predict(features(va, cats), num_iteration=model.best_iteration_))
hold = metrics(va.posted_rate.values, p)
hold["n"] = int(len(va)); hold["best_iter"] = int(model.best_iteration_)
stats["holdout"] = hold

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
axes[0].scatter(va.posted_rate, p, s=3, alpha=.15, color=TEAL, lw=0)
lim = [0, va.posted_rate.max() * 1.02]
axes[0].plot(lim, lim, color="#C0392B", lw=1)
axes[0].set_xlabel("Actual ($)"); axes[0].set_ylabel("Predicted ($)")
axes[0].set_title(f"Holdout Oct 2025 · MAE ${hold['MAE']:.0f} · MAPE {hold['MAPE']:.2f}%",
                  loc="left", fontweight="bold", fontsize=10)
errors = (p - va.posted_rate.values) / va.posted_rate.values * 100
axes[1].hist(np.clip(errors, -25, 25), bins=60, color=TEAL, alpha=.85)
axes[1].margins(y=0.12)
axes[1].set_xlabel("Percentage error (%), clipped to ±25%"); axes[1].set_ylabel("Loads")
axes[1].set_title(f"Residuals: median {np.median(errors):+.1f}%, 1st/99th pct "
                  f"{np.percentile(errors,1):+.1f}/{np.percentile(errors,99):+.1f}%",
                  loc="left", fontweight="bold", fontsize=10)
for a in axes:
    a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIG / "holdout_diagnostics.png"); plt.close(fig)

# --- 4. q-q of residual by distance band (where the money is)
band = pd.cut(va.distance, [0, 500, 1000, 1500, 3500])
band_mae = pd.DataFrame({"band": band, "abs_pct": np.abs(errors)}).groupby("band", observed=True).agg(
    n=("abs_pct", "size"), MAPE=("abs_pct", "mean")).round(2)
band_mae.to_csv("report/figures/mape_by_distance.csv")

# --- final model on all labelled data, for the December story
final = fit(train)
dec = december_raw.copy()
dec["weight"] = dec["weight"].fillna(fills["weight"])
coords = {}
pairs = pd.concat([
    train[["pickup", "pickup_lat", "pickup_lon"]].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
    train[["delivery", "delivery_lat", "delivery_lon"]].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})])
for city, lat, lon in pairs.drop_duplicates("city")[["city", "lat", "lon"]].itertuples(index=False):
    coords[city] = (lat, lon)
for side in ("pickup", "delivery"):
    c = dec[side].map(coords)
    dec[f"{side}_lat"] = c.str[0]; dec[f"{side}_lon"] = c.str[1]
allcats = {c: train[c].astype("category").cat.categories for c in ("equipment", "pickup", "delivery")}
dec_pred = predict(final, dec.assign(**{k: dec[k] for k in dec}), allcats)
stats["december"] = {"min": float(dec_pred.min()), "max": float(dec_pred.max()),
                     "mean": float(dec_pred.mean()),
                     "by_dow": {str(d): float(dec_pred[dec.date.dt.day_of_week == d].mean()) for d in range(7)}}

imp = pd.Series(final.feature_importances_, index=features(train).columns).sort_values(ascending=False)
stats["importance"] = {k: int(v) for k, v in imp.items()}
stats["cv"] = pd.read_csv("outputs/cv_metrics.csv").round(4).to_dict("records")
json.dump(stats, open("report/stats.json", "w"), indent=2, default=str)
print(json.dumps({k: stats[k] for k in ("raw_rows", "train_rows", "dropped", "valid_rows", "holdout", "december")}, indent=2, default=str))
print(band_mae.to_string())
print(imp.to_string())
