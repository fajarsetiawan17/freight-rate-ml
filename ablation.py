"""Feature ablation used in the report. Rolling-origin CV, 3 folds, L1 objective."""
import numpy as np, pandas as pd, lightgbm as lgb
from pipeline import load, clean, drop_bad_labels, features, metrics

PARAMS = dict(objective="regression_l1", learning_rate=0.03, num_leaves=63, min_child_samples=40,
              subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0,
              random_state=42, verbose=-1, n_estimators=3000)

def run(train, extra=(), drop=()):
    rows = []
    for start in ("2025-08-01", "2025-09-01", "2025-10-01"):
        lo, hi = pd.Timestamp(start), pd.Timestamp(start) + pd.offsets.MonthBegin(1)
        tr, va = train[train.date < lo], train[(train.date >= lo) & (train.date < hi)]
        cats = {c: tr[c].astype("category").cat.categories for c in ("equipment", "pickup", "delivery")}
        Xtr, Xva = features(tr), features(va, cats)
        if drop:
            Xtr = Xtr.drop(columns=list(drop)); Xva = Xva.drop(columns=list(drop))
        for col in extra:
            Xtr[col] = tr[col].values; Xva[col] = va[col].values
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(Xtr, np.log(tr.posted_rate), eval_set=[(Xva, np.log(va.posted_rate))],
              eval_metric="l1", callbacks=[lgb.early_stopping(150, verbose=False)])
        rows.append(metrics(va.posted_rate.values, np.exp(m.predict(Xva, num_iteration=m.best_iteration_))))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

train = drop_bad_labels(clean(load()[0])[0])
tests = {
    "A: core (distance, weight, equipment, pickup, delivery)":
        dict(drop=("lat_mid", "lon_mid", "north_south", "east_west", "day_of_week")),
    "B: A + geography": dict(drop=("day_of_week",)),
    "C: B + day_of_week (chosen)": dict(),
    "D: C + market_index": dict(extra=("market_index",)),
    "E: C + quote_signal": dict(extra=("quote_signal",)),
    "F: C + market_index + quote_signal": dict(extra=("market_index", "quote_signal")),
}
out = []
for name, kw in tests.items():
    r = run(train, **kw)
    out.append({"variant": name, **r})
    print(f"{name:52s} MAE {r['MAE']:6.2f} RMSE {r['RMSE']:6.1f} MAPE {r['MAPE']:.2f}%")
pd.DataFrame(out).to_csv("outputs/ablation.csv", index=False)
