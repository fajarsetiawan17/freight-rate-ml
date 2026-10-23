"""Freight rate prediction — final pipeline.

Single entry point: clean data, engineer features, CV-validate, fit on all labelled
data, write validation_predictions.csv and december_chart_inputs.csv.

    python pipeline.py            # train + write submission files
    python pipeline.py --cv       # also print rolling-origin CV table
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# $/mile outside this band is a data error, not a price (0.04% of rows)
RPM_MIN, RPM_MAX = 1.0, 6.0
# best iterations found by rolling-origin CV, scaled to the full training set
N_ESTIMATORS = 600
PARAMS = dict(objective="regression_l1", learning_rate=0.03, num_leaves=63,
              min_child_samples=40, subsample=0.85, subsample_freq=1,
              colsample_bytree=0.85, reg_lambda=1.0, random_state=42, verbose=-1)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    return train, valid, december, template


def clean(df: pd.DataFrame, fill_values: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fix data-quality issues. Returns the cleaned frame plus imputation values
    so the exact same fills can be reused on unlabelled data."""
    df = df.copy()
    if "weight" in df:
        df["weight"] = df["weight"].abs()          # 292 sign-flipped weights

    fills = fill_values or {
        "weight": float(df["weight"].median()),
        "market_index": float(df["market_index"].median()),
    }
    if "weight" in df:
        df["weight"] = df["weight"].fillna(fills["weight"])
    if "market_index" in df:
        df["market_index"] = df["market_index"].fillna(fills["market_index"])
    return df, fills


def city_coords(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """City -> (lat, lon). december_chart_inputs.csv ships no coordinates."""
    pairs = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].rename(
            columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        train[["delivery", "delivery_lat", "delivery_lon"]].rename(
            columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
    ])
    return {city: (float(lat), float(lon))
            for city, lat, lon in pairs.drop_duplicates("city")[["city", "lat", "lon"]].itertuples(index=False)}


def add_coords(df: pd.DataFrame, lookup: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    for side in ("pickup", "delivery"):
        if f"{side}_lat" not in df:
            coords = df[side].map(lookup)
            df[f"{side}_lat"] = coords.str[0]
            df[f"{side}_lon"] = coords.str[1]
    return df


def drop_bad_labels(df: pd.DataFrame) -> pd.DataFrame:
    rpm = df["posted_rate"] / df["distance"]
    bad = (rpm > RPM_MAX) | (rpm < RPM_MIN)
    return df[~bad].copy()


def features(df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["log_distance"] = np.log(df["distance"])
    f["log_weight"] = np.log(df["weight"].clip(lower=1000))
    f["lat_mid"] = (df["pickup_lat"] + df["delivery_lat"]) / 2
    f["lon_mid"] = (df["pickup_lon"] + df["delivery_lon"]) / 2
    f["north_south"] = df["delivery_lat"] - df["pickup_lat"]
    f["east_west"] = df["delivery_lon"] - df["pickup_lon"]
    f["day_of_week"] = df["date"].dt.day_of_week
    for col in ("equipment", "pickup", "delivery"):
        f[col] = df[col].astype("category")
        if categories:
            f[col] = f[col].cat.set_categories(categories[col])
    return f


def fit(train: pd.DataFrame, n_estimators: int = N_ESTIMATORS) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **PARAMS)
    model.fit(features(train), np.log(train["posted_rate"]))
    return model


def predict(model: lgb.LGBMRegressor, df: pd.DataFrame, categories: dict) -> np.ndarray:
    return np.exp(model.predict(features(df, categories)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE": (np.abs(err) / y).mean() * 100, "bias": err.mean()}


def rolling_cv(train: pd.DataFrame) -> pd.DataFrame:
    """Three expanding-window folds; every fold predicts one unseen future month."""
    rows = []
    for start in ("2025-08-01", "2025-09-01", "2025-10-01"):
        lo, hi = pd.Timestamp(start), pd.Timestamp(start) + pd.offsets.MonthBegin(1)
        tr, va = train[train.date < lo], train[(train.date >= lo) & (train.date < hi)]
        cats = {c: tr[c].astype("category").cat.categories for c in ("equipment", "pickup", "delivery")}
        model = lgb.LGBMRegressor(n_estimators=3000, **PARAMS)
        model.fit(features(tr), np.log(tr["posted_rate"]),
                  eval_set=[(features(va, cats), np.log(va["posted_rate"]))],
                  eval_metric="l1", callbacks=[lgb.early_stopping(150, verbose=False)])
        p = np.exp(model.predict(features(va, cats), num_iteration=model.best_iteration_))
        rows.append({"fold": f"train<{start} -> {start}+1mo", "n_train": len(tr),
                     "best_iter": model.best_iteration_, **metrics(va["posted_rate"].values, p)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="print rolling-origin CV table")
    args = ap.parse_args()

    train_raw, valid_raw, december_raw, template = load()
    train = drop_bad_labels(clean(train_raw)[0])
    valid, fills = clean(valid_raw)
    december, _ = clean(december_raw, fills)
    lookup = city_coords(train)
    december = add_coords(december, lookup)
    print(f"train {len(train):,} rows after cleaning | valid {len(valid):,} | december {len(december)}")

    if args.cv:
        cv = rolling_cv(train)
        print(cv.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        OUT.mkdir(exist_ok=True)
        cv.to_csv(OUT / "cv_metrics.csv", index=False)

    model = fit(train)
    categories = {c: train[c].astype("category").cat.categories for c in ("equipment", "pickup", "delivery")}

    preds = pd.DataFrame({"load_id": template["load_id"],
                          "predicted_rate": predict(model, valid, categories).round(2)})
    december["predicted_rate"] = predict(model, december, categories).round(2)
    december_out = december[list(december_raw.columns)]

    OUT.mkdir(exist_ok=True)
    preds.to_csv(OUT / "validation_predictions.csv", index=False)
    december_out.to_csv(OUT / "december_chart_inputs.csv", index=False)
    print(f"wrote {OUT/'validation_predictions.csv'} ({len(preds):,} rows)")
    print(f"wrote {OUT/'december_chart_inputs.csv'} ({len(december)} rows)")
    print(f"predicted_rate: min {preds.predicted_rate.min():.2f} "
          f"median {preds.predicted_rate.median():.2f} max {preds.predicted_rate.max():.2f}")


if __name__ == "__main__":
    main()
