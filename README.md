# Freight Rate Prediction

Machine Learning Engineer assessment submission. Predicts the posted rate of a
truckload from lane, distance, equipment, weight and quote date.

Holdout (train ≤ Sep 2025, predict Oct 2025): **MAE $52 · RMSE $278 · MAPE 1.99%**
on 4,782 unseen loads.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python pipeline.py --cv            # train, validate, write outputs/
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv
```

`score.py` validates both files and writes `scorer_results/candidate_december.png`.

## Layout

```
data/                       supplied inputs (train_test, validation, december, template)
pipeline.py                 clean -> features -> fit -> submission files  (the deliverable)
ablation.py                 feature-set ablation behind section 6.1 of the report
make_report_assets.py       figures + statistics used by the report
build_report.py             renders report/Freight_Rate_ML_Report.docx
outputs/                    validation_predictions.csv, december_chart_inputs.csv, cv_metrics.csv
report/                     Freight_Rate_ML_Report.pdf (+ .docx, figures)
scorer_results/             candidate_december.png
```

## Approach in one paragraph

Rate is nearly a pure function of distance (corr of logs = 0.99), so the model
only has to learn level corrections on top of it. LightGBM with an L1 objective
on `log(rate)`, nine features — log distance, log weight, equipment, origin,
destination, midpoint lat/lon, north–south and east–west deltas, day of week.
Validation is rolling-origin, three expanding-window folds each predicting one
unseen future month, because the real submission is a two-month forward forecast
and random K-fold would leak across time.

## Data-quality fixes

| Issue | Rows | Handling |
|---|---|---|
| Rate below $1.00 or above $6.00 per mile | 599 | dropped from training only — these are label errors; the loads still get predictions |
| Negative weight (sign flip, min −47,500 lb) | 292 | absolute value |
| Missing weight | 300 train / 165 val | median from training, reused unchanged |
| Missing market_index | 374 train / 249 val | median from training, reused unchanged |
| Duplicate route+date+equipment (different rate) | 495 | kept — separate loads, not duplicates |
| No city coordinates in december_chart_inputs.csv | 31 | 64-city lookup rebuilt from training data |

## Two columns deliberately excluded

`market_index` and `quote_signal` are not used. Both are level-shifted between
the development window and the prediction window, so a tree model that learns
their relationship extrapolates a level the market is not paying. Adding them
lowers holdout quality sharply:

| Feature set | MAE | MAPE |
|---|---|---|
| structural (chosen) | $53.96 | 2.13% |
| + market_index | $95.48 | 3.81% |
| + quote_signal | $63.88 | 2.60% |

Full ablation in `outputs/ablation.csv`, written by `ablation.py`.

## Rolling-origin CV

| Train up to | Validated month | Best iter | MAE | RMSE | MAPE |
|---|---|---|---|---|---|
| 2025-08-01 | Aug 2025 | 486 | $53.56 | $204 | 2.23% |
| 2025-09-01 | Sep 2025 | 553 | $56.38 | $215 | 2.18% |
| 2025-10-01 | Oct 2025 | 691 | $51.93 | $278 | 1.99% |

Mean MAE $53.96, mean MAPE 2.13%.

## Report and walkthrough

- `report/Freight_Rate_ML_Report.pdf` — approach, validation design, ablation,
  data-quality notes, December chart, limitations.
- Loom walkthrough: <ADD LINK>
