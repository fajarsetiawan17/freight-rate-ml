# Loom Recording Brief — Freight Rate ML Assessment

Working document for one take. Target runtime **2 minutes 30 seconds**. Camera must be on (face visible, bottom-right corner). Main screen: either compact slides or straight to code — never both.

## 1. Setup before pressing Record

- Close unused Chrome tabs. Keep three: `github.com/fajarsetiawan17/freight-rate-ml`, `report/Freight_Rate_ML_Report.pdf` at page 2, and `pipeline.py` in the editor.
- Editor: 16 pt font, 150 percent zoom, terminal panel at the bottom. Code must stay readable at 720p.
- Terminal already sits in the repo folder. Run it once beforehand so output is warm and there is no build wait on camera.
- Microphone: 5-second test. Normal speaking level, no hiss. Room without a fan.
- Light in front of your face, not behind. No window behind your head.
- Loom: 1080p, camera on, microphone on, **never** record the tab only.
- Windows notifications: turn on Focus Assist for the duration.

## 2. Time map

| Segment | Duration | On screen |
|---|---|---|
| Opening | 0:00–0:12 | Camera, then repo title |
| Data exploration findings | 0:12–0:45 | Report page 2 (scatter + histogram) |
| Data-quality issues | 0:45–1:15 | Data-quality table in README |
| Model reasoning | 1:15–1:45 | Report page 4 (ablation) |
| Training/validation split | 1:45–2:10 | Terminal: `python pipeline.py --cv` |
| Code walkthrough | 2:10–2:30 | `pipeline.py` + green `score.py` |

If you run long, cut from the code walkthrough — it compresses easiest. Never cut the data-quality or split segments.

## 3. Script per segment

### 0:00–0:12 — Opening
On screen: camera, then share to the repo.

> "Hi, I'm Fajar Setiawan. This is my submission for the Machine Learning Engineer assessment on freight rate prediction. I'll cover four things: what I found in the data, the data-quality issues, why I chose this model, and how I split and validated."

Don't read your CV. Don't apologise. Twelve seconds, no more.

### 0:12–0:45 — Data exploration findings
On screen: `Freight_Rate_ML_Report.pdf` page 2.

> "The biggest finding from exploration: rate is almost a pure function of distance. Correlation of log rate against log distance is 0.99. So the model doesn't need to rediscover the distance-price relationship — it only has to learn level corrections on top of it. What's left is the effect of equipment, lane, weight, and ship day.
>
> The rate distribution is heavily right-skewed, so I model the log of rate rather than raw rate."

Point at the left-hand scatter plot when you say 0.99.

### 0:45–1:15 — Data-quality issues
On screen: the "Data-quality fixes" table in the repo README.

> "There are six issues I found and handled.
>
> First, 599 rows have impossible rates — below one dollar or above six dollars per mile. Those are label errors, not prices. I drop them from training but still produce predictions for those loads.
>
> Second, 292 rows have negative weight, down to minus 47,500 pounds. That's a sign flip, so I take the absolute value.
>
> Third, weight is missing on 300 training rows and 165 validation rows. I fill it with the training median, and reuse that same median value for validation and December so nothing leaks.
>
> Fourth, `market_index` is missing on 374 rows and I treat it the same way.
>
> Fifth, 495 rows share the same route, date, and equipment but carry different rates. Those are separate loads, not duplicates. I keep them.
>
> Sixth, the December file ships no city coordinates at all. I rebuild a 64-city lookup from the training data."

If time is tight, skip the fifth point. The first two are mandatory.

### 1:15–1:45 — Model reasoning
On screen: ablation table on report page 4.

> "I use LightGBM with an L1 objective on log rate. Three reasons: L1 is robust to outliers and approximates median optimisation, which fits a long-tailed target; gradient boosting handles the distance-equipment-lane interactions without me hand-writing them; and the dataset is tabular, so heavier models buy nothing.
>
> Ten features: log distance, log weight, equipment, origin, destination, midpoint latitude, midpoint longitude, north-south delta, east-west delta, and day of week.
>
> Two supplied columns are deliberately excluded: `market_index` and `quote_signal`. Both are level-shifted between the development window and the prediction window. Include them and the model extrapolates a level the market isn't paying. The ablation proves it: MAE rises from 53.96 dollars to 95.48 dollars just by adding `market_index`. Dropping it isn't an oversight — it's a measured decision."

Say 53.96 and 95.48 exactly. This is the strongest evidence in the whole recording.

### 1:45–2:10 — Training and validation split
On screen: terminal, run `python pipeline.py --cv`.

> "Validation is rolling-origin with three expanding-window folds. Each fold trains up to the end of one month, then predicts the next month, which it has never seen. I don't use random K-fold because the real submission is a two-month forward forecast — a random split leaks across time and makes the score look better than it is.
>
> Results: MAE 53.56, 56.38, and 51.93 dollars, MAPE about 2.1 percent. Bias flips sign across folds, from plus 23 to minus 27 dollars, which says there's no systematic drift the model is missing.
>
> Those are the numbers I report — not the score on training data."

When the terminal finishes, point at the three fold rows and the MAE column.

### 2:10–2:30 — Code walkthrough
On screen: `pipeline.py`.

> "Everything runs through one entry point, `pipeline.py`, 171 lines. The main flow: `clean` handles the data-quality issues I just described, `features` builds the ten features, `drop_bad_labels` removes the corrupted labels, `fit` trains on log rate, `predict` returns the rate in dollars, `rolling_cv` produces the table you just saw, and `metrics` computes MAE, RMSE, MAPE, and bias.
>
> The important detail: `clean` returns its imputation values, and those get reused on validation and December. No test-set statistic ever leaks back into training.
>
> Finally, the assessment's own `score.py` runs unmodified and it passes. Outputs land in `outputs/`, and the December chart is in `scorer_results`."

Point at `def clean` and the `fills` line, then close on the green `score.py`.

## 4. Phrases to avoid

- "I tried things" or "it happened to work" — replace with a measured decision.
- Any number that isn't in the report. Every figure in this script is already in the document.
- "The model is accurate" without a number. Always MAE or MAPE.
- Apologising up front. Start immediately.
- Reading tables row by row. Name the important ones only.

## 5. If they ask follow-ups

**Why not XGBoost or CatBoost?** Equivalent results on tabular data this size; LightGBM trains fastest and has native categorical handling, so no one-hot needed.

**Why L1 instead of RMSE?** Rate has a long tail. RMSE gets dominated by a handful of expensive loads, and the scorer metric is percentage-based — L1 on logs sits closest to that.

**Why no level calibration?** It was tested. Mean-residual calibration at 30, 45, 60, and 90 days made MAE worse on average, from 56.71 to 74.80 dollars. Rejected on measurement, not for lack of trying.

**What's the baseline?** A constant dollars-per-mile baseline scores 8.19 percent MAPE. The model scores 1.99 percent.

**Why 600 estimators?** That's early stopping across the three CV folds — 486, 553, and 691 — scaled to the full training size.

## 6. Post-recording checklist

- Runtime 2:00–3:00. Under 2 minutes reads rushed, over 3 gets trimmed by the reviewer.
- Face visible for at least 80 percent of the runtime.
- No notifications, no sensitive tabs, no tokens or passwords anywhere on screen.
- Audio clean without clipping. Replay the first 30 seconds to check.
- Loom link set to **anyone with the link can view**, not private.
- Link pasted into the repo README and into the submission body.

## 7. Reference numbers

| Metric | Value |
|---|---|
| Raw training rows | 48,000 |
| Rows used for training | 47,401 |
| Rows dropped (broken labels) | 599 |
| Validation loads | 12,000 |
| Holdout MAE | 51.93 dollars |
| Holdout RMSE | 277.61 dollars |
| Holdout MAPE | 1.99 percent |
| CV MAE, three folds | 53.56 / 56.38 / 51.93 |
| CV MAPE | 2.23 / 2.18 / 1.99 percent |
| Baseline $/mile MAPE | 8.19 percent |
| MAE with `market_index` | 95.48 dollars |
| MAE with `quote_signal` | 63.88 dollars |
| December predictions | 811–831 dollars, mean 822 |
