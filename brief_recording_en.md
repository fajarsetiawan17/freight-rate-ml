# Loom Recording Brief — Freight Rate ML Assessment

Working document for one take. Target runtime **2 minutes 45 seconds**. Camera must be on (face visible, bottom-right corner).

The demo runs **inside the Colab notebook**, not in a local editor. Every part of the assessment is shown from one tab: exploration, data quality, CV, the scorer, and the shape of both submission files.

Notebook: `report/Freight_Rate_Walkthrough.ipynb` — 22 cells, 11 sections, about 3 minutes for `Run all` on a free CPU.

## 1. Setup before pressing Record

**Mandatory: run `Runtime → Run all` to completion BEFORE recording.** The notebook must already have every output printed and the chart rendered. Otherwise the recording dies waiting 3 minutes.

- Open the notebook through the Colab link in the README. Don't open the raw `.ipynb` from GitHub.
- Wait until every cell shows as finished, the December chart is visible, and the left TOC lists 11 sections.
- Collapse the left panel (TOC) if the screen is narrow — but leave it open if you want the structure visible. Pick one, don't change mid-recording.
- Browser zoom 110–125 percent. Default 14 px cell font reads fine at 1080p; don't shrink it.
- Hide the other Colab left-panel tabs (Files, Secrets) — a personal Drive folder must never appear on screen.
- Close other tabs. Keep two: the Colab notebook and `github.com/fajarsetiawan17/freight-rate-ml`.
- Microphone: 5-second test. Normal speaking level, no hiss. Room without a fan.
- Light in front of your face, not behind. No window behind your head.
- Loom: 1080p, camera on, microphone on, **never** record the tab only.
- Turn on Windows Focus Assist for the duration.

### Which cells to run live, which to point at

| Cell | Treatment | Why |
|---|---|---|
| §1 stats + correlation | **live** | Instant, and the numbers are the evidence |
| Rate per mile (chart) | **live** | Instant, chart appears in place |
| §2 data quality | **live** | Instant, this is the strongest evidence |
| §3 `pipeline.py --cv` | **cached** | 1–2 minutes on Colab — never run live |
| Ablation (reads CSV) | **live** | Instant |
| §4 `score.py` | **cached** | 10–20 seconds, too slow for a 10-second segment |
| December chart | **cached** | Already rendered by the earlier run |
| §5 file shapes | **live** | Instant |

When pointing at cached output, click the cell once so it highlights, then talk. Don't pretend to run it.

## 2. Time map

| Segment | Duration | On screen |
|---|---|---|
| Opening | 0:00–0:12 | Camera, then the Colab tab |
| Setup and repo map | 0:12–0:25 | §0 Setup |
| Exploration findings | 0:25–0:55 | §1 + rate-per-mile chart |
| Data quality | 0:55–1:25 | §2 Data-quality issues |
| Model reasoning + ablation | 1:25–1:50 | §3 heading + ablation table |
| Split and CV | 1:50–2:15 | §3 `--cv` output |
| Scorer and December chart | 2:15–2:40 | §4 + chart |
| Close + submission files | 2:40–2:45 | §5 |

If you run long, cut the close and the setup segment. Never cut the data-quality or split segments.

## 3. Script per segment

### 0:00–0:12 — Opening
On screen: camera, then share to the Colab tab.

> "Hi, I'm Fajar Setiawan. This is my submission for the Machine Learning Engineer assessment on freight rate prediction. Everything I show you runs from a single Colab notebook that calls the actual submitted code."

Twelve seconds. Don't read your CV, don't apologise.

### 0:12–0:25 — Setup and repo map
On screen: §0 Setup, already executed.

> "The notebook clones its own repository, then installs LightGBM, scikit-learn, and matplotlib. Twenty-two cells total, and the whole pipeline runs on a free CPU in three minutes. The data is the data supplied by the assessment — it ships inside the repo."

Point at the `data: [...]` output.

### 0:25–0:55 — Exploration findings
On screen: §1, then the rate-per-mile chart.

> "Biggest finding: rate is almost a pure function of distance. Correlation of log rate against log distance is 0.99 once the impossible price rows are removed — and still 0.967 with every row included. So the model doesn't need to rediscover the distance-price relationship, only learn level corrections on top of it.
>
> The rate distribution is right-skewed with a skewness of 1.9, so I model the log of rate rather than raw rate.
>
> The right-hand chart shows rate per mile: median 2.15 dollars, but with a tail out to 14 dollars. That's not a price, that's a label error — and that leads into the next section."

Point at the 0.9919 figure in the output, then at the histogram.

### 0:55–1:25 — Data quality
On screen: §2 Data-quality issues, run live.

> "Six issues, all counted in this cell.
>
> First, 599 rows have impossible rates — below one dollar or above six dollars per mile. Those are label errors, not prices. I drop them from training, but those loads still get predictions.
>
> Second, 292 rows have negative weight, down to minus 47,500 pounds. That's a sign flip, so I take the absolute value.
>
> Third, weight is missing on 300 training rows and 165 validation rows. I fill it with the training median, and reuse that same median for validation and December so nothing leaks.
>
> Fourth, `market_index` is missing on 374 rows, handled the same way.
>
> Fifth, 495 rows share the same route, date, and equipment but carry different rates. Those are separate loads, not duplicates. I keep them.
>
> Sixth, the December file ships no city coordinates at all — only seven columns. I rebuild a 64-city lookup from the training data."

If time is tight, skip the fifth point. The first two are mandatory.

### 1:25–1:50 — Model reasoning and ablation
On screen: §3 heading, then the ablation table.

> "I use LightGBM with an L1 objective on log rate. Three reasons: L1 is robust to outliers and approximates median optimisation, which fits a long-tailed target; gradient boosting handles the distance-equipment-lane interactions without me hand-writing them; and the data is tabular, so heavier models buy nothing.
>
> Ten features: log distance, log weight, equipment, origin, destination, midpoint latitude, midpoint longitude, north-south delta, east-west delta, and day of week.
>
> Two columns are deliberately excluded: `market_index` and `quote_signal`. Both are level-shifted between the development window and the prediction window. Include them and the model extrapolates a level the market isn't paying. The ablation measures it: MAE rises from 53.96 to 95.48 dollars just by adding `market_index`."

Say 53.96 and 95.48 exactly. This is the strongest evidence in the recording.

### 1:50–2:15 — Split and CV
On screen: the `!python pipeline.py --cv` output (cached).

> "Validation is rolling-origin with three expanding-window folds. Each fold trains up to the end of one month, then predicts the next month, which it has never seen. I don't use random K-fold because the real submission is a two-month forward forecast — a random split leaks across time and makes the score look better than it is.
>
> Results: MAE 53.56, 56.38, and 51.93 dollars, MAPE about 2.1 percent. Bias flips sign across folds, from plus 23 to minus 27 dollars — no systematic drift slips through.
>
> I ran this cell before recording: it takes one to two minutes on Colab. These are the numbers I report, not the score on training data."

When pointing, highlight the MAE column and `best_iter` 486, 553, 691.

### 2:15–2:40 — Scorer and December chart
On screen: §4 output, then the chart.

> "Finally, the assessment's own `score.py` runs unmodified. Output: 12,000 final predictions validated, 31 December predictions validated, chart created.
>
> This chart is the artifact the brief asks for by name. December predictions move between 811 and 831 dollars, mean 822. Mid-week peak, Sunday trough — the weekly shape you'd expect for freight."

Point at the "Validated 12,000 final predictions" line, then the chart.

### 2:40–2:45 — Close
On screen: §5 Submission files.

> "Two files come out: `validation_predictions.csv` with exactly two columns, and the December file with the original seven columns in the original order plus `predicted_rate`. Code, data, report, and this notebook are all in the public repo. Thanks."

Point at `(12000, 2)` and `(31, 7)`.

## 4. Phrases to avoid

- "I tried things" or "it happened to work" — replace with a measured decision.
- Any number that isn't in the notebook or the report. Every figure here is already in one of them.
- "The model is accurate" without a number. Always MAE or MAPE.
- Apologising up front. Start immediately.
- Reading tables row by row. Name the important ones only.
- "Colab is slow" or any comment about the connection. If it's loading, stay quiet and point at the next cell.

## 5. If they ask follow-ups

**Why Colab and not just the repo?** The repo is enough to run it; the notebook just makes the whole flow verifiable in one click with no local install.

**Why not XGBoost or CatBoost?** Equivalent results on tabular data this size; LightGBM trains fastest and has native categorical handling, so no one-hot needed.

**Why L1 instead of RMSE?** Rate has a long tail. RMSE gets dominated by a handful of expensive loads, and the scorer metric is percentage-based — L1 on logs sits closest to that.

**Why no level calibration?** It was tested. Mean-residual calibration at 30, 45, 60, and 90 days made MAE worse, from 56.71 to 74.80 dollars. Rejected on measurement.

**What's the baseline?** A constant dollars-per-mile baseline scores 8.19 percent MAPE. The model scores 1.99 percent.

**Why 600 estimators?** Early stopping across the three folds — 486, 553, and 691 — scaled to the full training size.

**Why doesn't the notebook render the report PDF?** The report needs a Word/LibreOffice step. That's why the PDF ships in the repo instead of being regenerated in Colab.

## 6. Post-recording checklist

- Runtime 2:30–3:00. Under 2 minutes reads rushed, over 3 gets trimmed by the reviewer.
- Face visible for at least 80 percent of the runtime.
- Notebook already `Run all` before recording; no cell still spinning.
- No notifications, no personal Drive tabs, no tokens or passwords anywhere on screen.
- Audio clean without clipping. Replay the first 30 seconds to check.
- Loom link set to **anyone with the link can view**, not private.
- Link pasted into the repo README and into the submission body.

## 7. Reference numbers

| Metric | Value |
|---|---|
| Notebook cells | 22, across 11 sections |
| Raw training rows | 48,000 |
| Rows used for training | 47,401 |
| Rows dropped (broken labels) | 599 |
| Validation loads | 12,000 |
| Correlation log distance–log rate (valid rows) | 0.9919 |
| Correlation log distance–log rate (all rows) | 0.9674 |
| Rate skewness | 1.90 |
| Median dollars per mile | 2.15 |
| Holdout MAE | 51.93 dollars |
| Holdout RMSE | 277.61 dollars |
| Holdout MAPE | 1.99 percent |
| CV MAE, three folds | 53.56 / 56.38 / 51.93 |
| CV MAPE | 2.23 / 2.18 / 1.99 percent |
| Baseline $/mile MAPE | 8.19 percent |
| MAE with `market_index` | 95.48 dollars |
| MAE with `quote_signal` | 63.88 dollars |
| December predictions | 811–831 dollars, mean 822 |
| Submission file shapes | (12000, 2) and (31, 7) |
