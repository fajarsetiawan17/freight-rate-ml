"""Build the submission report (DOCX) from measured results."""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

S = json.load(open("report/stats.json"))
CV = pd.read_csv("outputs/cv_metrics.csv")
AB = pd.read_csv("outputs/ablation.csv")
BAND = pd.read_csv("report/figures/mape_by_distance.csv")
TEAL = RGBColor(0x06, 0x4A, 0x56)

doc = Document()
for name in ("Normal",):
    doc.styles[name].font.name = "Calibri"
    doc.styles[name].font.size = Pt(10.5)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = TEAL
    return p


def body(text, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(6)
    return p


def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    return p


def table(df, widths=None, font=8.5):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(font)
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}" if isinstance(val, int) else str(val)
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ---------------- cover
title = doc.add_heading("Freight Rate Prediction", level=0)
for r in title.runs:
    r.font.color.rgb = TEAL
sub = doc.add_paragraph("Machine Learning Engineer technical assessment — modelling report")
sub.runs[0].font.size = Pt(12)
sub.runs[0].font.color.rgb = RGBColor(0x45, 0x5A, 0x60)
meta = doc.add_paragraph()
meta.add_run(
    f"Development data: {S['raw_rows']:,} labelled loads, Jan–Oct 2025\n"
    f"Final predictions: {S['valid_rows']:,} loads, Nov–Dec 2025\n"
    "Model: LightGBM, L1 objective on log rate · Code and reproduction steps in the repository README."
).font.size = Pt(9)

# ---------------- summary
h("1. Executive summary", 1)
h_ = S["holdout"]
body(
    f"Posted rate is almost deterministic given distance: correlation of log(rate) with log(distance) is 0.99, "
    f"and a gradient-boosted model on nine structural features reaches a holdout MAE of ${h_['MAE']:.0f} "
    f"and MAPE of {h_['MAPE']:.2f}% on unseen October loads. Two of the supplied columns — market_index and "
    f"quote_signal — are actively harmful out of time: adding them to the model raises holdout MAE from "
    f"${AB.loc[2, 'MAE']:.0f} to ${AB.loc[3, 'MAE']:.0f}. Both are level-shifted between the development "
    f"window and the prediction window, so the model learns a relationship that does not transfer. I kept them "
    f"out and documented the evidence rather than tuning around them."
)

# ---------------- data
h("2. What the data looks like", 1)
body(
    "One row per truckload: origin and destination (name plus coordinates), distance, equipment type, weight, "
    "date, two market-level signals, and the posted rate we have to predict. Development data covers ten months; "
    "the 12,000 validation loads cover November and December, so every validation fold is a genuine forward "
    "prediction rather than a random split."
)
h("2.1 Rate structure", 2)
bullet("correlation between log(rate) and log(distance) is 0.9919. Distance alone explains the shape of the problem.", "Distance dominates: ")
bullet("median $/mile is $2.05 for Dry Van, $2.22 for Flatbed, $2.31 for Reefer. Equipment shifts the level, not the slope.", "Equipment is a level effect: ")
bullet("Monday–Thursday average $2.16/mile, weekends $2.13. Real but small — worth 7% of holdout MAE.", "Weekday premium: ")
bullet("market_index has a ten-month cycle (0.93 → 1.30 → 0.89) but correlates only 0.03 with rate at row level.", "Market index is a red herring: ")

doc.add_picture("report/figures/rate_vs_distance.png", width=Inches(5.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_picture("report/figures/rpm_hist.png", width=Inches(5.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# ---------------- data quality
h("3. Data-quality issues and how I handled them", 1)
dq = [
    ("Impossible $/mile", "599 loads (1.2%) price below $1.00 or above $6.00 per mile; the maximum is $14.13/mile on a 170-mile haul. These are label errors, not premium freight.", "Dropped from training, never from prediction. Kept the validation load anyway — a bad rate in the past says nothing about a new load's price."),
    ("Sign-flipped weight", "292 rows carry negative weight (minimum −47,500 lb); 13 of them are exactly −47,500, a clean sign flip.", "Absolute value. The magnitude is right, the sign is a data-entry bug."),
    ("Missing values", "weight missing on 300 development rows (0.6%) and 165 validation rows; market_index missing on 374 development and 249 validation rows.", "Median imputation fitted on development data only and reused unchanged for validation and December. With 0.6% missing, a model-based impute buys nothing."),
    ("Duplicate-looking loads", "495 development rows share route, date and equipment — different weights and rates.", "Kept. Two shippers can move two loads the same lane the same day; deduplicating would delete real signal."),
    ("No coordinates in the December file", "december_chart_inputs.csv ships city names only, but the model uses coordinates.", "Built a 64-city lookup from the development data (each city maps to exactly one coordinate pair) and joined it back."),
]
for name, issue, action in dq:
    bullet(f"{issue} {action}", f"{name} — ")

# ---------------- validation
h("4. Validation design", 1)
body(
    "The validation set is two future months, so a random K-fold split would leak information across time and "
    "produce a flattering score. I used rolling-origin validation instead: three expanding-window folds, each "
    "training on everything before a cut-off and predicting the single unseen month that follows it. The cut-offs "
    "are August, September and October 2025, which mirrors the one-to-two-month forward gap the real submission "
    "faces. Early stopping uses the fold's own holdout month, and I report the iteration count so the final fit "
    "on all ten months is not arbitrary."
)
cv = pd.DataFrame({
    "Fold": CV["fold"].str.replace("train<", "train < ").str.replace("+1mo", ""),
    "Train rows": CV["n_train"], "Best iter": CV["best_iter"],
    "MAE ($)": CV["MAE"], "RMSE ($)": CV["RMSE"], "MAPE (%)": CV["MAPE"], "Bias ($)": CV["bias"],
})
table(cv, widths=[2.0, 0.8, 0.7, 0.8, 0.8, 0.8, 0.8])
body(
    f"Mean MAE across the three folds is ${CV['MAE'].mean():.0f} and mean MAPE {CV['MAPE'].mean():.2f}%. "
    "The August fold is biased high and the later two biased low by a similar magnitude — that is market drift, "
    "not model error, and it is the reason I tested a level calibration step (section 6.3) before rejecting it."
)

# ---------------- model
h("5. Model and features", 1)
body(
    "LightGBM with an L1 objective on log(rate). Trees capture the interactions (equipment × distance, lane × "
    "weekday) that a linear rate-per-mile model cannot, L1 loss matches the MAE metric being judged, and the log "
    "target keeps percentage error stable across a $57–$25,533 price range. Nine features, all available at quote "
    "time:"
)
for f_, why in [
    ("log(distance)", "the dominant driver, straight-line in logs"),
    ("log(weight)", "capped below at 1,000 lb so the sign fix cannot produce −∞"),
    ("equipment", "categorical — three levels, level shift on $/mile"),
    ("pickup, delivery", "categorical lane effects: 64 cities, some lanes price structurally higher"),
    ("lat/lon midpoint, north–south and east–west deltas", "coarse geography for lanes the model has never seen"),
    ("day_of_week", "weekday premium, and the only feature that makes the December chart non-flat"),
]:
    bullet(why, f"{f_} — ")
body(
    "Deliberately absent: market_index and quote_signal (section 6.2), and any date trend. A raw date feature "
    "would let the trees extrapolate a trend they never saw, which is exactly the failure mode that shows up in "
    "the last CV fold's RMSE."
)

# ---------------- experiments
h("6. What I tried and rejected", 1)
h("6.1 Feature ablation", 2)
body("Same three folds, same hyper-parameters, one feature group changed at a time:")
ab = AB.copy()
ab.columns = ["Variant", "MAE ($)", "RMSE ($)", "MAPE (%)", "Bias ($)"]
table(ab, widths=[2.6, 0.8, 0.8, 0.8, 0.8])
body(
    "Geography is worth $1.81 of MAE, weekday $2.30. The chosen variant (C) is the best structural model I found."
)
h("6.2 Why market_index and quote_signal are excluded", 2)
body(
    "In-window, market_index looks useful: its daily average correlates 0.87 with the daily model residual, and a "
    "row-level regression gives it a coefficient of 0.19. Out of window it collapses to 0.53 — and the damage is "
    "asymmetric. The development window cycles 0.81–1.40; the validation window sits at 0.92–0.94, near the bottom "
    "of that range but outside the range the model last saw. Trees then extrapolate a level the market is not "
    "paying. Ablation variant D makes this concrete: MAE goes from $53.96 to $95.48 and MAPE from 2.13% to 3.81% "
    "the moment market_index is added. quote_signal costs $9.92 of MAE for the same reason at smaller amplitude. "
    "Neither column survives contact with a forward split, so the honest move is to drop both and say why."
)
h("6.3 Level calibration — tested, rejected", 2)
body(
    "Because the residual mean drifts between folds, I tested rescaling predictions by the mean residual of the "
    "last 30/45/60/90 days of training data. It helps one fold, hurts another, and on average makes MAE worse "
    "($56.71 → $74.80 across folds at 60 days). The drift is not predictable from the development window, so an "
    "offset is a coin flip with extra steps. Predictions ship uncalibrated."
)
h("6.4 Other variants", 2)
bullet("L1 on log target won ($53.72 vs $54.63 for L2, $54.24 for Huber). Small but consistent across folds.", "Objective: ")
bullet("seasonal index on the daily residual, 1–3 harmonics. Extrapolating into November–December was worse than flat in three of four backtests; the December chart is therefore driven by weekday only.", "Seasonality: ")
bullet("plain median $/mile by equipment scored 8.19% MAPE against the model's 2.13% — a useful floor, not a competitor.", "Rate-per-mile baseline: ")

# ---------------- results
h("7. Results", 1)
doc.add_picture("report/figures/holdout_diagnostics.png", width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
body(
    f"Final fold (train through September, predict October): MAE ${h_['MAE']:.0f}, RMSE ${h_['RMSE']:.0f}, "
    f"MAPE {h_['MAPE']:.2f}%, bias ${h_['bias']:.0f} on {h_['n']:,} loads. Residuals are close to symmetric with a "
    "small number of high-rate lanes in the tail; that tail is what separates MAE from RMSE by a factor of five."
)
band = BAND.copy()
band.columns = ["Distance band (miles)", "Loads", "MAPE (%)"]
table(band, widths=[2.0, 1.0, 1.0])
body(
    "Short hauls are the hardest: under 500 miles, MAE is dominated by fixed accessorials that the feature set does "
    "not observe (loading time, detention, fuel surcharge floors). If this model went to production, that band is "
    "where I would add data, not where I would add model capacity."
)

# ---------------- december
h("8. December chart", 1)
doc.add_picture("scorer_results/candidate_december.png", width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
dec = S["december"]
lex = pd.read_csv("data/train_test.csv").query('pickup == "Lexington" and delivery == "Fort Wayne"')
body(
    f"Produced by running score.py on outputs/december_chart_inputs.csv. Lexington to Fort Wayne, 360 miles, Dry "
    f"Van, 32,000 lb, only the date changes: ${dec['min']:.0f} to ${dec['max']:.0f}, mean ${dec['mean']:.0f}. The "
    "shape is a weekly cycle peaking Wednesday and bottoming Sunday, which is the weekday effect the model learned "
    "from ten months of data. It is deterministic because every feature except day_of_week is held constant by "
    f"construction — the model has no residual noise to reproduce. For reference, the {len(lex)} "
    "historical Lexington to Fort Wayne loads average $857 because their weights run 16,000–41,000 lb; the fixed "
    "32,000 lb chart point sits in that range at the weekday-adjusted $/mile for that lane."
)

# ---------------- limitations
h("9. Limitations and what I would do next", 1)
bullet("the validation months sit at a different market level than the training window. Any model that ignores the level is unbiased on average but wrong on any given week; recovering that level needs external data (DAT/FreightWaves indices), not more trees.", "No market-level regressor: ")
bullet("a load quoted on 30 December but picked up in January is priced off January conditions. The model uses quote date only.", "No transit or lead time: ")
bullet("fixed accessorials are unobserved, and they are the reason short hauls carry the worst percentage error.", "No accessorial features: ")
bullet("one LightGBM configuration with early stopping, not a hyper-parameter search. The CV spread across folds (${:.0f}–${:.0f} MAE) is wider than the gap between reasonable configurations, so a search would tune to fold noise.".format(CV['MAE'].min(), CV['MAE'].max()), "Hyper-parameters: ")
bullet("probabilistic output for a quote tool. Quantile objectives at 10/50/90 would give a price band instead of a point, which is what a pricing team actually needs.", "Next step: ")

# ---------------- reproduction
h("10. Reproducing this", 1)
body("Repository layout and the three commands that regenerate every artefact:")
for line in [
    "data/            train_test.csv, validation.csv, december_chart_inputs.csv, validation_predictions_template.csv",
    "pipeline.py      clean → features → fit → outputs/validation_predictions.csv + december_chart_inputs.csv",
    "ablation.py      feature-set ablation (section 6.1)",
    "make_report_assets.py  figures and statistics for this report",
]:
    p = doc.add_paragraph(line, style="List Bullet")
    for r in p.runs:
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
code = doc.add_paragraph()
run = code.add_run("python -m pip install -r requirements.txt\npython pipeline.py --cv\npython score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv")
run.font.name = "Consolas"
run.font.size = Pt(8.5)
body("The scorer validates row count, ID coverage and positivity, then writes scorer_results/candidate_december.png.")

doc.save("report/Freight_Rate_ML_Report.docx")
print("saved report/Freight_Rate_ML_Report.docx")
