# Predictability and Performance-Ceiling Analysis

**Question:** why does three-class dwell prediction stall near 0.40 Macro-F1, and is 0.90 plausible?
**Protocol for every new number (pre-registered, see `complete_project_audit.md` §7):** selection-aware cohort (discharges 2021-07-01 → 2021-09-05); three rolling-origin development folds with a 7-day label embargo; LightGBM with fixed hyperparameters, balanced class weights, argmax; no tuning on any evaluation fold; holdout TEST not scored. Primary statistic = mean of per-fold Macro-F1; pooled Macro-F1 with a 95% vessel-call cluster-bootstrap interval. Every number is recomputed from `reports/ml/reassessment/predictions/<exp_id>.parquet`.
**Caveat:** these are development-fold estimates. Three folds and ~20 configurations were compared, so the best values carry mild selection optimism. The holdout has not been used.

---

## 1. Results

| Exp | Prediction time / information | Mean-fold Macro-F1 | Pooled [95% CI] | F1 ON / AR / CR | CRIT recall | Valid |
|---|---|---|---|---|---|---|
| R-BASE-01 | legacy split, legacy 13 features (reproduction) | 0.398 | — | 0.666 / 0.424 / 0.105 | 0.114 | legacy protocol |
| R-A0-LEGACY13-COHORT | legacy 13 features, corrected cohort/folds | **0.318** | 0.358 [0.317, 0.388] | 0.519 / 0.385 / 0.169 | — | yes |
| R-A1-ASOF-NOCUSTOMS | discharge; no customs/quarantine status | 0.348 | 0.345 [0.305, 0.378] | 0.425 / 0.433 / 0.176 | — | yes |
| R-A1-ASOF | discharge; as-of customs, quarantine, vessel call, yard | **0.398** | 0.402 [0.357, 0.446] | 0.524 / 0.467 / 0.215 | — | yes |
| R-A1-ORDINAL | same, Frank–Hall ordinal | 0.406 | 0.410 [0.369, 0.453] | 0.539 / 0.476 / 0.215 | — | yes |
| R-A1-REGRESSION | same, log-dwell regression → thresholds | 0.383 | 0.385 [0.349, 0.424] | 0.557 / 0.518 / 0.081 | — | yes |
| R-A1-UNWEIGHTED | same, no class weights | 0.395 | 0.398 | — | — | yes |
| R-A1-INNER-TUNED | same, multipliers tuned on inner training slice | 0.380 | 0.393 | — | — | yes |
| R-B12-ASOF | discharge + 12 h | 0.479 | 0.480 [0.435, 0.514] | 0.666 / 0.501 / 0.274 | — | yes |
| R-B24-ASOF | discharge + 24 h | **0.520** | 0.518 [0.486, 0.547] | 0.696 / 0.519 / 0.340 | 0.536 | yes |
| R-B24-ORDINAL | +24 h ordinal | 0.515 | 0.516 [0.481, 0.552] | 0.679 / 0.556 / 0.314 | — | yes |
| R-B24-INNER-TUNED | +24 h inner-tuned multipliers | 0.514 | 0.519 | — | — | yes |
| R-B48-ASOF | discharge + 48 h | **0.542** | 0.546 [0.517, 0.572] | 0.655 / 0.555 / 0.428 | 0.611 | yes |
| R-B48-INNER-TUNED | +48 h inner-tuned multipliers | 0.526 | 0.551 | — | — | yes |
| R-C-CUSTOMS | when customs release is known (date + 24 h) | 0.683 (all) / **0.455 open subset** | 0.716 [0.671, 0.749] | 0.813 / 0.662 / 0.672 | — | yes, see §3 |
| R-E-JOBDEL | when the delivery job is created | 0.826 (all) / **0.477 open subset** | — | — | — | yes, see §3 |
| R-ORACLE-CUSTOMS | discharge + **exact future** customs date | 0.710 | 0.724 | 0.833 / 0.652 / 0.687 | — | **no** (bound) |
| R-ORACLE-CUSTOMS-JOBDEL | discharge + exact future customs date and delivery-job time | **0.827** | 0.831 | — | — | **no** (bound) |

Full metrics (precision, recall, balanced accuracy, weighted F1, accuracy, CRITICAL PR-AUC, confusion matrix, per fold) are in `reports/ml/reassessment/results.json` and `experiment_registry.csv`.

---

## 2. Which explanation does the evidence support?

| Candidate cause | Verdict | Evidence |
|---|---|---|
| 1. Insufficient predictive features | **Partly.** Stage 01 omitted legitimate at-discharge signal. | Customs/quarantine status adds +0.05 at discharge (0.348 → 0.398). |
| 2. Missing operational information | **Primary cause.** | Exact customs date at discharge → 0.71; + delivery-job time → 0.83. These events are absent at discharge, and their upstream causes (PIB submission, lane, DO/payment, trucking) are absent from the extract entirely. |
| 3. Prediction timestamp | **Major, but slow.** | +0.08 at 12 h, +0.12 at 24 h, +0.14 at 48 h; ON_TIME stops being predictable after 65 h, so landmarks cannot go much later without losing the three-class task. |
| 4. Noisy labels | Minor. | 2.0% / 0.5% of stays within ±1.5 h (timestamp jitter) of the cut-offs. Customs date precision (±12 h) blurs the 65 h boundary in the oracle. |
| 5. Class boundaries | Secondary. | Thresholds are percentiles of a biased window; they do not create the information gap. A different target would shift but not remove it. |
| 6. Temporal distribution shift | **Major at discharge.** | Train CRITICAL prevalence 18–21% vs eval 8–13% across folds. Optimal decision multipliers flip from [2.0, 2.0] to [0.4, 0.4] between folds. The fold covering the post-congestion week is consistently worst. |
| 7. Survivorship and censoring | **Major for the legacy numbers**, removed here. | Legacy features lose 0.08 under the corrected cohort (0.398 → 0.318). Legacy test missing ~53% of CRITICAL stays. |
| 8. Insufficient training data | **No.** | B24 learning curve: 0.515 / 0.521 / 0.535 / 0.535 at 25/50/75/100% of vessel calls. |
| 9. Weak model architecture | **No.** | LightGBM/XGBoost/ordinal/regression differ by ≤ 0.02. Legacy registry: 41 configs spread 0.33–0.40. |
| 10. Validation methodology | **Yes for legacy claims.** | Multipliers tuned and reported on the same val set; no embargo; selection-biased windows; unreproducible ceiling claims. |
| 11. Feature-engineering limits | Partly. | Vessel-call heterogeneity (CRITICAL 2.8%–30.1% across calls) is only weakly captured by as-of shares; importer identity is anonymised away. |
| 12. Inherent unpredictability | **Not established.** | The process is largely determined by observable administrative milestones (R² of log-dwell: 0.62 from customs date, 0.815 with delivery-job time). It is unpredictable *from what is known early*, not intrinsically random. |

---

## 3. Formulation caveats

- **Landmarks (B).** Population = stays still in the yard at t, so fast stays leave the evaluation population as L grows (cohort size 19.3k → 18.2k → 15.4k → 10.3k across the dev folds). Scores at different landmarks describe different populations and must not be pooled. All classes stay open for L < 65.1 h.
- **Event-driven (C, E).** Prediction time depends on the stay. Many stays reach it after 65 h (ON_TIME impossible) or after 161 h (class already determined). The headline Macro-F1 mixes forecasting with bookkeeping. On the subset where all three classes are open, customs-release prediction scores 0.455. The conservative +24 h availability rule also drops stays that leave on the clearance day.
- **Oracles.** They inject future information and are reported only to size the value of data that would forecast those events. They are not ceilings for models using *forecasts* of those events: a forecast cannot beat exact knowledge, so 0.83 is an optimistic bound for a discharge-time model even with B2–B3 data.

---

## 4. Direct answers

| Question | Answer |
|---|---|
| Is 0.90 plausible with the current dataset at discharge? | **No.** Measured 0.40 (CI 0.36–0.45). Even exact knowledge of future customs and delivery-job times gives 0.83. |
| Is 0.90 plausible with additional operational data? | **Only with data on all three process stages:** customs-release drivers (PIB submission/lane), delivery-order/payment, **and** pickup timing (truck appointments). Customs + DO data alone are bounded near 0.83. |
| Is 0.90 plausible with a different formulation? | **Not with the current data while all three classes remain open.** Event-driven formulations reach 0.68–0.826 only by including stays whose class is already determined by elapsed time. Their open-subset scores are 0.455–0.477. |
| Exact changes required | (1) extended, uncensored extract; (2) PIB lifecycle and DO/payment timestamps; (3) truck-appointment data; (4) a landmark/process model that consumes them; (5) a confirmatory evaluation on never-seen months, on the open-class population. |
| Closest defensible alternative now | Landmark prediction at +24 h/+48 h (≈ 0.52–0.54 Macro-F1, CRITICAL recall 0.54–0.61, ~40 h median lead time), pending one holdout confirmation. |

**Assumptions and limits of these estimates:** one terminal, ~9 weeks, one congestion episode; three development folds; LightGBM with fixed settings; customs availability rule = date + 24 h (conservative); start timestamps jittered by the source. Treat "bounds" as empirical under these assumptions, not as theorems.
