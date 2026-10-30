# Data Recovery, Information Gap and 0.90 Feasibility

**Companion documents:** `data_provenance_and_acquisition.md` (who holds what), `data_request_draft.md` (ready-to-send request, not sent), `missing_data_specification.md` (field-level specification), `cohort_reconstruction_design.md` (censoring-aware dataset design), `reassessment/formulation_feasibility.csv`, `experiment_registry.csv`.
**Code:** `ML/ml_engineer/reassessment/censoring.py`, `simulate_extraction.py`, `value_of_information.py`. All scores recomputed from saved prediction arrays; R-A1-ASOF re-run reproduced its saved predictions exactly (max probability difference 0.0).
**Not done (by design):** no new model tuning, no holdout scoring.

---

## 1. Formulation feasibility: who is still uncertain at prediction time?

Development evaluation windows (19,341 arrivals). "Open" = still in the yard and elapsed < 65.1 h, so ON_TIME, AT_RISK and CRITICAL are all still possible.

| Formulation | Eligible (in yard) | **Open, all three classes** | CRITICAL share among open | Median lead time (open) | Measured Macro-F1 (open) |
|---|---|---|---|---|---|
| A: discharge | 100% | 100% | 9.4% | 52.6 h | 0.398 |
| B12: +12 h | 94.3% | 94.3% | 10.0% | 43.9 h | 0.479 |
| B24: +24 h | 79.4% | 79.4% | 11.8% | 42.4 h | 0.520 |
| B48: +48 h | 53.5% | 53.5% | 17.5% | 40.5 h | 0.542 |
| C: customs release known | 74.5% | 56.9% | **3.5%** | 23.2 h | 0.455 |
| E: delivery job created | 99.8% | 73.8% | **1.2%** | 11.8 h | 0.477 |
| T: truck arrival (appointment proxy) | 100% | 59.9% | **0.0%** | 0.5 h | not meaningful |

**Finding.** CRITICAL stays reach their customs, delivery-order and pickup events *after* 65 h, so they have mostly left the open population by the time those events happen. Event-driven formulations are not three-class forecasting tasks; their high headline scores (0.68, 0.83) come from stays that elapsed time has already resolved. **A three-class forecast must be made at discharge or an early landmark, using status information about why a container has not moved yet.**

A truck-appointment formulation would behave like T (lead time ≈ booking lead, CRITICAL nearly absent), so it can refine ON_TIME vs AT_RISK but cannot by itself carry Macro-F1. Appointments are valuable as *features* at early landmarks for boxes whose pickup is already booked, not as a separate prediction stage.

---

## 2. Value of information: how good must forecasts of the missing events be?

Method (`value_of_information.py`): at discharge, add the realised event time (relative to discharge) plus Gaussian error σ, emulating a forecaster built from the missing data. Same model, folds and features as R-A1-ASOF. These are **bounds, not results**: a forecaster that only *knows status* (e.g. "declaration filed, red lane") produces errors that are not Gaussian and may be larger.

C = customs release (date precision + 24 h; exact timestamps would do slightly better), J = delivery-job creation, P = pickup (truck arrival).

| Event times known at discharge | σ = 0 h | σ = 6 h | σ = 12 h | σ = 24 h | σ = 48 h |
|---|---|---|---|---|---|
| None (R-A1-ASOF, production model) | 0.398 | | | | |
| C: customs release | 0.710 | 0.701 | 0.689 | 0.669 | 0.574 |
| C + J: + delivery job | 0.827 | 0.819 | 0.799 | 0.761 | 0.674 |
| C + J + P: + pickup | 0.996 | 0.958 | **0.920** | 0.861 | 0.757 |

Per-class F1 (ON / AR / CR): C+J σ=0 → 0.903 / 0.773 / 0.817; C+J+P σ=12 → 0.947 / 0.882 / 0.937.

Landmark checks:

| Landmark + forecasts (σ = 12 h) | Mean-fold Macro-F1 | Pooled 95% CI | F1 ON / AR / CR |
|---|---|---|---|
| B24 + C + J | 0.796 | [0.770, 0.817] | 0.844 / 0.742 / 0.805 |
| B24 + C + J + P | **0.916** | [0.903, 0.925] | 0.926 / 0.884 / 0.940 |
| B48 + C + J | 0.751 | [0.724, 0.774] | 0.661 / 0.787 / 0.804 |

The 24 h landmark with ±12 h forecasts of all three stages is the only configuration whose lower confidence bound clears 0.90. Its open population still contains 11.8% CRITICAL, with 42 h median lead time.

**Reading the table.**
- **Customs information alone cannot reach 0.90**, even if perfect (0.71). Its value is real (+0.31 over the production model) and is robust to forecast errors up to ~24 h.
- **Customs + delivery order cannot reach 0.90 either**, even if perfect (0.83). The remaining confusion is ON_TIME ↔ AT_RISK around the 65 h boundary, which is decided by *when the truck comes* after the delivery job (median 13 h, p90 52 h).
- **0.90 requires all three stages (customs release, delivery order, pickup) to be forecast at prediction time with errors of roughly ±12–15 h each** (0.920 at σ = 12 h, 0.861 at σ = 24 h). The σ = 0 row with pickup is trivial (truck arrival is ~2 h before exit); only the noisy rows are informative.
- These are optimistic: real forecasters will have heavier-tailed errors for exactly the stuck containers that define CRITICAL.

---

## 3. Can the missing data close the gap to 0.90?

**What the evidence says.**

1. **With the current extract:** no. 0.40 at discharge, 0.54 at 48 h, and the information needed is absent.
2. **With customs-declaration and delivery-order/payment data (Tiers 1–3, obtainable for 2021):** the bound is 0.83 at discharge with perfect forecasts and ~0.80 with ±12 h forecasts. Realistic forecasters from status data will land below that. **0.90 is not supported by the evidence for 2021 data**, because pickup timing cannot be forecast without appointment data, and TPS had no appointment system in 2021.
3. **With truck-appointment data (TPS Terminal Booking System, trial from 7 July 2025):** 0.90 becomes **plausible but unproven**. It requires the three event times to be predictable to about ±12–15 h at the prediction time. At discharge that is unlikely for stuck containers. At a 24 h landmark it becomes credible for containers whose declaration is filed, DO requested and pickup booked. Whether the rest (the CRITICAL candidates) are identifiable from their *absence* of those statuses is the decisive empirical question, and only a 2025+ extract can answer it.
4. **Changing the formulation does not close the gap**: event-driven times remove CRITICAL from the open population (§1).

**Therefore:** the credible route to 0.90 is (a) Tier 1–3 data for 2021 to build and validate the status-based landmark model and measure each source's value (target 0.65–0.80), then (b) a 2025+ extract with appointments for the 0.90 attempt. Committing to 0.90 on 2021 data alone is not justified.

---

## 4. Answers to the seven questions

| # | Question | Answer |
|---|---|---|
| 1 | **What data first?** | **Tier 1 from the dataset authors:** the June–November 2021 raw capture re-extracted **by discharge date with unfinished stays**, exact customs time of day, hashed **Importer ID**, `CONTAINER_KEY`, cost attributes, and any omitted activities. This needs no new extraction from the terminal and fixes the cohort, adds ~2 months, and adds the strongest unexploited heterogeneity signal (importer). |
| 2 | **Who can provide it?** | Prof. Riyanarto Sarno (ITS, corresponding author, `riyanarto@if.its.ac.id`) and Dr. Hanung Nindito Prasetyo (ITS / Telkom University). Terminal-held fields (SPPB receipt, SP2, payment): **TPS Surabaya**, with the authors as sponsors. Customs-side PIB/lane/SPJM: **KPPBC TMP Tanjung Perak** (DJBC/CEISA) under an MoU. 2025+ truck appointments: TPS (Terminal Booking System). |
| 3 | **Cheapest route?** | One email to the corresponding author with the tiered request (`data_request_draft.md`) plus a data-use agreement. Do not start with a DJBC PPID request: record-level data are usually refused. |
| 4 | **Highest-value missing fields?** | (1) Declaration filing time and lane (PIB, SPJM): explains why boxes are stuck before release. (2) DO request/release and terminal payment: the customs → delivery-job lag. (3) Exact SPPB receipt time (cheap, removes the ±24 h ambiguity). (4) Importer/broker ID: history features. (5) Truck appointments: only exist for 2025+ periods. Full table in `missing_data_specification.md`. |
| 5 | **Primary formulation?** | **Early landmarks with all three classes open: discharge, +12 h, +24 h**, one model with elapsed time as a feature, evaluated per landmark on the open population. Use 24 h as the operational default. Event-driven predictions become features (status as of t), not separate tasks. |
| 6 | **Experiments once data arrive?** | See §5. |
| 7 | **Evidence of progress toward 0.90?** | See §6. |

---

## 5. Experiments once data are obtained (pre-registered order)

| # | Data needed | Experiment | Success criterion | Stop/reject criterion |
|---|---|---|---|---|
| X1 | Tier 1 extract | Rebuild the cohort with `censoring.label_status`. Check that `exit_window_extract` reproduces the 95,648 published stays. Rerun R-A1/B12/B24/B48 unchanged on the new cohort with folds spanning June–October | Published stays reproduced; scores within ±0.03 of current (no new artefacts) | Published stays not reproducible → source differs; stop and resolve |
| X2 | Tier 1 (Importer ID) | Temporally safe importer/broker history features (past dwell of *completed* stays only) | ≥ +0.03 mean-fold Macro-F1 at A and B24 | < +0.01 |
| X3 | Tier 1 (exact customs time) | Replace date + 24 h availability with exact time | ≥ +0.02 at B12/B24 | — |
| X4 | Tier 2 (SPPB receipt, SP2, payment) | Status-as-of-t features at landmarks | B24 ≥ 0.65 | < 0.58 → terminal billing data do not carry the DO signal |
| X5 | Tier 3 (PIB filing, lane, SPJM) | (a) forecast hours-to-release from PIB status; (b) feed forecast + status into the landmark model | Release forecast error small enough to reach the σ needed in §2; B24 ≥ 0.75 | Forecast error ≥ the σ where §2 falls below 0.75 |
| X6 | 2025+ extract with appointments | Add appointment status at landmarks | B24 ≥ 0.85 on open population | — |
| X7 | All of the above | Two-stage process model (customs → delivery job → pickup) with discrete-time hazard, single confirmatory run on never-seen months | Macro-F1 ≥ 0.90 (see §6) | — |

Model-family work (CatBoost, ordinal/hazard, prior-shift adaptation) runs only as part of X2–X7, never as a stand-alone tuning exercise.

---

## 6. Evidence that would demonstrate genuine progress toward 0.90

1. **Stage gates on the development folds, each on the open three-class population:** ≥ 0.55 (current data) → ≥ 0.65 (terminal billing/DO) → ≥ 0.75 (customs declaration status) → ≥ 0.85 (pickup information). Each gate is passed only with the 95% vessel-call bootstrap lower bound above the previous gate.
2. **Value attributable to each data source**, measured by ablation on the same folds.
3. **A single confirmatory evaluation** on months that were never used for any decision. With the Tier 1 extract that means October 2021 arrivals, using a 161 h follow-up from the 9 November cutoff. Report the prediction timestamp, eligible population and censoring rule in advance; include every open stay; give all metrics from one saved array.
4. **Reproduction on a second period** (the 2015 TPS extract, re-extracted by discharge date, or a 2025+ extract) to show the gain is not specific to the July 2021 congestion.
5. A 0.90 claim is valid only for the open population at the declared prediction time. Scores that include stays already resolved by elapsed time are reported separately and never used as the headline.
