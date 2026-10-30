# DS2 Complete Project Audit (Stage 03 Forensic Reassessment)

**Scope:** implementation, dataset, target, features, leakage, models, evaluation protocol and historical results of `AbdulRahman1807/DS2` as of commit `3147f47`.
**Method:** every claim below was either re-executed from code in this repository or is explicitly marked as *not reproducible*. New code lives in `ML/ml_engineer/reassessment/`; every new score is recomputed from a saved prediction file under `reports/ml/reassessment/predictions/`.
**Companion documents:** `dataset_suitability_assessment.md`, `predictability_and_ceiling_analysis.md`, `0.90_macro_f1_strategy.md`, `final_recommendation.md`, `experiment_registry.csv`.

---

## 1. Executive findings

| # | Finding | Severity | Evidence |
|---|---|---|---|
| F1 | **The dataset is selected on the outcome.** The source extract contains only stays whose `TRUCK_OUT` lies in 2021-07-01 → 2021-09-30. June discharges are present only if they dwelled long (left truncation); September discharges are present only if they left quickly (right censoring). | Critical | Raw `TRUCK_OUT` end range is exactly 2021-07-01 00:07 → 2021-09-30 23:51 while discharges start 2021-05-12. All 169 stays discharged before 2021-06-21 are `CRITICAL_DELAY`. |
| F2 | Stage 01 partitions inherit both biases. `train.parquet` contains 3,622 June discharges (21.7% CRITICAL, by construction) and the thresholds p50/p85 were fitted on this biased sample; `test.parquet` is missing an estimated ~53% of its CRITICAL stays. | Critical | §4.2, censoring estimate in §4.4. |
| F3 | **The legacy baseline is reproducible**: EXP-BASE-01 = 0.3981 and EXP-BASE-01-OPT-MACRO = 0.4028 Macro-F1 on `val.parquet`, with confusion matrices identical to the registry. | Info | `R-BASE-01`, `R-BASE-01-MULT` (reproduced from code). |
| F4 | Under a selection-aware cohort with embargoed rolling-origin folds, the **same 13 legacy features score 0.318 mean-fold Macro-F1** (pooled 0.358). Part of the historical 0.40 came from features that encode the extraction window (`discharge_month`, trailing departure statistics). | High | `R-A0-LEGACY13-COHORT`. |
| F5 | `final_model_recommendation.md` reports a confusion matrix that **does not belong to the 0.4028 prediction array**. Its matrix implies accuracy 0.5608 / Macro-F1 0.3985 / weighted-F1 0.5764, while the text reports 0.5785 / 0.4028 / 0.5555. The registry matrix (reproduced here) is the correct one. Train/val/test date ranges in that report are also wrong (val ends 2021-09-11/12, not 09-17). | High | §6.2. |
| F6 | The 0.4028 headline used decision multipliers **selected on the same validation set** on which it is reported. It is an in-sample-tuned validation score, not an independent estimate. | High | `emergency_experiment_results.md`, registry rows `*-OPT-*`. |
| F7 | **The two most recent commits contain no code.** The 41-row registry, the ceiling experiments (shuffled CV 0.5738, "train memorisation" 0.9935, "leaky upper bound" 0.4624, "Bayes group-mode oracle") and the label-collision statistics cannot be re-executed. | High | `git show --stat 6bf197f 3147f47` (reports/CSVs only). |
| F8 | Several quantitative claims in `predictability_investigation.md` / `performance_ceiling_assessment.md` **do not reproduce**: OLS on the legacy numeric features gives R² = 0.073 (dwell) / 0.148 (log-dwell), not 0.0142; the "empirical example from raw terminal data" (MSCU, 21.4 t, 22.4 h vs 194.2 h) matches no record in the dataset. "Mathematical proof" / "information-theoretically impossible" language is not supported by the analyses performed. | High | §6.3. |
| F9 | **Customs status at discharge was wrongly excluded as post-discharge information.** CUSTOMS_DEL precedes discharge for 40% of stays by raw stamp, and 15.5% of cohort stays are cleared at least one full day before discharge (conservative). Those stays are 87.2% ON_TIME / 1.4% CRITICAL versus 47.1% / 15.5% for the rest. | High (opportunity) | §5, `dataset_suitability_assessment.md`. |
| F10 | `CUSTOMS_DEL` has date-only precision and 00:00 marks the **start** of the clearance day (59% of JOB_DEL events follow within 24 h, 0.15% precede). Any use of the raw stamp as an availability time leaks up to 24 h. | High (leakage trap) | §5.2. |
| F11 | Legacy `has_quarantine_flag` / `has_red_flag_pib` are computed over the **entire stay** (33% / 46% of flags arrive after discharge) and are therefore unsafe at discharge as implemented. They were correctly excluded from the frozen baseline but are described as "conditional". | Medium | §5.1. |
| F12 | `carrier_prefix` is the first three letters of the ISO 6346 container **owner** code. The most frequent values (TEM, TCL, TRH, SEG) look like leasing-company prefixes, so this field is probably not the shipping line (to be confirmed against the BIC registry). | Medium | §5.1. |
| F13 | All event *start* timestamps carry synthetic-looking jitter: start = end − U(17.6, 90.9) min for almost every activity. The often-cited "TRUCK_OUT precedes UNSTACK in 63% of stays" is very likely this artifact, not pre-logging. Timing resolution below ~1.5 h should not be trusted. | Medium | §4.5. |
| F14 | Stage 02 documents (landmark design, evaluation protocol) were written but **never implemented**. | Info | No landmark code existed before this reassessment. |

---

## 2. What is actually implemented (code path)

```
Dataset of Dwelling Time Process Fix.xlsx (952,069 events, 12 columns, sha256 fd4ddc8e…)
  └─ ML/data_engineer/ingest_mendeley.py   → data/raw/mendeley_event_log.parquet
       └─ cleaner.py   strip names, container_id = case_id[:11], carrier_prefix = case_id[:3],
                       call_date = case_id[-10:]; quarantine stays without TRUCK_OUT (1 stay);
                       pivot_case_attributes: first non-null CTR_SIZE/TYPE (BAPLIE row),
                       GROSS/YARD_BLOCK/YARD_SLOT (DISCHARGE row), JOB_DEL_DOCTYPE (JOB_DEL row),
                       flags = "activity ever occurred in stay"
       └─ dwell_clock.py  dwell_hours = TRUCK_OUT.end − DISCHARGE.start;
                          thresholds = percentile(train dwell, 50 / 85)
       └─ build_pipeline.py  train dates = first 70% of call dates ≥ 2021-06-10 → fit thresholds
       └─ rolling_metrics.py daily departures by TRUCK_OUT date; rolling 7D closed='left';
                              joined on call_date (vessel call date, not discharge timestamp)
       └─ splitter.py  70/15/15 of unique call dates ≥ 2021-06-10; repeated containers removed
                       from val/test into repeated_stays.parquet
  └─ ML/ml_engineer/baseline_pipeline.py  13 features, median-impute + scale, TargetEncoder(cv=5)
       for categoricals, XGBoost/LightGBM/LogReg, train.parquet → val.parquet
  └─ ML/ml_engineer/classifier.py, train.py, feature_importance.py (wrappers)
  └─ ML/evaluation_engineer/*, ML/integration_engineer/*  (stubs: 9–14 lines each)
```

**Executable code that produced results:** data pipeline, `baseline_pipeline.py` (→ `baseline_validation.json`), anomaly detection. **Results with no generating code:** `emergency_experiment_results.*`, `experiment_registry.csv` rows other than those reproducible through `baseline_pipeline.py`, every `EXP-CEIL-*` row, the label-collision and entropy analyses, the depth-sweep table. Model artifacts (`models/*.joblib`) exist but the code that created `best_macro_f1_model.joblib` / `lightgbm_fe_model.joblib` is not in the repository. Evaluation-engineer and integration-engineer modules are stubs.

**Tests:** 59 legacy tests pass. They check contracts, fold ordering and shapes; none detects selection bias, availability-time leakage of date-precision milestones, or report/metric consistency. 11 new tests (`tests/test_reassessment.py`) add a future-event perturbation test for every as-of feature, embargo and holdout isolation checks, and metric consistency checks.

---

## 3. Target verification

| Item | Documentation | Code | Data (verified) |
|---|---|---|---|
| Dwell definition | TRUCK_OUT end − DISCHARGE start | `dwell_clock.calculate_dwell_durations` (identical) | min 1.51 h, median 54.98 h, max 2,592 h; 0 negative |
| ON_TIME cut | ≤ 65.1102 h | p50 of train dwell | recomputed p50 of `train.parquet` = 65.11017 h ✔ |
| CRITICAL cut | > 160.8993 h | p85 of train dwell | recomputed p85 = 160.89930 h ✔ |
| Provenance | "operational" | percentile of a **selection-biased** training window | Unbiased cohort p50/p85 = 60.19 / 150.91 h; all-stays p50/p85 = 54.98 / 140.28 h |

The thresholds are correctly implemented but are statistical artefacts of a biased window, not operational SLAs. They are retained unchanged for every Stage 03 experiment, because the task definition must not move to inflate scores. Label noise from timestamp jitter is small: 1.99% of cohort stays lie within ±1.5 h of 65.11 h and 0.49% within ±1.5 h of 160.90 h.

---

## 4. Dataset and partition audit

### 4.1 Unit of analysis
One row = one container stay (`case_id` = container + vessel call date), 95,648 stays (95,647 with gate-out), 95,567 containers (81 containers with more than one stay). 235 vessel calls, uniquely identified by the berthing (`VESSEL_ATB`) end timestamp. Single terminal (documented as Surabaya, Indonesia), 2021.

### 4.2 Selection on the outcome
Weekly class composition by discharge week (original thresholds):

| Discharge week | Stays | CRITICAL | ON_TIME | Note |
|---|---|---|---|---|
| ≤ 2021-06-20 | 169 | 100% | 0% | only stays still in yard on 1 July |
| 06-21 → 06-27 | 695 | 38.0% | 0.0% | left-truncated |
| 06-28 → 07-04 | 6,010 | 12.3% | 49.9% | partially left-truncated |
| 07-12 → 07-18 | 8,053 | 27.9% | 21.2% | **genuine congestion episode** |
| 07-19 → 07-25 | 8,036 | 22.3% | 33.1% | genuine congestion episode |
| 08-02 → 08-29 | 31,590 | 7.0–11.0% | 52–69% | |
| 09-13 → 09-19 | 6,717 | 4.4% | 65.1% | right-censored |
| 09-20 → 09-26 | 5,017 | 1.3% | 71.5% | right-censored |
| 09-27 → 09-30 | 2,089 | 0.0% | 97.8% | right-censored |

The July congestion is real (it persists inside the clean cohort), but its magnitude in `train.parquet` and the September "recovery" in `val`/`test` are partly sampling artefacts.

### 4.3 Selection-aware cohort (new)
Discharges in **[2021-07-01, 2021-09-06)**. Every such stay gates out after the extract start (no left truncation), and each has ≥ 600 h of follow-up, so at most ~2.3% of CRITICAL stays (dwell > 600 h) can be missing. 69,453 stays; class mix 53.3% / 33.4% / 13.3%.

### 4.4 Censoring of Stage 01 partitions (estimated from fully followed July discharges)
| Partition | Stays | Expected stays missing from extract | Expected share of CRITICAL missing |
|---|---|---|---|
| `val.parquet` window (08-27 → 09-12) | 20,869 | ~0.5% | ~2% |
| `test.parquet` window (09-12 → 09-30) | 14,583 | ~19.6% | **~53%** |

`test.parquet` cannot measure CRITICAL performance and should stay quarantined permanently for that purpose.

### 4.5 Timestamp quality
- Start timestamps: start = end − U(17.6 min, 90.9 min) for all activities, with millisecond precision. End timestamps are the authoritative event times.
- `CUSTOMS_DEL` end is 00:00 for 100% of rows (date only); `SP_RED_FLAG_PIB` for 5.4%.
- 6 `CUSTOMS_DEL` dates (e.g. 1954-07-14) and 1 `SP_RED_FLAG_PIB` date (2021-01-17) are implausible; masked in the new case table.
- 0.46% of stays gate out before the customs date: process exceptions or recording error, retained.

### 4.6 Split mechanics (legacy)
- Split on `call_date`, not on discharge timestamp; 113 train stays are discharged after the first validation discharge.
- No label embargo: training stays discharged up to 2021-08-27 have labels that resolve up to mid-September, i.e. after validation begins. This leaks congestion-regime information.
- Container disjointness is enforced (correct, but there are only 81 repeat containers).

---

## 5. Feature availability matrix

Availability = earliest time the terminal could know the value. "At discharge" = `DISCHARGE.start`.

| Feature | Source | Availability timestamp | Safe at prediction time? | Leakage risk | Recommendation |
|---|---|---|---|---|---|
| `carrier_prefix` | case_id[:3] (ISO 6346 owner code) | before arrival | Yes | Low; mis-named, owner/lessor ≠ line; 390 levels → overfitting | Keep as `owner_prefix`, frequency-cap rare levels |
| `CTR_SIZE`, `CTR_TYPE` | BAPLIE row | BAPLIE end (median 28 h pre-discharge) | Yes | Low | Keep |
| `GROSS` | DISCHARGE row | discharge | Yes | Low | Keep |
| `YARD_BLOCK` | DISCHARGE row | discharge | Yes (provisional) | Low-medium: quarantine blocks Z/Y/YY/W1 are assigned only when the quarantine flag is known at discharge (8,058 of 8,144); 82 post-discharge exceptions arrive median 0.6 h later | Keep; ablate in confirmatory run |
| `YARD_SLOT` | DISCHARGE row | discharge | Yes (provisional) | Low | Keep |
| `discharge_hour`, `discharge_day_of_week`, `is_weekend` | DISCHARGE start (jittered ≤ 1.5 h) | discharge | Yes | Low | Keep |
| `discharge_month` | DISCHARGE start | discharge | Available, but **encodes the extraction window** (June = long stays only) and never sees future months in training | **High (artefact)** | **Remove** |
| `trailing_7d_volume/critical/at_risk/mean_dwell`, `historical_delay_rate_7d` | departures by TRUCK_OUT date, 7D window closed-left, joined on `call_date` | prior day | Timing causal | **Medium**: window before 07-08 truncated by the extract; joined by vessel call date not discharge time; departures of early July are long June stays | Replace with exact-time as-of terminal state with masked incomplete windows (done) |
| `weather_severity_index`, `has_active_disruption` | PortWatch/GDACS | date | n/a | none; constant 0 | Drop |
| `has_quarantine_flag` (legacy) | any HAS_QUARANTINE_FLAG event in stay | 67% pre-discharge, 33% post | **No (as implemented)** | High | Replace with `known_HAS_QUARANTINE_FLAG` at t |
| `has_red_flag_pib` (legacy) | any SP_RED_FLAG_PIB in stay | 54% pre-discharge | **No (as implemented)** | High | Replace with as-of version |
| `has_behandle_inspection` | any BEHANDLE event | median +29 h (job) / +41 h (stack) | No at discharge | High | As-of only (landmarks) |
| `JOB_DEL_DOCTYPE` | JOB_DEL row | JOB_DEL end (median +32 h) | No at discharge | High | As-of only (`doc_type_known`) |
| `total_events_in_stay` | count of all events | gate-out | **Never** | Critical (target proxy) | Forbidden |
| `dwell_*`, `truck_out_end_ts`, `target_*` | target | — | Never | Target | Forbidden |
| **New** `known_CUSTOMS_DEL`, `age_CUSTOMS_DEL`, `customs_rel_discharge_h` | CUSTOMS_DEL date | **date + 24 h** | Yes with this rule | High if raw stamp used | Use only with +24 h rule (tested) |
| **New** `known_/age_` of quarantine, red flag, behandle, JOB_DEL, TRUCK_IN … | event end (+24 h for midnight-stamped red flags) | event end | Yes (as-of) | Low (perturbation-tested) | Use |
| **New** vessel-call shares (cleared / flagged / released / departed by t), call size, discharge rank | other stays of the same call | as-of t | Yes | Low (perturbation-tested) | Use |
| **New** yard inventory, uncleared/unreleased inventory, 24 h / 7 d departures and their dwell and class mix, 7 d clearance lag | all stays | as-of t; windows masked when they reach before 2021-07-01 | Yes | Low (perturbation-tested); slight undercount for stays leaving after 09-30 | Use |

**Target encoding:** legacy `TargetEncoder(cv=5)` uses shuffled internal folds. A training row's own target is excluded and the validation transform uses training statistics only, so this is not test leakage; within training it mixes future rows into past encodings (minor). The new pipeline uses LightGBM native categoricals with a training-only category list and rare levels mapped to OTHER.

---

## 6. Historical results verification

### 6.1 Reproduced
| Claim | Reproduced? | Value |
|---|---|---|
| EXP-BASE-01 (XGB balanced, 13 features) | ✔ exact | Macro-F1 0.3981, acc 0.5430, bal-acc 0.4117, crit F1 0.1049, crit PR-AUC 0.0909 |
| EXP-BASE-01-OPT-MACRO (×[1.4,1.2,1.3]) | ✔ exact, **but multipliers chosen on the same val set** | Macro-F1 0.4028, acc 0.5785 |
| Thresholds 65.1102 / 160.8993 | ✔ | — |

### 6.2 Inconsistent
`final_model_recommendation.md` §4 prints confusion matrix `[[9090,3692,1110],[2367,2448,634],[643,725,170]]`. That matrix gives accuracy 0.5608, Macro-F1 0.3985, F1 = (0.6994, 0.3976, 0.0985). The reproduced 0.4028 array has matrix `[[9698,3148,1046],[2727,2210,512],[861,507,170]]` (accuracy 0.5785). The per-class table in that report is a mixture of both arrays. The report should be read as superseded.

### 6.3 Not reproducible / not supported
| Claim | Status |
|---|---|
| 41-experiment registry (MOD/FE/IMB/CEIL rows) | No generating code. Rows are kept in the consolidated registry and flagged `code_available = no`. |
| "OLS R² = 1.42% with all 13 features" | Re-fit gives R² 0.073 (dwell) and 0.148 (log dwell) on train. |
| "71.93% of identical-feature groups contain multiple classes … Bayes error is non-zero" | No code. Even if true, label collisions within coarse groups do not bound out-of-sample Macro-F1. It is a heuristic, not a proof. |
| "Shuffled CV ceiling 0.5738" | No code; shuffled CV is also not a ceiling for temporal generalisation, only a different (leakier) estimate. |
| "Leaky retrospective upper bound 0.4624" | No code; the claim that post-discharge milestones only reach 0.46 is contradicted by `R-ORACLE-*` and the landmark results in `predictability_and_ceiling_analysis.md`. |
| MSCU example | No matching record in the data. |

---

## 7. Stage 03 protocol (pre-registered before any new model was fitted)

- **Cohort:** discharges in [2021-07-01, 2021-09-06).
- **Development folds** (train on cohort discharges before `train_end`; evaluate on `[eval_start, eval_end)`; 7-day label embargo; training rows also need prediction time < `eval_start`):
  F1 train < 07-19 → eval 07-26…08-01 · F2 train < 07-29 → eval 08-05…08-11 · F3 train < 08-08 → eval 08-15…08-21.
- **Holdout TEST:** train < 08-20 → eval 08-27…09-05. **Not scored in this round.** This window overlaps the legacy `val.parquet` window, which earlier stages used to select among 13-feature models; it has never been used for any Stage 03 feature, formulation or hyperparameter decision.
- **Legacy `test.parquet`:** untouched, and unusable for CRITICAL metrics (§4.4).
- **Fixed model:** LightGBM, 500 trees, lr 0.03, 31 leaves, min_child_samples 100, class_weight balanced, seed 42, argmax decision. No hyperparameter or threshold was tuned on any evaluation fold in this round.
- **Reporting:** mean-of-folds Macro-F1 (primary), pooled Macro-F1 with a 95% vessel-call cluster-bootstrap interval, full per-class report and confusion matrix from the same saved array.

Results are in `predictability_and_ceiling_analysis.md` and `experiment_registry.csv`.
