# Cohort Reconstruction Design (Arrival-Based, Censoring-Aware)

**Purpose:** specify how a correctly extracted dataset must be built and how each stay enters training and evaluation, so the three-class target stays comparable while removing outcome-dependent sampling.
**Code:** `ML/ml_engineer/reassessment/censoring.py` (labelling rule, Kaplan–Meier class mix, the published exit-window selection), `simulate_extraction.py` (validation), tests in `tests/test_reassessment.py`.

---

## 1. What is wrong with the published extract

The published file is three monthly *gate-out* files (July–September 2021), not an arrival cohort. The 2019 paper by the same research group used this convention explicitly: a stay is evaluated in the month it completes. That is reasonable for monthly KPI reporting but wrong for prediction:

| Effect | Mechanism | Size in the published data |
|---|---|---|
| Left truncation | arrivals before 1 July appear only if they left after 1 July (long stays) | 3,660 stays discharged before 1 July; 100% CRITICAL before 21 June |
| Right censoring by omission | arrivals in late September appear only if they already left (short stays) | ~53% of CRITICAL stays missing for arrivals 12–30 Sept |
| Missing unfinished stays | containers in the yard on 30 Sept are absent entirely | unknown count; by simulation 6–10% of arrivals since 1 July are still in the yard at a given cutoff |

**Simulation (our data, pretending the extract ended at cutoff C):**

| Cutoff C | True CRITICAL share (arrivals 1 Jul → C) | Exit-window design (published) | Completed-only arrivals | Censoring-aware, complete-label window | Recent arrivals (last 161 h): truth vs exit-window/completed-only |
|---|---|---|---|---|---|
| 2021-08-10 | 17.2% | 16.5% | 15.9% | exact (20.0% = truth for that window) | 8.1% vs **0.0%** |
| 2021-08-20 | 15.8% | 15.0% | 14.4% | exact (16.8% = truth) | 9.6% vs **0.0%** |
| 2021-08-30 | 14.2% | 14.1% | 13.6% | exact (15.0% = truth) | 8.3% vs **0.0%** |

The aggregate class mix of the exit-window design looks almost right **only because its two biases cancel**: old long stays are added at the start and recent long stays are dropped at the end. The error sits at the time boundaries, where temporal validation and deployment happen. Also, 6–9% of exit-window rows are arrivals from before the window.

---

## 2. Required extract (what to ask the data owner for)

1. **Selection on arrival, not exit:** every import container with `DISCHARGE` in [W_start, W_end].
2. **Look-back buffer:** all stays *in the yard* at W_start (discharged earlier) as context rows only, so yard inventory, call composition and trailing departure statistics are correct from day one. These rows are never training or evaluation targets.
3. **Follow-up to the extraction cutoff X ≥ W_end + 7 days**, with every event up to X, **including stays still in the yard at X** (no TRUCK_OUT).
4. **Explicit extraction metadata:** cutoff timestamp X, extraction query, and a flag for stays deleted, transferred (PLP/bonded transfer), re-exported, or abandoned (auctioned, `LELANG`), since these are exits other than TRUCK_OUT.
5. **Full event history per stay**, including repeated events (status changes, re-handling), not only the first occurrence.

---

## 3. Labelling rule at cutoff X (implemented: `censoring.label_status`)

| Situation at X | Label | Use |
|---|---|---|
| Exited (TRUCK_OUT ≤ X) | exact class from dwell | train / evaluate |
| In yard, elapsed > 160.9 h | **CRITICAL_DELAY**: dwell > 160.9 h is already a fact, not an assumption | train / evaluate |
| In yard, 65.1 h < elapsed ≤ 160.9 h | partial: AT_RISK or CRITICAL | censoring-aware training only |
| In yard, elapsed ≤ 65.1 h | unknown | censoring-aware training only |
| Non-gate exit (transfer, re-export, auction) | separate outcome; excluded from the 3-class target with a count reported | reported, not silently dropped |

A missing TRUCK_OUT is never read as ON_TIME or CRITICAL by itself; only elapsed time decides.

---

## 4. How each observation enters training and evaluation

Let a model be trained at time T (the deployment time).

| Arrival discharge time d0 | Training (model built at T) | Evaluation |
|---|---|---|
| d0 ≤ T − 160.9 h | **complete-label training set**; every class is determined at T if unfinished stays are kept | — |
| T − 160.9 h < d0 ≤ T | censoring-aware models only (partial/unknown labels); excluded from plain classifiers | — |
| Evaluation window [E_s, E_e] with E_s ≥ T + 0 and E_e ≤ X − 160.9 h | — | every label determined; no follow-up horizon assumption needed |
| Context rows (in yard at W_start) | features only | never |

With unfinished stays retained, the evaluation window needs only **161 h** of follow-up instead of the 600 h used for the current cohort. That lets the last month of any extract be evaluated.

For landmark/event formulations the same rule applies to the prediction time t: a row is eligible if the stay is in the yard at t, and it is evaluated only if its label is determined by X.

---

## 5. Censoring-aware learning options

| Option | How | When to use | Caveat |
|---|---|---|---|
| **Delayed-label classifier (default)** | Train only on arrivals ≥ 160.9 h before T (all labels known) | Primary comparable experiments on the three-class target | Discards the most recent week; acceptable given ~1,000 arrivals/day |
| **Cumulative binary (Frank–Hall) with partial labels** | Task 1: dwell > 65.1 h (known for every row with elapsed > 65.1 h or exited). Task 2: dwell > 160.9 h (known if exited or elapsed > 160.9 h). Each task uses every row whose binary label is determined at T | Uses partial labels without imputing them | Two models must be made monotone (min-clipping, done in `fit_predict_ordinal`) |
| **Discrete-time hazard with landmarking** | Person-period rows in 12 h bins from discharge (or from the landmark) until exit or censoring; gradient-boosted hazard; class probabilities = 1 − S(65.1), S(65.1) − S(160.9), S(160.9) | Recent-regime adaptation, unified model across landmarks | Rows are correlated within stay; evaluate per stay, not per period |
| Kaplan–Meier / cohort life tables | Class-mix monitoring, prior-shift estimation | Estimating current CRITICAL prevalence for decision thresholds | Unreliable on the last 161 h alone (simulation gave 18.4% vs true 8.3% at one cutoff); must pool with older arrivals |

The three-class target and thresholds (65.1102 h / 160.8993 h) stay unchanged in every option.

---

## 6. Checks the rebuilt dataset must pass

1. `exit_window_extract` applied to the new data reproduces the published 95,648 stays (proves it is the same source).
2. Class mix of the complete-label window is stable when the cutoff X is moved earlier by 1–4 weeks, after applying the labelling rule (no boundary drift).
3. Count of stays in the yard at X is reported, along with its elapsed-time distribution.
4. No feature uses an event with availability time > t (perturbation test).
