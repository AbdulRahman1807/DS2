# Stage 02 Prediction Design: Classical Machine Learning Formulations & Protocol

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning & Logistics Data Engineering
**Scope:** Stage 02 Task Formulation, Classical ML Candidate Architectures, and Prediction Contracts
**Status:** Approved Architectural Specification
**Artifact Directory:** `reports/ml/`
**Closeout Reference:** [`reports/ml/stage01_closeout.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage01_closeout.md)

---

## 1. Technological Constraint: Classical Machine Learning Only

> [!IMPORTANT]
> This project operates under an explicit, permanent constraint: **Classical Machine Learning Exclusively**.
> Deep Learning is strictly outside the project scope. This includes:
> - No LSTM, GRU, RNN, or recurrent architectures.
> - No Transformers, Attention mechanisms, or Foundation models.
> - No Neural Networks, Multilayer Perceptrons (MLPs), or PyTorch/TensorFlow frameworks.
>
> All Stage 02 research, baselines, and feature engineering must be built using classical tabular, tree-based, and statistical estimators (e.g., LightGBM, XGBoost, CatBoost, Scikit-Learn ensembles, Cox Proportional Hazards, Random Survival Forests, and discrete-time hazard models).

The weak predictive signal observed in Stage 01 at $t_{\text{discharge}}$ (PR-AUC $\approx 0.094$) is an information-availability defect, not proof of algorithmic limitations in classical ML. At discharge start, critical administrative events have not yet happened. Stage 02 explores whether **time-valid operational information, combined with disciplined feature engineering and classical ML**, can unlock reliable bottleneck prediction.

---

## 2. Comparative Evaluation of Four Classical ML Prediction Formulations

We systematically analyze four candidate prediction formulations across 10 standardized evaluation criteria.

```
+----------------------------------------------------------------------------------------------------+
|                                    STAGE 02 FORMULATION SPECTRUM                                   |
+--------------------+-------------------------+----------------------------+------------------------+
| Option A           | Option B (RECOMMENDED)  | Option C                   | Option D               |
| Discharge-Start    | Landmark-Based Mid-Stay | Repeated Dynamic Update    | Time-to-Event Survival |
| t = discharge      | t = discharge + 12h/24h | t = on_event_arrival       | Discrete hazard h(t)   |
| Static baseline    | Fixed cohorts           | Multi-prediction stays     | Censoring-aware        |
+--------------------+-------------------------+----------------------------+------------------------+
```

---

### Option A — Discharge-Start Prediction ($t_{\text{discharge}}$)
Continuation of the Stage 01 formulation, remaining anchored at the moment the container is unstacked from the vessel.

1. **Prediction Timestamp:** Exact crane lift-off timestamp (`discharge_start_ts`).
2. **Prediction Target:** 3-tier ordinal class (`ON_TIME` $\le 65.1\text{h}$, `AT_RISK` $65.1 - 160.9\text{h}$, `CRITICAL_DELAY` $> 160.9\text{h}$) or continuous dwell duration.
3. **Required Input Fields:** Manifest EDI (carrier, container dimensions, gross weight), diurnal/calendar cycles, and trailing 7-day completed departure metrics.
4. **Required Event History:** Trailing completed gate-outs over $[t-7, t-1]$. Zero in-stay history.
5. **Data Availability Gaps:** Complete absence of dynamic yard density, crane moves/hour, and post-discharge administrative filings.
6. **Censoring Implications:** Unaffected by mid-stay censoring, but external test partition suffers 85.4% right-censoring survivorship bias.
7. **Operational Usefulness:** High theoretical planning horizon (earliest possible alert), but very low operational reliability (precision $\approx 10\%$, recall $\approx 11\%$).
8. **Evaluation Methodology:** Standard chronological temporal folds on `discharge_start_ts`.
9. **Minimum Viable Classical Baseline:** Frozen Stage 01 Conservative XGBoost / LightGBM baseline (13 features).
10. **Risks & Limitations:** Hard predictive ceiling. Without external data collection (yard density, crane rates), further tabular tuning on arrival features alone will overfit historical noise.

---

### Option B — Landmark-Based Mid-Stay Prediction ($t_{\text{discharge}} + \Delta t$) [RECOMMENDED PRIMARY TASK]
Evaluates container bottleneck risk at fixed elapsed-time landmarks after discharge, specifically:
- **Landmark 1 ($L_{12}$):** Exactly **12 hours** after discharge start ($t_{\text{discharge}} + 12\text{h}$).
- **Landmark 2 ($L_{24}$):** Exactly **24 hours** after discharge start ($t_{\text{discharge}} + 24\text{h}$).

1. **Prediction Timestamp:** Exactly $t_{\text{discharge}} + \Delta t$ (where $\Delta t \in \{12\text{h}, 24\text{h}\}$).
2. **Prediction Target:** Binary or 3-class prediction of whether the container will exceed the critical delay threshold ($>160.9\text{h}$ or operational SLA $>168\text{h}$), conditioned on having remained in the yard up to landmark $L$.
3. **Required Input Fields:** Arrival features PLUS all operational milestones occurring strictly within $[t_{\text{discharge}}, t_{\text{discharge}} + \Delta t]$:
   - Early delivery order release indicator (`has_do_released_within_12h`).
   - Early in-yard container repositioning moves within the window.
   - Customs declaration filing status at 12h/24h.
   - Elapsed dwell already consumed ($\Delta t$).
4. **Required Event History:** Full event log timestamped within the interval $[t_{\text{discharge}}, t_{\text{discharge}} + \Delta t]$.
5. **Data Availability Gaps:** Stays in the historical extract have partial event milestone coverage. Events without sub-day timestamps cannot be safely windowed.
6. **Censoring Implications:**
   - *Departed Stays:* Containers that departed *before* the landmark ($\text{dwell} < \Delta t$) are resolved and excluded from the active landmark cohort.
   - *Active Stays:* Only containers still inside the terminal at $t_{\text{discharge}} + \Delta t$ are scored.
   - Eliminates ambiguity: evaluated strictly on active lingering stays.
7. **Operational Usefulness:** **VERY HIGH.** In maritime logistics, container terminal dispatchers conduct morning triage reviews (e.g. 07:00 shift handover). Containers lingering 12h or 24h without customs filings or delivery orders represent actionable targets for demurrage expediting.
8. **Evaluation Methodology:** Chronological partitioning based on the *landmark timestamp* ($t_L = t_{\text{discharge}} + \Delta t$). Metric evaluation on active cohorts.
9. **Minimum Viable Classical Baseline:** Regularized Logistic Regression and LightGBM / CatBoost trained on the landmark feature matrix.
10. **Risks & Limitations:** Containers exiting before the landmark are excluded, shifting the baseline critical prevalence of the remaining cohort. Requires strict timestamp validation to avoid using events recorded at e.g. 14 hours inside a 12-hour landmark.

---

### Option C — Repeated Dynamic Event-Triggered Prediction
Re-evaluates risk dynamically whenever a new process milestone is logged (e.g. at discharge, at initial yard slotting, at customs channel routing, at delivery order issuance).

1. **Prediction Timestamp:** Dynamic timestamp $t_{\text{event}}$ of each incoming event.
2. **Prediction Target:** Probability of exceeding critical threshold given current process state and elapsed dwell.
3. **Required Input Fields:** Static arrival features, cumulative elapsed dwell, sequence length of completed activities, and categorical state of latest event.
4. **Required Event History:** Full preceding process history for that stay up to $t_{\text{event}}$.
5. **Data Availability Gaps:** Non-uniform event logging; some containers have 3 events, others have 12. Timestamp inaccuracies in milestone recording.
6. **Censoring Implications:** Stays with longer dwell accumulate more events, generating an imbalanced number of training rows per container and causing severe survivorship bias.
7. **Operational Usefulness:** Moderate. Generates repeated, fluctuating alerts for the same container, creating alarm fatigue unless complex alert suppression rules are engineered.
8. **Evaluation Methodology:** Highly complex in classical ML. Standard cross-validation leaks information if multiple rows from the same container are split across folds. Requires cluster-grouped chronological splitting.
9. **Minimum Viable Classical Baseline:** Tabular gradient boosting (LightGBM) trained on flattened state-transition features with container-level grouped temporal splits.
10. **Risks & Limitations:** **Severe statistical dependency.** Violates the independent and identically distributed (i.i.d.) assumption of classical tabular models. Prone to severe survivorship bias (containers that stay 15 days generate 10 predictions, while on-time containers generate 2).

---

### Option D — Time-to-Event / Survival Prediction
Reformulates the task from fixed-duration classification to classical survival analysis, predicting the conditional survival curve $S(t) = P(\text{dwell} > t)$ or the hazard rate $h(t)$.

1. **Prediction Timestamp:** $t_{\text{discharge\_start}}$ or fixed landmarks.
2. **Prediction Target:** Time until terminal departure (`truck_out_end_ts`), explicitly accounting for right-censoring flags ($c_i = 1$ if departed, $c_i = 0$ if censored).
3. **Required Input Fields:** Tabular baseline covariates and duration until event/censoring.
4. **Required Event History:** Trailing metrics and baseline covariates.
5. **Data Availability Gaps:** Standard survival models require accurate right-censoring indicators, which must be constructed from terminal observation window boundaries.
6. **Censoring Implications:** **OPTIMAL.** Classical survival models are mathematically designed to handle right-censored data without discard or distortion, making this formulation capable of utilizing the quarantined test set.
7. **Operational Usefulness:** High. Provides an entire estimated survival curve, allowing dispatchers to calculate the probability of exceeding any arbitrary SLA (e.g. 3 days, 5 days, 7 days) from a single model.
8. **Evaluation Methodology:** Harrell's Concordance Index (C-index), time-dependent Brier score, and Integrated Brier Score (IBS) on out-of-time validation data.
9. **Minimum Viable Classical Baseline:**
   - Semi-parametric: Cox Proportional Hazards with elastic-net regularization.
   - Non-parametric / Ensemble: Random Survival Forest (`sksurv.ensemble.RandomSurvivalForest`) or discrete-time hazard modeling with LightGBM.
10. **Risks & Limitations:** The Proportional Hazards assumption ($h(t \mid x) = h_0(t) \exp(x^\top \beta)$) is strongly violated by the macro operational regime shift between July and August. Non-proportional extensions or discrete-time hazard models are required.

---

## 3. Objective Formulation Comparison Matrix

| Evaluation Dimension | Option A: Discharge-Start | Option B: Landmark Mid-Stay (RECOMMENDED) | Option C: Repeated Dynamic | Option D: Time-to-Event Survival |
| :--- | :---: | :---: | :---: | :---: |
| **Statistical Independence** | High (1 row/case) | **High (1 row/case per landmark)** | Poor (Multiple correlated rows) | High (1 row/case) |
| **Classical ML Fit** | Excellent (Tabular GBDT) | **Excellent (Standard Tabular GBDT)**| Challenging (Multi-row grouping)| Specialized (Cox / RSF / Hazard)|
| **Operational Feasibility** | High (Early notice) | **Highest (Actionable shift triage)**| Low (Alarm fatigue risk) | High (Continuous curve) |
| **Censoring Robustness** | Vulnerable | **Good (Active cohort conditioned)** | Highly Vulnerable | **Mathematical Design (Optimal)** |
| **Incremental Signal Potential**| Near Zero (Data ceiling) | **High (Captures early in-yard events)**| High (Captures all events) | Moderate (Better math, same data)|
| **Implementation Complexity** | Minimal | **Low / Moderate** | Very High | Moderate / High |
| **Production Alert Policy** | Static threshold | **Shift-based expediting list** | Event-driven alert engine | SLA probability threshold |

---

## 4. Formal Recommendation: Primary Stage 02 Task

### Primary Selection: Option B (Landmark-Based Mid-Stay Prediction at 12h & 24h)
We formally recommend **Option B** as the primary modeling task for Stage 02, supported by **Option D (Discrete-Time Hazard Modeling)** as a complementary analytical tool for censoring correction.

### Why Option B is Selected
1. **Breaks the Arrival Information Ceiling:** Option B allows the model to observe early administrative milestones (`JOB_DEL_DOCTYPE` release status, customs filing initiation, early yard repositioning) that are strictly unobservable at $t_{\text{discharge}}$.
2. **Maintains Classical ML Rigor:** Unlike Option C, each container has exactly one observation per landmark cohort. Standard classical ML algorithms (LightGBM, XGBoost, Random Forest, Logistic Regression) can be trained without violating sample independence.
3. **Direct Operational Alignment:** Terminal dispatchers operate on shift schedules. A 12-hour or 24-hour landmark triage report directly informs the morning or evening yard work queues.
4. **Clean Evaluation Protocol:** Pre-landmark departures are cleanly resolved (known non-bottlenecks). The model focuses its predictive capacity exclusively on the lingering containers that represent actual operational risk.

### Why Other Options Were Not Selected as Primary
- **Option A:** Stage 01 proved that arrival features at $t_{\text{discharge}}$ hit an empirical ceiling of $\sim 0.094$ PR-AUC. Continuing Option A alone without new external data sources is unproductive.
- **Option C:** Creates severe longitudinal correlation, multi-row survivorship bias, and complex multi-prediction evaluation issues that add technical debt without operational clarity.
- **Option D:** Standard continuous survival libraries (e.g. `scikit-survival`) introduce specialized dependencies, and continuous Cox models assume proportional hazards, which fail under macro regime shifts. However, a **discrete-time hazard model using classical GBDTs** will be maintained as a secondary validation tool.

---

## 5. Precise Prediction Contract for Stage 02 Primary Task (Option B)

### 5.1 Landmark Cohort Definition
For landmark horizon $\Delta t \in \{12.0\text{ hours}, 24.0\text{ hours}\}$:
- **Eligibility Filter:** Containers where:
  $$\text{discharge\_start\_ts} + \Delta t \le \min(\text{truck\_out\_end\_ts}, t_{\text{eval\_cutoff}})$$
  (i.e. Containers that have not departed the terminal prior to landmark $L$).
- **Resolved Exclusions:** Containers departing before landmark $L$ ($\text{dwell\_hours} \le \Delta t$) are resolved as non-critical and documented in cohort accounting.

### 5.2 Landmark Feature Allowlist Contract

| Feature Group | Feature Name | Description | Availability Rule |
| :--- | :--- | :--- | :--- |
| **Static Baseline** | `carrier_prefix`, `CTR_SIZE`, `CTR_TYPE`, `GROSS` | Physical container manifest attributes | Inherited from Stage 01 |
| **Discharge Cycle** | `discharge_hour`, `discharge_day_of_week`, `is_weekend` | Quay crane lift timing cycle | Inherited from Stage 01 |
| **Trailing Baseline**| `trailing_7d_volume`, `trailing_7d_mean_dwell`, etc. | 7-day lagged completed departures | Inherited from Stage 01 |
| **Landmark State** | `elapsed_landmark_hours` | Constant $\Delta t$ (12.0 or 24.0 hours) | Known at landmark |
| **Process Milestone**| `has_job_del_doctype_released_by_landmark` | Binary flag: delivery order released $\le t_L$ | Computed strictly on events with timestamp $\le t_L$ |
| **Process Milestone**| `hours_to_job_del_doctype_release` | Elapsed hours from discharge to DO release | NaN / Imputed if not yet released |
| **In-Yard Activity** | `moves_count_by_landmark` | Count of yard crane reshuffle moves $\le t_L$ | Computed strictly on events with timestamp $\le t_L$ |
| **In-Yard Activity** | `last_yard_activity_code` | Categorical code of latest activity $\le t_L$ | Latest event timestamped $\le t_L$ |

> [!CAUTION]
> **Anti-Leakage Invariant:** Any event with `Start Timestamp` $> t_{\text{discharge\_start}} + \Delta t$ is STRICTLY FORBIDDEN from entering the landmark feature matrix. Automated CI contracts must enforce this invariant.

### 5.3 Target Definition at Landmark $L$
- **Target Formulation (Conditional Critical Delay):**
  $$y_L = \mathbb{I}(\text{dwell\_hours} > 160.8993\text{ hours})$$
  Given that the container has already survived $\Delta t$ hours in the terminal.

---

## 6. Minimum Viable Classical ML Baseline for Stage 02

The minimum viable baseline for Option B will compare:
1. **Majority Prior Baseline:** Empirical critical rate within the eligible landmark cohort.
2. **Regularized Logistic Regression:** Linear baseline with standard scaling and one-hot encoding.
3. **Random Forest Classifier:** 100 trees, max depth 8, measuring non-linear interactions between elapsed milestone timing and container weight.
4. **LightGBM Classifier (Primary Production Candidate):** Gradient-boosted decision trees with native categorical handling, evaluated both unweighted and with class weighting.
5. **Discrete-Time Hazard Estimator:** Multi-period tabular GBDT predicting the hazard of gate-out in subsequent 24-hour intervals.

No deep learning models will be built, evaluated, or benchmarked. All Stage 02 experiments will remain strictly within classical Machine Learning.
