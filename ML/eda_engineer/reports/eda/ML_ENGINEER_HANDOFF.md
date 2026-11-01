# Machine Learning Engineer Handoff: Container Dwell Time & Bottleneck Prediction

> **Assigned Audience:** Machine Learning Engineers, Data Scientists, and Software Engineering Students  
> **Repository:** `AbdulRahman1807/DS2`  
> **Status:** Stage 01 Data Engineering & Exploratory Data Analysis Complete (`MODEL-READY WITH RESTRICTIONS`)  
> **Prerequisites:** Basic Python, Pandas, and foundational Machine Learning concepts  

---

## Table of Contents
1. [Project Overview & Business Problem](#1-project-overview--business-problem)
2. [The Physical Container Journey & Event Log Structure](#2-the-physical-container-journey--event-log-structure)
3. [From Raw Telemetry to Model-Ready Datasets (Pipeline Architecture)](#3-from-raw-telemetry-to-model-ready-datasets-pipeline-architecture)
4. [Target Construction & Operational Classes](#4-target-construction--operational-classes)
5. [Partition Architecture: Train, Validation, Test, Burn-In, and Repeated Stays](#5-partition-architecture-train-validation-test-burn-in-and-repeated-stays)
6. [Complete Visualization Inventory](#6-complete-visualization-inventory)
7. [In-Depth Explanation of Major EDA Analyses & Findings](#7-in-depth-explanation-of-major-eda-analyses--findings)
8. [Complete Feature Dictionary & Groupings](#8-complete-feature-dictionary--groupings)
9. [Feature Concepts: Extraction vs. Engineering vs. Selection vs. Encoding](#9-feature-concepts-extraction-vs-engineering-vs-selection-vs-encoding)
10. [Engineered Features: Mathematical Formulas & Availability Timing](#10-engineered-features-mathematical-formulas--availability-timing)
11. [Excluded Features & Non-Active Predictors](#11-excluded-features--non-active-predictors)
12. [Data Leakage Deep Dive: Prevention & Contract Enforcement](#12-data-leakage-deep-dive-prevention--contract-enforcement)
13. [Operational Non-Stationarity: Imbalance, Temporal Drift & Cold-Start](#13-operational-non-stationarity-imbalance-temporal-drift--cold-start)
14. [Observation-Window Censoring & Test Cohort Survivorship Bias](#14-observation-window-censoring--test-cohort-survivorship-bias)
15. [Automated Test Suite & Verification Guarantees](#15-automated-test-suite--verification-guarantees)
16. [Known Limitations & Pending Project-Owner Decisions](#16-known-limitations--pending-project-owner-decisions)
17. [Actionable ML Implementation Roadmap](#17-actionable-ml-implementation-roadmap)
18. [Glossary of Maritime & Machine Learning Terms](#18-glossary-of-maritime--machine-learning-terms)
19. [Source Map & File Reference Index](#19-source-map--file-reference-index)

---

## 1. Project Overview & Business Problem

### 1.1 The Global Logistics Challenge
Global container shipping moves approximately 90% of global trade. When an ocean-going container vessel arrives at an import container terminal, hundreds to thousands of steel shipping containers are discharged from vessel holds onto the wharf.

In modern container terminal operations, **dwell time**—the duration a container occupies valuable yard real estate between vessel discharge and terminal gate-out—is the central operational bottleneck:
- **Terminal Gridlock:** When containers linger excessively in the yard, stack blocks reach capacity utilization (>80%). Yard cranes (Rubber-Tyred Gantry cranes / RTGs) must execute costly reshuffling moves ("digging") to unearth buried boxes.
- **Port Congestion Propagation:** Yard paralysis cascades backward to vessel berths, delaying ship turnaround, and forward to inland supply chains, causing trucker queues and factory supply starvation.
- **Demurrage & Detention Costs:** Extended dwell incurs heavy financial penalties for cargo owners and shipping lines.

### 1.2 The Machine Learning Objective
The goal of this project is to build an **AI-powered operational decision system** that predicts the dwell duration risk category of every import container at the instant it arrives at the terminal.

Specifically, at the moment of container discharge, our model will classify the container into one of three operational risk tiers:
1. `0: ON_TIME` (clears the yard rapidly, dwell $\le 65.11	ext{ hours}$)
2. `1: AT_RISK` (moderate dwell delay, $65.11	ext{ hours} < 	ext{dwell} \le 160.90	ext{ hours}$)
3. `2: CRITICAL_DELAY` (severe congestion dwell, $	ext{dwell} > 160.90	ext{ hours}$)

### 1.3 Intended Prediction Timestamp ($t_{	ext{prediction}}$)
The model is designed to execute at the **Discharge Milestone** ($t_{	ext{prediction}} = 	ext{discharge\_start\_ts}$). At this exact timestamp:
- The container is being lifted from the vessel cell by the quay crane.
- The Terminal Operating System (TOS) assigns an initial target yard block and bay in the stack.
- The AI model outputs a risk score to guide TOS dispatchers: routing predicted `CRITICAL_DELAY` boxes to dedicated, accessible overflow blocks while positioning `ON_TIME` boxes in rapid-delivery lanes.

---

## 2. The Physical Container Journey & Event Log Structure

Understanding our data requires understanding how a container physically traverses an import terminal.

```
       OCEAN VOYAGE                    BERTH / WHARF                   CONTAINER YARD                    GATE-OUT
+--------------------------+    +--------------------------+    +--------------------------+    +------------------+
| 1. BAPLIE EDI Message    | -> | 2. Vessel ATB            | -> | 4. STACK / YARD DWELL    | -> | 6. TRUCK_IN      |
|    (Stowage plan arrives |    |    (Ship berths at dock; |    |    (Box rests in block;  |    |    (Drayage truck|
|     ~30h pre-discharge)  |    |     ~5.5h pre-discharge) |    |     Customs docs cleared)|    |     arrives)     |
+--------------------------+    +--------------------------+    +--------------------------+    +------------------+
                                             |                               |                               |
                                             v                               v                               v
                                +--------------------------+    +--------------------------+    +------------------+
                                | 3. DISCHARGE             |    | 5. CUSTOMS & INSPECTION  |    | 7. TRUCK_OUT     |
                                |    (Quay crane lifts box |    |    (Optional SP_RED_FLAG,|    |    (Box departs; |
                                |     from vessel cell)    |    |     BEHANDLE inspection) |    |     Dwell clock  |
                                |    **PREDICTION TIME**   |    +--------------------------+    |     TERMINATES)  |
                                +--------------------------+                                    +------------------+
```

### 2.1 The Milestone Activities in the Event Log
The cleaned event log (`data/interim/cleaned_event_log.parquet`) contains **952,069 event rows** tracking 20 distinct terminal activities across **95,647 container stays**:

| Activity Name | Typical Timing | Median Duration | Description |
|---|---|---|---|
| `BAPLIE` | -29.3 hours pre-discharge | 18.0 min | Bay Plan EDI message sent by the ocean carrier detailing equipment size, type, weight, and stowage cell. |
| `VESSEL_ATB` | -5.5 hours pre-discharge | 70.0 min | Actual Time of Berthing: vessel touches the dock and secures moorings. |
| `DISCHARGE` | **0.0 hours (t0)** | **87.9 min** | Quay crane lifts container from vessel hold and places it onto a terminal chassis. **(Prediction Anchor)** |
| `STACK` | +0.4 hours post-discharge | 22.0 min | Yard crane places the container into its assigned ground slot in a yard block. |
| `HAS_QUARANTINE_FLAG` | -1.5 hours to +10.0 hours | 84.0 min | Quarantine hold placed on container by port health authority. (67.4% pre-discharge, 32.6% in-yard). |
| `SP_RED_FLAG_PIB` | -14.0 hours to +15.0 hours | Date-only (00:00) | Indonesian Customs red-flag notice requiring physical inspection. (54.4% pre-discharge, 45.6% in-yard). |
| `BEHANDLE_QUARANTINE_JOB`| +24.0 hours to +72.0 hours | 60.0 min | Physical customs inspection conducted in-yard by terminal personnel. |
| `JOB_DEL` | +32.2 hours post-discharge | 21.0 min | Delivery order / customs release permit (`JOB_DEL_DOCTYPE`: SPPB, DO, etc.) issued for cargo release. |
| `TRUCK_IN` | Variable (Dwell end - 1.2h) | 65.0 min | External drayage truck enters terminal security gates to pick up the import container. |
| `UNSTACK_TO_TRUCK` | Variable (Dwell end - 0.5h) | 88.0 min | Yard crane lifts the container off the stack and mounts it onto the external truck chassis. |
| `TRUCK_OUT` | **Dwell end (+1.5h to +2592h)** | **24.0 min** | Drayage truck passes terminal exit optical gate. **(Dwell clock termination)** |

---

## 3. From Raw Telemetry to Model-Ready Datasets (Pipeline Architecture)

The Data Engineering pipeline is 100% deterministic, auditable, and automated via [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py). It executes seven sequential processing stages:

```
[Raw Excel: 63.8 MB]
        |
        v
+-------------------------------+
| Stage 1: Ingestion & Parquet  | -> data/raw/mendeley_event_log.parquet (952,080 rows)
+-------------------------------+
        |
        v
+-------------------------------+
| Stage 2: Normalization &      | -> data/interim/cleaned_event_log.parquet (952,069 rows)
|          Quarantine           | -> data/interim/quarantined_stays.parquet (1 stay: CSNU6526902)
+-------------------------------+
        |
        v
+-------------------------------+
| Stage 3: Case Pivoting        | -> data/interim/case_attributes.parquet (95,647 cases x 12 cols)
+-------------------------------+
        |
        v
+-------------------------------+
| Stage 4: Dwell Clock &        | -> Computes dwell_hours = (t_truck_out - t_discharge) / 3600
|          Percentile Labeling  | -> Fits cutoffs strictly on train: p50=65.1102h, p85=160.8993h
+-------------------------------+
        |
        v
+-------------------------------+
| Stage 5: Rolling Metrics      | -> Computes lagged completed departures [t-7, t-1]
|          Engine               | -> Adds trailing_7d_volume, historical_delay_rate_7d, etc.
+-------------------------------+
        |
        v
+-------------------------------+
| Stage 6: Time & Group         | -> Partitions into 5 leak-free parquets:
|          Splitter             |    train (60,698), val (20,879), test (13,959), burn_in (38), repeated (73)
+-------------------------------+
        |
        v
+-------------------------------+
| Stage 7: Dataset Manifest     | -> data/processed/dataset_manifest.json (SHA-256 hashes & metadata)
+-------------------------------+
```

### 3.1 Cleaning, Normalization & Tie-Breaking
1. **Activity Canonicalization:** Standardizes string casing, strips trailing whitespace, and unifies synonym variants.
2. **Deterministic Tie-Breaking:** If two events for the same container share identical timestamps, ties are broken deterministically by sorting on secondary keys: `[case_id, Start Timestamp, End Timestamp, activity]`.
3. **Invalid Record Quarantine:** Stays with uncompleted terminal trajectories (e.g. `DISCHARGE` present but `TRUCK_OUT` never recorded) are quarantined to `data/interim/quarantined_stays.parquet`. Exactly 1 case (`CSNU6526902-2021-04-19`, 11 event rows) was quarantined due to missing gate-out.

---

## 4. Target Construction & Operational Classes

### 4.1 The Continuous Dwell Formula
Container dwell duration is calculated strictly from validated milestone timestamps, completely bypassing the raw `Duration` column (which holds unparseable Excel string artifacts):
$$	ext{dwell\_hours} = rac{t_{	ext{truck\_out\_end}} - t_{	ext{discharge\_start}}}{3600}$$
$$	ext{dwell\_days} = rac{	ext{dwell\_hours}}{24}$$

### 4.2 Training-Derived Quantile Thresholds
To prevent **lookahead bias**, percentile cutoffs were fitted **strictly on the 60,698 training stays** (`train.parquet`) and then frozen across all other partitions:

| Quantile | Value in Hours | Value in Days | Operational Meaning |
|---|---|---|---|
| **$p_{50}$ (Median)** | **`65.1102 hours`** | 2.71 days | 50% of training containers clear the yard within ~2.7 days. Standard container free-time allowance. |
| **$p_{85}$ (85th Percentile)**| **`160.8993 hours`** | 6.70 days | Containers exceeding ~6.7 days represent severe operational dwell bottlenecks and incur heavy demurrage. |

### 4.3 Target Class Definition
$$	ext{target\_class} = egin{cases} 
0 \quad (	ext{ON\_TIME}) & 	ext{if } 	ext{dwell\_hours} \le 65.1102	ext{h} \
1 \quad (	ext{AT\_RISK}) & 	ext{if } 65.1102	ext{h} < 	ext{dwell\_hours} \le 160.8993	ext{h} \
2 \quad (	ext{CRITICAL\_DELAY}) & 	ext{if } 	ext{dwell\_hours} > 160.8993	ext{h}
\end{cases}$$

---

## 5. Partition Architecture: Train, Validation, Test, Burn-In, and Repeated Stays

The dataset is partitioned chronologically based on vessel call dates (`call_date`) with strict container entity isolation:

```
Vessel Call Date Axis (2021)
May 12                     June 10                                August 26   August 27          Sept 11   Sept 12          Sept 30
  |---------------------------|---------------------------------------|----------|-------------------|--------|-------------------|
  |   BURN-IN PARTITION       |          TRAINING PARTITION           |  BUFFER  |    VALIDATION     | BUFFER |    TEST PARTITION |
  |   38 stays                |          60,698 stays                 |  CLEAN   |    20,879 stays   | CLEAN  |    13,959 stays   |
  |   (Historical backlog)    |          (70% of dense window)        |          |    (15% window)   |        |    (15% window)   |
  |---------------------------|---------------------------------------|----------|-------------------|--------|-------------------|
```

### 5.1 Partition Breakdown Table

| Partition File | Split Tag | Sample Count | Unique Containers | Call Date Range | Observable Share | Target Mix (0 / 1 / 2) | Role in Machine Learning |
|---|---|---|---|---|---|---|---|
| `burn_in.parquet` | `burn_in` | 38 | 38 | `2021-05-12` to `2021-06-08` | 0.26% | 0% / 0% / 100% | **EXCLUDED FROM TRAINING.** Severe left-truncation; containers remained in yard for months. |
| `train.parquet` | `train` | 60,698 | 60,690 | `2021-06-10` to `2021-08-26` | 98.03% | 50.0% / 35.0% / 15.0% | **CORE TRAINING SET.** Primary dataset for fitting models and preprocessors. |
| `val.parquet` | `val` | 20,879 | 20,879 | `2021-08-27` to `2021-09-11` | 99.69% | 66.5% / 26.1% / 7.4% | **PRIMARY OUT-OF-TIME BENCHMARK.** Fully observable cohort inside clean band for model selection. |
| `test.parquet` | `test` | 13,959 | 13,959 | `2021-09-12` to `2021-09-30` | 85.42% | 72.4% / 24.8% / 2.8% | **RESTRICTED TEST BENCHMARK.** Right-truncated; suffers from survivorship bias. |
| `repeated_stays.parquet` | `repeated_eval` | 73 | 73 | `2021-08-27` to `2021-09-27` | 94.27% | 56.2% / 41.1% / 2.7% | **LONGITUDINAL EVALUATION.** Repeat container visits quarantined to guarantee train/val/test disjointness. |
| **Total / Combined** | — | **95,647** | **95,566** | `2021-05-12` to `2021-09-30` | — | — | Clean band contains 77,364 stays across all partitions. |

### 5.2 Entity Isolation Guarantee
$$	ext{Containers}(	ext{Train}) \cap 	ext{Containers}(	ext{Val}) = \emptyset, \quad 	ext{Containers}(	ext{Train}) \cap 	ext{Containers}(	ext{Test}) = \emptyset, \quad 	ext{Containers}(	ext{Val}) \cap 	ext{Containers}(	ext{Test}) = \emptyset$$
When a container visited the terminal multiple times during 2021, its initial visit was kept in `train.parquet`, while its subsequent repeat visits during the validation and test periods were quarantined to `repeated_stays.parquet`. This guarantees **100% mutual entity disjointness** without moving future dates into the training set.

---

## 6. Complete Visualization Inventory

The EDA pipeline generates exactly **10 figures** located in [`reports/eda/figures/`](file:///Users/abdul/Desktop/DS/reports/eda/figures/) (indexed also in [`reports/eda/visualization_inventory.csv`](file:///Users/abdul/Desktop/DS/reports/eda/visualization_inventory.csv)):

| Figure Number & Filename | Title & Purpose | Chart Type | Data & Subset | Key Empirical Finding |
|---|---|---|---|---|
| [**01_dwell_distribution.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/01_dwell_distribution.png) | Dwell Time Distribution Across Splits | 2-panel density step histogram | All stays ($n=95,536$) vs Clean band ($n=77,334$) | Dwell is unimodal with peak around 60–70h. No natural trough exists between classes; thresholds are operational policy choices. Test set is shifted leftward due to truncation. |
| [**02_dwell_by_category.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/02_dwell_by_category.png) | Dwell by Container Type, Size & Customs Flags | 3-panel boxplots (log scale) | Clean-band model splits ($n=77,334$) | Container dimensions (20ft vs 40ft vs 45ft) show almost identical dwell distributions. Customs red-flags and Behandle inspections shift median dwell from 60h to >140h. |
| [**03_extract_truncation.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/03_extract_truncation.png) | Extract Selection Truncation by Call Week | 2-panel stacked time series (bars + log lines) | All labeled stays ($n=95,647$) across 21 weeks | Edge weeks are truncated. Final two weeks in September miss 25% and 50% of stays because slow containers had not yet departed by Sept 30. |
| [**04_weekly_dwell_vs_cutoffs.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/04_weekly_dwell_vs_cutoffs.png) | Weekly Dwell Drift vs. Fixed Train Cutoffs | 2-panel time series (dwell lines + % critical bars) | Call weeks with $\ge 200$ stays | Dwell is non-stationary: dataset contains one massive operational congestion wave in mid-July (p85 spiked to ~260h, >28% critical). Operations calmed in August/September. |
| [**05_call_rhythm.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/05_call_rhythm.png) | Daily Vessel Call Rhythms & Schedule Gaps | 2-panel chart (daily series + weekday averages) | 104 dense call dates | 9 dates have zero vessel calls; 8 of 9 are Thursdays (mean 332 stays vs ~1000 on other days). Confirms scheduled shipping rhythm, not missing data. |
| [**06_noise_floor.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/06_noise_floor.png) | Operational Noise Floor of 7-Day Delayed Share | 2-panel time series (delay share + noise band) | Stays indexed by gate-out date ($n=95,647$) | Historical delay share fluctuates by $\pm 2.7	ext{ pp}$ ($1\sigma$) due to day-to-day randomness. Shifts smaller than $5.4	ext{ pp}$ ($2\sigma$) are indistinguishable from noise. |
| [**07_reporting_coverage.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/07_reporting_coverage.png) | Reporting Coverage of Optional Milestones | 2D Heatmap with cell values | Clean-band weeks across optional activities | Optional activities show substantial coverage drift: quarantine flags range from 18% to 38% across weeks, exceeding binomial sampling noise by 10x. |
| [**08_split_shift.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/08_split_shift.png) | Covariate and Target Label Shift Across Splits | 2-panel chart (horizontal PSI bar + stacked bars) | Model splits (`train`, `val`, `test`) | Static features have near-zero PSI ($<0.05$). Operational rolling metrics have massive PSI ($>3.0$). Critical delay class collapses from 15.0% in train to 7.4% in val and 2.8% in test. |
| [**09_milestone_timing.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/09_milestone_timing.png) | Milestone Timing Relative to DISCHARGE Start | Horizontal range plot ($p_{10}, p_{50}, p_{90}$) | 952,069 events across all stays | `BAPLIE` occurs -29.3h pre-discharge; `JOB_DEL` occurs +32.2h post-discharge; `BEHANDLE` occurs 100% in-yard. Provides irrefutable proof of target leakage for post-arrival features. |
| [**10_discharge_cycles.png**](file:///Users/abdul/Desktop/DS/reports/eda/figures/10_discharge_cycles.png) | Operational Discharge Weekday & Hourly Cycles | 2-panel 2D Heatmaps | Clean-band stays ($n=77,364$) | Discharge operates 24/7. Containers discharged on Fridays and Saturdays dwell substantially longer (median 75–85h vs 55–60h on weekdays) due to weekend customs/trucking closures. |

---

## 7. In-Depth Explanation of Major EDA Analyses & Findings

### 7.1 Dataset Cardinality & Data Quality Ledger
- **Objective:** Verify record completeness, distinct entity keys, and data cleanliness.
- **Method:** Evaluated unparseable strings, negative durations, midnight signatures, and sentinel years across all 952,069 events ([`reports/eda/tables/quality.csv`](file:///Users/abdul/Desktop/DS/reports/eda/tables/quality.csv)).
- **Findings:**
  - Zero unparseable timestamps or negative milestone durations.
  - Six `CUSTOMS_DEL` events had sentinel year timestamps (e.g. `1954-04-02`); isolated in `cleaner.py`.
  - Date-only midnight timestamps (`00:00:00`) occur consistently on customs events (`CUSTOMS_DEL`, `SP_RED_FLAG_PIB`), causing sub-day negative hourly offsets that reflect discharge hour-of-day rather than true chronological violations.
- **ML Implication:** The core timestamp clock (`discharge_start_ts` to `truck_out_end_ts`) is clean and reliable.

### 7.2 Univariate Feature Signal Screening
- **Objective:** Measure univariate association between each candidate feature and dwell duration.
- **Method:** Spearman rank correlation ($ho$) for continuous features; Kruskal-Wallis effect size ($\epsilon^2$) for categoricals ([`reports/eda/tables/signal_arrival.csv`](file:///Users/abdul/Desktop/DS/reports/eda/tables/signal_arrival.csv)).
- **Findings:**
  - Individual arrival-time predictors explain modest variance in dwell. The strongest legitimate arrival predictor is `historical_delay_rate_7d` ($ho = 0.082$), followed by `CTR_TYPE` ($\epsilon^2 = 0.050$), `YARD_BLOCK` ($\epsilon^2 = 0.049$), `trailing_7d_mean_dwell` ($ho = 0.045$), and `carrier_prefix` ($\epsilon^2 = 0.042$).
  - Post-discharge features show much higher correlation: `has_behandle_inspection` ($ho = 0.153$), `total_events_in_stay` ($ho = 0.138$).
- **ML Implication:** Linear models will struggle; non-linear gradient boosted trees (LightGBM) are needed to capture multi-feature operational interactions.

---

## 8. Complete Feature Dictionary & Groupings

The repository defines exactly **32 columns** across all Parquet datasets:

```
========================================================================================================
FEATURE NAME               DATA TYPE   TIER / STATUS      DESCRIPTION & FORMULA / PROVENANCE
========================================================================================================
1.  case_id                string      METADATA           Unique container stay identifier (e.g. AAAU9001220-2021-06-28)
2.  container_id           string      METADATA           ISO 6346 container code (e.g. AAAU9001220)
3.  call_date              string      METADATA           Vessel call arrival date (YYYY-MM-DD)
4.  discharge_start_ts     timestamp   METADATA           Discharge commencement timestamp (Anchor t0)
5.  split_tag              string      METADATA           Partition identifier (train, val, test, burn_in, repeated_eval)
--------------------------------------------------------------------------------------------------------
6.  carrier_prefix         string      ACTIVE (Equipment) 3-4 letter ocean shipping line code (from container_id)
7.  CTR_SIZE               float       ACTIVE (Equipment) Container length in feet (20.0, 40.0, 45.0) from BAPLIE EDI
8.  CTR_TYPE               string      ACTIVE (Equipment) ISO container category code (GP, HC, REEFER, OPEN_TOP)
9.  GROSS                  float       ACTIVE (Equipment) Container total gross mass in kg (from DISCHARGE event)
10. YARD_BLOCK             string      ACTIVE (Location)  Target yard storage block code (e.g. B1, C3)
11. YARD_SLOT              string      ACTIVE (Location)  Specific ground bay/row/tier slot location code
--------------------------------------------------------------------------------------------------------
12. discharge_hour         int         ACTIVE (Calendar)  Hour of day (0–23) of vessel discharge commencement
13. discharge_day_of_week  int         ACTIVE (Calendar)  Day of week (0=Monday, 6=Sunday)
14. discharge_month        int         ACTIVE (Calendar)  Calendar month (6=June, 9=September)
15. is_weekend             int         ACTIVE (Calendar)  Binary indicator: 1 if Saturday or Sunday, else 0
--------------------------------------------------------------------------------------------------------
16. trailing_7d_volume     float       ACTIVE (Rolling)   Lagged 7-day completed departures count [t-7, t-1]
17. trailing_7d_critical   float       ACTIVE (Rolling)   Lagged 7-day completed critical departures [t-7, t-1]
18. trailing_7d_at_risk    float       ACTIVE (Rolling)   Lagged 7-day completed at-risk departures [t-7, t-1]
19. historical_delay_rate_7d float     ACTIVE (Rolling)   Lagged 7-day proportion: (crit + at_risk) / volume
20. trailing_7d_mean_dwell float       ACTIVE (Rolling)   Lagged 7-day completed departures average dwell (hours)
--------------------------------------------------------------------------------------------------------
21. weather_severity_index float       EXCLUDED (Weather) Constant 0.0 in 2021 (IMF PortWatch GDACS catalog coverage)
22. has_active_disruption  int         EXCLUDED (Weather) Constant 0 in 2021 (IMF PortWatch GDACS catalog coverage)
--------------------------------------------------------------------------------------------------------
23. has_quarantine_flag    int         CONDITIONAL (Hold) Quarantine inspection flag (32.6% occurs post-discharge)
24. has_red_flag_pib       int         CONDITIONAL (Hold) Customs PIB red flag (45.6% occurs post-discharge)
--------------------------------------------------------------------------------------------------------
25. JOB_DEL_DOCTYPE        string      FORBIDDEN (Leak)   Customs delivery release document type (median +32.2h in yard)
26. total_events_in_stay   int         FORBIDDEN (Leak)   Total stay milestone tally; unobservable until gate-out
27. has_behandle_inspection int        FORBIDDEN (Leak)   Physical customs inspection (100% occurs in-yard during stay)
--------------------------------------------------------------------------------------------------------
28. dwell_hours            float       TARGET (y)         Continuous dwell duration in hours: (truck_out - discharge) / 3600
29. dwell_days             float       TARGET (y)         Continuous dwell duration in days: dwell_hours / 24
30. truck_out_end_ts       timestamp   TARGET (y)         Terminal exit timestamp; defines stay completion
31. target_class           int         TARGET (y)         Primary 3-tier discrete classification label (0, 1, 2)
32. target_class_name      string      TARGET (y)         String classification label (ON_TIME, AT_RISK, CRITICAL_DELAY)
========================================================================================================
```

---

## 9. Feature Concepts: Extraction vs. Engineering vs. Selection vs. Encoding

As a software engineer entering machine learning, you must distinguish between these five foundational data science stages:

```
[Raw Event Database]
        |
        v  (1. FEATURE EXTRACTION: Querying, pivoting, and extracting attributes from raw logs)
[Raw Case Attributes Table]
        |
        v  (2. FEATURE ENGINEERING: Mathematical transformations, rolling metrics, diurnal cycles)
[Enriched Feature Table]
        |
        v  (3. TARGET CONSTRUCTION & LEAKAGE ISOLATION: Removing targets & post-arrival columns)
[Candidate Feature Allowlist]
        |
        v  (4. FEATURE SELECTION: Filtering zero-variance constants & picking active subset)
[Active Baseline Predictors (15 features)]
        |
        v  (5. ENCODING & SCALING: TargetEncoding, OneHot, StandardScaler fitted STRICTLY on train)
[Numeric Feature Matrix X] -> Fed into ML Model (LightGBM)
```

1. **Feature Extraction:** Pulling raw milestone fields from individual event log rows into a unified case row (e.g. extracting `carrier_prefix` from `container_id`).
2. **Feature Engineering:** Creating new mathematical representations that do not exist in the raw data (e.g. computing the 7-day lagged rolling delay rate `historical_delay_rate_7d`).
3. **Feature Selection:** Choosing which engineered features to feed into a specific model (e.g. dropping constant-0 weather features and excluding post-discharge inspection flags).
4. **Target Construction:** Defining the ground-truth label $y$ that the model is trying to predict (e.g. discretizing dwell into classes 0, 1, 2 using frozen percentiles).
5. **Leakage Prevention:** Enforcing structural isolation so that target columns and future operational milestones never enter the feature matrix $X$.

---

## 10. Engineered Features: Mathematical Formulas & Availability Timing

### 10.1 Diurnal & Calendar Rhythm Features
Extracted from `discharge_start_ts`:
- `discharge_hour` $= 	ext{Timestamp.hour} \in [0, 23]$
- `discharge_day_of_week` $= 	ext{Timestamp.dayofweek} \in [0, 6]$ (0 = Monday, 6 = Sunday)
- `discharge_month` $= 	ext{Timestamp.month} \in [6, 9]$
- `is_weekend` $= \mathbb{I}(	ext{discharge\_day\_of\_week} \ge 5) \in \{0, 1\}$

### 10.2 Trailing 7-Day Rolling Metrics Engine
Implemented in [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py).
To prevent **lookahead leakage**, rolling operational metrics for arrival day $t$ are calculated **strictly on completed departures from previous days** using a closed-left 7-day trailing window $[t-7, t-1]$:
1. Daily aggregates are formed by gate-out date ($	au = 	ext{truck\_out\_end\_ts.date}$):
   $$	ext{DailyVol}(	au) = \sum_{i \in 	ext{Departures}(	au)} 1, \quad 	ext{DailyCrit}(	au) = \sum_{i \in 	ext{Departures}(	au)} \mathbb{I}(y_i = 2)$$
2. Continuous calendar reindexing fills zero-departure days (e.g. empty Thursdays) with 0 for counts and `NaN` for mean dwell.
3. 7-day rolling summation lagged by 1 day (`closed='left'`):
   $$	ext{trailing\_7d\_volume}(t) = \sum_{d=t-7}^{t-1} 	ext{DailyVol}(d)$$
   $$	ext{trailing\_7d\_critical}(t) = \sum_{d=t-7}^{t-1} 	ext{DailyCrit}(d)$$
   $$	ext{trailing\_7d\_at\_risk}(t) = \sum_{d=t-7}^{t-1} 	ext{DailyAtRisk}(d)$$
4. Historical delay velocity:
   $$	ext{historical\_delay\_rate\_7d}(t) = egin{cases} rac{	ext{trailing\_7d\_critical} + 	ext{trailing\_7d\_at\_risk}}{	ext{trailing\_7d\_volume}} & 	ext{if } 	ext{volume} > 0 \ 0.0 & 	ext{if } 	ext{volume} = 0 \end{cases}$$

---

## 11. Excluded Features & Non-Active Predictors

| Feature Name | Tier | Why Excluded from Active Baseline | What Would Be Needed to Reconsider |
|---|---|---|---|
| `weather_severity_index` | Excluded | Constant `0.0` across all 95,647 stays. IMF PortWatch GDACS catalog tracks major global disasters; recorded zero events in Indonesia in mid-2021. | Local micro-meteorological feed (precipitation, wind speed, squall radar) from the local port weather station. |
| `has_active_disruption` | Excluded | Constant `0` across all 95,647 stays. Same catalog coverage limitation. | Real-time port labor strike, seismic, or typhoon alerts linked to terminal UN/LOCODE. |
| `has_quarantine_flag` | Conditional | 32.6% of quarantine flags are placed **after** discharge while the container sits in the yard. Using the full-stay flag introduces future leakage. | Real-time EDI hold table confirming the quarantine flag timestamp was strictly prior to `discharge_start_ts`. |
| `has_red_flag_pib` | Conditional | 45.6% of customs red flags are issued **after** discharge commencement. | Pre-berthing customs declaration table verifying PIB registration prior to discharge. |
| `JOB_DEL_DOCTYPE` | Forbidden | Occurs median **32.2 hours after discharge** inside the container yard. 99.92% occurs post-discharge. | Valid only in Stage 02 Dynamic In-Yard sequence modeling; strictly forbidden at arrival. |
| `total_events_in_stay` | Forbidden | Cumulative count of stay events; physically unobservable until the container leaves the terminal gate. | Strictly forbidden at arrival. |
| `has_behandle_inspection`| Forbidden | Physical yard customs inspection; occurs 100% in-yard during container stay. | Valid only in Stage 02 Dynamic In-Yard sequence modeling. |

---

## 12. Data Leakage Deep Dive: Prevention & Contract Enforcement

### 12.1 Concrete Examples of Data Leakage in Logistics
Data leakage occurs when information from the future (or information derived from the target) is mistakenly included in the training features:
- **Example A (Target Derivative Leakage):** Including `truck_out_end_ts` in features. A decision tree would simply compute `truck_out_end_ts - discharge_start_ts` and achieve 100% accuracy, but the model would be completely useless in production where `truck_out_end_ts` is unknown.
- **Example B (Milestone Leakage):** Using `JOB_DEL_DOCTYPE`. When a cargo owner takes 5 days to submit customs paperwork, `JOB_DEL` happens on Day 5. If the model sees `JOB_DEL_DOCTYPE == 'SPPB'`, it learns that the paperwork was delayed, leaking the fact that the container stayed in the yard for over 5 days!
- **Example C (Rolling Metric Contamination):** Calculating 7-day rolling delays using all containers discharged during week $t$. At 9:00 AM on Monday, you cannot know whether a container discharged at 10:00 AM on Monday will be delayed. Using same-day departures creates lookahead leakage.

### 12.2 Machine-Checkable Contract Enforcement
The Data Engineering contract module [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py) provides machine-checkable protection:
```python
# Downstream ML extraction must ALWAYS invoke the contract extractor
X, y = extract_modeling_matrix(
    df,
    scenario=PredictionScenario.ARRIVAL_TIME,
    target_col="target_class",
    allow_conditional=False, # Rejects quarantine/PIB flags
    strict=True,             # Rejects missing features
)

# Active feature selection filters out constant-0 external weather placeholders
active_cols = get_active_arrival_predictors(exclude_zero_variance=True)
X = X[active_cols]

# Strict allowlist assertion raises LeakageViolationError if unknown columns enter
validate_feature_matrix(X, scenario="ARRIVAL_TIME", strict_allowlist=True)
```

---

## 13. Operational Non-Stationarity: Imbalance, Temporal Drift & Cold-Start

### 13.1 Class Imbalance Across Partitions
The target distribution is not static:
- **Training Set ($n=60,698$):** `ON_TIME` = 50.0%, `AT_RISK` = 35.0%, `CRITICAL_DELAY` = **15.0%**
- **Validation Set ($n=20,879$):** `ON_TIME` = 66.5%, `AT_RISK` = 26.1%, `CRITICAL_DELAY` = **7.4%**
- **Test Set ($n=13,959$):** `ON_TIME` = 72.4%, `AT_RISK` = 24.8%, `CRITICAL_DELAY` = **2.8%**

### 13.2 The July Congestion Wave
In mid-July 2021, the terminal experienced an acute congestion episode (weekly p85 dwell exceeded 260 hours; >28% critical delays). By late August, the crisis resolved. Models trained primarily on July data will experience **prior probability shift**; ML engineers must use probability calibration and evaluate threshold tuning.

### 13.3 The Three Cold-Start Regimes in Training Data
In `train.parquet`, completed departures only appear starting on July 1, 2021:
1. **Zero-History Cold Start (`call_date < 2021-07-01`, 4,422 stays, 7.29%):** `trailing_7d_volume == 0.0` and `historical_delay_rate_7d == 0.0`. This reflects unobserved pre-July departures in the extract, **not** an idle port.
2. **Partially Observed Window (`2021-07-01` to `2021-07-07`, 6,375 stays, 10.50%):** 1 to 6 days of departure history accumulated.
3. **Fully Observed Window (`call_date >= 2021-07-08`, 49,901 stays, 82.21%):** Full 7-day rolling history active.

---

## 14. Observation-Window Censoring & Test Cohort Survivorship Bias

### 14.1 The Mechanism of Truncation
The raw extract terminates on **`2021-09-30 23:51:25`** ($T_{	ext{end}}$).
Because data was queried from completed gate-outs, containers arriving in late September that stayed in the yard past September 30 were **never extracted**:

```
Discharge Date: Sept 12  ->  18 days to extract end  ->  Dwell up to 431h observable (85% observable)
Discharge Date: Sept 20  ->  10 days to extract end  ->  Dwell up to 240h observable (65% observable)
Discharge Date: Sept 28  ->   2 days to extract end  ->  Dwell up to  48h observable (50% observable)
```

### 14.2 The Survivorship Bias Fallacy
In `test.parquet` (call dates `2021-09-12` to `2021-09-30`):
- Long-dwelling containers arriving in late September are missing from the file.
- The completed stays that appear in `test.parquet` are disproportionately **fast-clearing containers**.
- Critical delays artificially collapse from 15.0% in train to 2.8% in test.
- A trivial dummy model predicting `ON_TIME` for every container achieves **72.4% accuracy** on `test.parquet`.
- **Methodological Rule:** Raw test set accuracy is an illusion. The ML Engineer must use `val.parquet` (99.7% observable) as the primary evaluation anchor.

---

## 15. Automated Test Suite & Verification Guarantees

The repository includes **43 automated regression tests** passing in ~12 seconds:

```
tests/test_cleaner.py .............. [ 5 passed ] -> Normalization, quarantine, deterministic tie-breaking
tests/test_dwell_clock.py ........... [ 4 passed ] -> Timestamp subtraction, train-only threshold fitting
tests/test_eda.py ................... [ 11 passed ] -> Quality ledger, observable share, PSI, milestone timing
tests/test_feature_contracts.py ..... [ 9 passed ] -> Target rejection, post-arrival rejection, manifest SHA256
tests/test_integration_de_eda.py .... [ 3 passed ] -> Real Parquet artifact loading, deliberate corruption catching
tests/test_pipeline_integration.py .. [ 6 passed ] -> 32-column schema, temporal monotonicity, entity isolation
tests/test_rolling_metrics.py ....... [ 4 passed ] -> Lookahead prevention (closed='left'), cold-start predicates
tests/test_splitter.py .............. [ 1 passed ] -> Chronological causality and entity disjointness
===================================== 43 passed in 12.60s =====================================
```

---

## 16. Known Limitations & Pending Project-Owner Decisions

1. **Test Set Right-Censoring:** Unbiased out-of-time evaluation on late-September arrivals requires obtaining extended terminal logs (October/November 2021).
2. **Prediction Timestamp Operational Sign-Off:** Project Owner must formally confirm that initial `YARD_BLOCK`, `YARD_SLOT`, and `GROSS` are available at `discharge_start_ts` from TOS pre-stowage work instructions.
3. **Customs Flag Policy:** Pre-arrival EDI holds (`has_quarantine_flag`, `has_red_flag_pib`) remain excluded from the primary baseline until an arrival-time EDI table is confirmed.
4. **Single Terminal & Single Regime:** Data covers one Indonesian container terminal over 3 months with one congestion wave. External validity across foreign ports requires recalibration.

---

## 17. Actionable ML Implementation Roadmap

The ML Engineer must execute the following sequential tasks:

```
[TASK 1: Verify Manifest & Environment]
 - Execute verify_dataset_manifest()
 - Confirm scikit-learn (v1.8.0), lightgbm (v4.7.0), xgboost (v3.3.0)
         |
         v
[TASK 2: Contract-Compliant Data Extraction]
 - Call extract_modeling_matrix(train_df, scenario='ARRIVAL_TIME', allow_conditional=False, strict=True)
 - Filter to 15 active features using get_active_arrival_predictors(exclude_zero_variance=True)
         |
         v
[TASK 3: Tier 0 Majority Class Baseline]
 - DummyClassifier(strategy='prior') on train.parquet -> evaluate on val.parquet
         |
         v
[TASK 4: Tier 1 Regularized Linear Baseline]
 - LogisticRegression(class_weight='balanced') with TargetEncoder / OneHotEncoder fitted STRICTLY on train
         |
         v
[TASK 5: Tier 2 Non-Linear Gradient Boosted Trees]
 - LightGBMClassifier with 2-fold forward-chaining CV on train.parquet (July 08 - Aug 26)
 - Evaluate on val.parquet; report Macro-F1 and Critical Delay PR-AUC
         |
         v
[TASK 6: Cold-Start Sensitivity Experiment]
 - Compare Tier 2 performance trained with vs without June cold-start stays (< 2021-07-01)
         |
         v
[TASK 7: Out-of-Fold Probability Calibration]
 - Calibrate probabilities using out-of-fold training predictions (NOT on val.parquet)
```

---

## 18. Glossary of Maritime & Machine Learning Terms

- **BAPLIE:** Bay Plan EDI message sent by shipping lines before arrival, specifying container stowage locations, sizes, and weights.
- **Dwell Time:** Total elapsed time a container sits in the port terminal from vessel discharge to terminal gate-out.
- **Discharge Milestone:** The exact operation of a quay crane lifting a container off a vessel onto a terminal trailer.
- **Gate-Out (`TRUCK_OUT`):** The final milestone where a highway drayage truck exits the terminal security gate with the container.
- **Lookahead Bias (Data Leakage):** Introducing future information into model training that would not be available at the moment of prediction.
- **Population Stability Index (PSI):** A statistical metric measuring the shift in a feature's distribution between two time periods.
- **Survivorship Bias:** Selection bias occurring when only completed stays are observed, systematically omitting slow-clearing containers still in the yard.
- **TOS (Terminal Operating System):** The central enterprise software (e.g. Navis N4) managing crane dispatch, yard stacking, and vessel discharge.
- **VESSEL_ATB (Actual Time of Berthing):** The timestamp when a vessel completes berthing alongside the terminal quay.

---

## 19. Source Map & File Reference Index

| Artifact / Module | Path | Purpose & Guarantees |
|---|---|---|
| Feature Contracts | [`ML/data_engineer/feature_contracts.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/feature_contracts.py) | Canonical allowlists, leakage validators, extraction API, manifest verification |
| Rolling Metrics Engine | [`ML/data_engineer/rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py) | Closed-left lagged historical operational aggregates, cold-start flags |
| Data Cleaner | [`ML/data_engineer/cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py) | Normalization, quarantine logic, deterministic tie-breaking, case pivoting |
| Dwell Clock | [`ML/data_engineer/dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py) | Accurate timestamp subtraction, train-only empirical quantile labeling |
| Splitter | [`ML/data_engineer/splitter.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/splitter.py) | Chronological partitioning, mutual container entity disjointness |
| Pipeline Orchestrator | [`ML/data_engineer/build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py) | End-to-end execution of Stages 1–7 and manifest generation |
| EDA Profiler | [`ML/eda_engineer/profiler.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/profiler.py) | Milestone timing, eligibility tier tagging, quality checks, univariate signal |
| Drift Analyzer | [`ML/eda_engineer/drift_analysis.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/drift_analysis.py) | Observation horizon (431.4h), clean band calculation, observable share, PSI |
| EDA Runner | [`ML/eda_engineer/eda_runner.py`](file:///Users/abdul/Desktop/DS/ML/eda_engineer/eda_runner.py) | End-to-end EDA execution, 10 chart generation methods, report compilation |
| Dataset Manifest | [`data/processed/dataset_manifest.json`](file:///Users/abdul/Desktop/DS/data/processed/dataset_manifest.json) | Cryptographic SHA-256 hashes, row counts, partition dates, frozen cutoffs |
| Visualization Inventory | [`reports/eda/visualization_inventory.csv`](file:///Users/abdul/Desktop/DS/reports/eda/visualization_inventory.csv) | Machine-readable metadata catalog for all 10 EDA figures |
| Written EDA Findings | [`reports/eda/findings.md`](file:///Users/abdul/Desktop/DS/reports/eda/findings.md) | Narrative report compiling all 25 tabular summaries and 10 figures |
| Remediation Log | [`ML/data_engineer/REMEDIATION_LOG.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/REMEDIATION_LOG.md) | Audited history of defects DE-01 through DE-16 and verified resolutions |
| Contracts Checklist | [`ML/data_engineer/DATA_CONTRACTS_CHECKLIST.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/DATA_CONTRACTS_CHECKLIST.md) | Machine-checkable compliance checklist for downstream engineers |
| Feature Matrix | [`ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md`](file:///Users/abdul/Desktop/DS/ML/data_engineer/FEATURE_PROVENANCE_MATRIX.md) | Comprehensive 32-column provenance, availability, and tier documentation |
