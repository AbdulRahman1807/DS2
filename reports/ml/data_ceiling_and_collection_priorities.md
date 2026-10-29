# Data Ceiling Assessment & Next-Phase Data Collection Priorities

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning & Logistics Data Engineering
**Scope:** Stage 01 Post-Audit Evaluation & Stage 02 Data Requirements
**Status:** Approved Architectural Document
**Artifact Directory:** `reports/ml/`

---

## 1. Executive Assessment: The Arrival-Time Data Ceiling

The Stage 01 predictive signal audit established that container dwell bottlenecks cannot be reliably predicted at the moment of quay crane discharge ($t_{\text{discharge\_start}}$) using static arrival features alone.
- **Observed Critical PR-AUC:** **0.0937** vs **0.0737** baseline prevalence on validation data.
- **Observed Critical Recall:** **11.18%** at default argmax decision rule (missing 88.82% of bottlenecks).
- **Separation Ratio:** The mean predicted critical risk score for actual delayed containers (0.267) is only **1.16x** higher than for non-delayed containers (0.229).

### Why Arrival Features Encounter a Hard Predictive Ceiling
At the instant a container is lifted from a vessel cell and set onto the quay apron:
1. **Administrative Delays Have Not Occurred:** Over 70% of prolonged container dwell is caused by customs clearance holds (`PPJK` document audits, physical inspection channel routing), consignee drayage payment credit disputes, or delivery order release (`JOB_DEL_DOCTYPE`) delays. None of these administrative outcomes exist at discharge start.
2. **Terminal Queueing Dynamics Are Systemic, Not Container-Specific:** Whether a container stays 48 hours or 200 hours is heavily driven by terminal yard stack congestion, crane gang availability, and drayage gate queues during the stay.
3. **Macro Operational Regimes Dominate Micro Attributes:** In July 2021, terminal mean dwell surged to 132.8 hours with 27.9% critical delays; in late August, mean dwell dropped to 63.2 hours with 7.4% critical delays. Static equipment features (size, type, gross weight) have almost zero ability to anticipate these macro shifts without real-time yard density metrics.

---

## 2. Prioritized Roadmap for Next-Phase Data Collection

The following five data sources are prioritized based on their potential to break the current predictive ceiling, their operational feasibility, and their temporal availability.

```
Horizon A (Pre-Arrival)        Horizon B (Discharge Start)          Horizon C (Mid-Stay Milestones)
[BAPLIE Manifest]              [Quay Crane Lift / Cycle]            [TOS Yard Reshuffles]
 - Carrier prefix               - Discharge hour / dow / weekend     - Yard block moves
 - Container size / type        - P1: Dynamic Yard Stacking Density  - P4: Customs Document Release
 - SOLAS VGM gross weight       - P2: Crane Moves/Hour & Berths      - P4: Consignee DO Booking
                                - P3: Gate OCR Queues                - Truck Appointment Scheduled
```

---

### Priority 1: Dynamic Yard Block Occupancy & Stacking Density
- **Description:** Real-time Terminal Operating System (TOS) yard snapshot captured at the exact moment of container drop-off.
  - Yard block utilization percentage (TEU currently stored vs block TEU capacity).
  - Average stack height in the target block (e.g., 4-high vs 2-high).
  - Reefer plug slot saturation (for refrigerated containers).
  - Reshuffle / dig ratio (number of non-productive crane moves required to retrieve bottom containers).
- **Prediction Horizon:** Horizon B ($t_{\text{discharge}}$).
- **Information Added:** Quantifies physical yard congestion at container arrival. If a block is 95% full and stacked 5-high, retrieval turnaround times multiply by $3\times$.
- **Leakage Risk & Safeguards:**
  - *Risk:* In historical extracts, TOS databases often record the *final* container slot location upon gate-out rather than the *initial* drop-off slot.
  - *Safeguard:* Telemetry must be timestamped at initial crane set-down. Reshuffle coordinates occurring after discharge must be masked.

---

### Priority 2: Quay Crane & Berth Telemetry
- **Description:** Live operational telemetry from the container berth and quay cranes (STS cranes).
  - Gross Crane Moves per Hour (GMPH) across the vessel call.
  - Number of active quay crane gangs working the vessel.
  - Vessel call turnaround delay (actual arrival timestamp vs scheduled window).
  - Total discharge volume scheduled for the vessel call.
- **Prediction Horizon:** Horizon B ($t_{\text{discharge}}$).
- **Information Added:** Captures vessel gang productivity and operational pressure on the quay apron. Low moves-per-hour indicate crane breakdowns or apron traffic jams that ripple into yard dispatch.
- **Leakage Risk & Safeguards:**
  - *Risk:* Post-discharge moves (e.g. export loading moves occurring after the container has already landed) must not be included.
  - *Safeguard:* Strict windowing: only crane moves recorded prior to or at the specific container's discharge start timestamp may be aggregated.

---

### Priority 3: Gate OCR Queues & Drayage Turnaround Metrics
- **Description:** Live gate system telemetry from optical character recognition (OCR) lanes and weighbridges.
  - External drayage truck queue length outside the terminal gate.
  - Trailing 4-hour truck turnaround time (gate-in to gate-out).
  - Hourly drayage appointment booking demand vs available gate passes.
- **Prediction Horizon:** Horizon B ($t_{\text{discharge}}$).
- **Information Added:** Identifies external drayage shortages or regional trucking bottlenecks (e.g. holiday curfews, road construction outside port access gates).
- **Leakage Risk & Safeguards:**
  - *Risk:* Lookahead leakage if gate statistics for the entire day are assigned to morning container discharges.
  - *Safeguard:* Trailing windowing: aggregate gate telemetry strictly over $[t-4\text{ hours}, t-15\text{ minutes}]$.

---

### Priority 4: In-Stay Customs and Consignee Event Milestones (Stage 02)
- **Description:** Longitudinal process event milestones occurring during the container's stay in the yard.
  - Customs import declaration (`PPJK`) submission timestamp.
  - Customs channel assignment (Green channel = instant release; Red channel = physical quarantine inspection).
  - Delivery Order (`JOB_DEL_DOCTYPE`) electronic release by the shipping line.
  - Consignee gate appointment booking timestamp.
  - Intermediate yard moves (shifting from discharge stack to inspection apron or rail transfer).
- **Prediction Horizon:** **Horizon C ($t > t_{\text{discharge}}$) — STAGE 02 ONLY.**
- **Information Added:** Direct observation of the primary drivers of extreme dwell delay. A container assigned to the customs red channel with an unreleased delivery order has a $>90\%$ probability of critical delay.
- **Leakage Risk & Safeguards:**
  - *CRITICAL LEAKAGE FOR STAGE 01:* Must **NEVER** be exposed to an arrival-time ($t_{\text{discharge}}$) baseline.
  - *Stage 02 Design:* Must be modeled using classical machine learning estimators (e.g., Landmark-based LightGBM/XGBoost classifiers, classical survival analysis, or discrete-time hazard models) that re-evaluate risk scores at landmark intervals as milestones occur:
    $$S(\text{Delay} \mid t_{\text{discharge}}) \xrightarrow{+\text{Red Channel}} S(\text{Delay} \mid t_{\text{customs}}) \xrightarrow{+\text{DO Release}} S(\text{Delay} \mid t_{\text{DO}})$$

---

### Priority 5: Arrival-Based Cohort Tracking to Eliminate Censoring Bias
- **Description:** Redesign the data extraction pipeline from a departure-filtered export to an **arrival-cohort tracking protocol**.
  - Current Defect: The Mendeley event log only recorded stays where `TRUCK_OUT` occurred before September 30, 2021. For containers arriving in late September, long-dwell containers were still in the yard and therefore completely absent from the extract (causing 85.4% right-censoring survivorship bias in `test.parquet`).
  - Remedy: Select an arrival cohort (e.g. all containers discharged between June 1 and July 31), and maintain an observation follow-up window through October 31 (a minimum 90-day follow-up).
- **Prediction Horizon:** Data pipeline architecture.
- **Information Added:** Ensures slow-moving cargo is fully represented across all test and validation partitions, eliminating artificial drop-offs in critical delay prevalence.

---

## 3. Limits of Generalization: Single-Terminal Extract

All conclusions in this repository are derived from a single historical extract:
- **Facility:** Terminal Petikemas Surabaya (TPS), Port of Tanjung Perak, Surabaya, Indonesia.
- **Timeframe:** June 10, 2021 to September 30, 2021.
- **Context:** Acute COVID-19 delta-wave emergency restrictions (`PPKM`), regional truck driver testing mandates, and global supply chain vessel bunching.

### Generalization Warnings for Logistics Stakeholders
1. **No Out-of-Port Generalizability:** TPS is a river-mouth gateway port serving East Java import/export cargo. Its operational patterns, equipment fleet, and customs clearance procedures cannot be assumed to represent transshipment megahubs (e.g. Singapore, Busan) or Western landlord ports (e.g. Los Angeles, Rotterdam).
2. **Temporal Volatility:** Training an ML model on historical data spanning a single acute disruption episode (July 2021) risks learning transient congestion correlations (e.g., that Tuesday discharges correlate with delays) that evaporate once normal berth operations resume.
3. **Recommendation for Multi-Terminal Rollout:** Before deploying predictive bottleneck models across an enterprise port network, models must be cross-validated across multiple distinct terminals with facility-specific operational metadata.
