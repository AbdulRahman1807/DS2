# Stage 02 Data Requirements & Acquisition Specification

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
**Repository:** `AbdulRahman1807/DS2`
**Author:** Senior Machine Learning & Logistics Data Engineering
**Scope:** Data Acquisition Priorities, Operational Field Instrumentation, and Leakage Specifications
**Status:** Approved Technical Requirements Document
**Artifact Directory:** `reports/ml/`
**Prediction Design Reference:** [`reports/ml/stage02_prediction_design.md`](file:///Users/abdul/Desktop/DS/reports/ml/stage02_prediction_design.md)

---

## 1. Executive Context: Moving Beyond Static Manifests

The Stage 01 audit established that static manifest attributes (carrier prefix, container dimensions, gross weight) available at $t_{\text{discharge}}$ cannot differentiate normal container dwell from severe administrative bottlenecks.

To empower classical machine learning models (LightGBM, Random Forest, discrete-time hazards) in Stage 02, the data pipeline must ingest time-valid operational signals across terminal operations:
1. **Existing Repository Signals (Immediate Extraction):** Signals that exist inside the current Mendeley event log extract but were quarantined in Stage 01 to prevent post-arrival leakage at $t_{\text{discharge}}$. These become valid at post-discharge landmarks ($t_{\text{discharge}} + 12\text{h}$ and $+24\text{h}$).
2. **External Operational Signals (Instrumentation Specification):** Telemetry from terminal operating systems (TOS), crane SCADA, optical character recognition (OCR) gate lanes, and customs EDI that must be collected from terminal IT systems.

---

## 2. Comprehensive Prioritized Data Acquisition Specification

The following specification details all operational signals, prioritized by their expected impact on bottleneck classification.

| Priority | Signal Name | Operational Meaning | Source System | Required Timestamp Format | Current Status in Repo | Safe at Stage 02 Prediction Horizon? | Expected Predictive Relevance | Integration Complexity | Potential Leakage Risk & Safeguards |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **P0** | **`has_job_del_doctype_by_landmark`** | Binary indicator that the delivery order release was logged by shipping line. | Terminal Event Log (`cleaned_event_log.parquet`) | UTC ISO-8601 (`YYYY-MM-DD HH:MM:SS`) | **Available in raw event log** | **YES (at landmark $L_{12}$ and $L_{24}$)** | **CRITICAL (High):** An unreleased DO at 24h indicates consignee payment dispute or document hold. | Low (Already in repo) | **HIGH LEAKAGE RISK:** Must strictly verify `event_timestamp <= landmark_timestamp`. Any DO release after landmark is leakage. |
| **P0** | **`elapsed_hours_to_do_release`** | Elapsed time (hours) from discharge to delivery order issuance. | Terminal Event Log (`activity = 'JOB_DEL_DOCTYPE'`) | Derived numeric float | **Available in raw event log** | **YES (at landmark)** | **CRITICAL (High):** Quantitative speed of administrative release. | Low (Derivable) | If DO released after landmark, value must be set to `UNRELEASED` (NaN / sentinel -1.0). |
| **P0** | **`in_yard_moves_count_by_landmark`** | Number of yard crane reshuffle or repositioning moves logged prior to landmark. | Terminal Event Log (`activity` count) | Event timestamp log | **Available in raw event log** | **YES (at landmark)** | **MODERATE / HIGH:** Stagnant containers with 0 moves vs containers moved to inspection pads. | Low (Already in repo) | Strict timestamp upper bound $\le t_{\text{discharge}} + \Delta t$. |
| **P0** | **Arrival-Cohort Follow-Up Stays** | Tracking of all containers discharged in a calendar window with minimum 90-day follow-up. | Data Engineering Extraction Protocol | Container discharge window $[t_{\text{start}}, t_{\text{end}}]$ | **NOT IN CURRENT REPO** (Current extract filtered on departure date) | DE Pipeline Protocol | **CRITICAL (High):** Permanently cures the 85.4% right-censoring survivorship bias in test data. | Moderate (Requires DE re-extract) | Zero leakage risk; fundamental data hygiene improvement. |
| **P1** | **`dynamic_yard_block_density`** | Real-time TEU utilization percentage of the assigned yard block at container placement. | Terminal Operating System (TOS Yard Management) | TOS point-in-time snapshot at crane set-down | **NOT IN CURRENT REPO** (Current extract has static `YARD_BLOCK` string) | **YES (at discharge and landmarks)** | **HIGH:** High block density ($>85\%$) causes exponential reshuffle delays during truck pickup. | Moderate (TOS integration required) | Must record *initial drop-off* block density, not final reshuffle block density. |
| **P1** | **`average_stack_height`** | Average container stacking tier (e.g. 2-high vs 5-high) in the assigned storage bay. | TOS Yard Stacking Engine | Bay telemetry at set-down | **NOT IN CURRENT REPO** | **YES (at discharge and landmarks)** | **HIGH:** Deeply stacked bottom containers require 3 to 4 non-productive crane shifts to retrieve. | Moderate (TOS integration) | Must represent stacking tier at time of set-down. |
| **P1** | **`reefer_plug_saturation`** | Percentage of electrical monitoring plug-in points occupied in the refrigerated container yard. | TOS Reefer Subsystem | Sensor log at set-down | **NOT IN CURRENT REPO** | **YES (at discharge and landmarks)** | **HIGH (for Reefers):** Explains why reefers delay when specialized plugs are saturated. | Low (TOS reefer module) | None if point-in-time logged. |
| **P1** | **`customs_channel_assignment`** | Green channel (instant electronic release) vs Yellow (document review) vs Red (physical inspection). | Indonesian Customs EDI (`CEISA` / `PPJK` interface) | Timestamp of customs declaration clearance | **NOT IN CURRENT REPO** | **YES (at landmarks when filed)** | **CRITICAL (Very High):** Red-channel cargo incurs 5–10 days dwell for physical container stripping. | High (External government EDI integration) | Only usable after declaration is filed. Missing before filing must be treated as `PENDING`. |
| **P2** | **`gross_crane_moves_per_hour`** | Hourly move rate of quay cranes discharging the vessel call. | Crane SCADA / Telemetry | Hourly aggregate preceding container lift | **NOT IN CURRENT REPO** | **YES (at discharge and landmarks)** | **MODERATE:** Terminal productivity proxy; low rates indicate mechanical or apron congestion. | Moderate (SCADA logs) | Must use trailing move rates prior to container discharge, not whole-call average. |
| **P2** | **`gate_truck_turnaround_time_4h`**| Trailing 4-hour median road truck gate-in to gate-out turnaround time. | OCR Gate System / Weighbridge | Trailing $[t-4\text{h}, t-15\text{m}]$ | **NOT IN CURRENT REPO** | **YES (at discharge and landmarks)** | **MODERATE:** Direct measure of external trucking bottlenecks and drayage gate queues. | Moderate (Gate OCR logs) | Must strictly exclude gate activity occurring after container discharge/landmark. |
| **P2** | **`drayage_appointment_queue`** | Total number of drayage truck gate appointments booked for the container's shipping line today. | Truck Appointment System (TAS) | Booking schedule snapshot | **NOT IN CURRENT REPO** | **YES (at discharge and landmarks)** | **MODERATE:** Predicts whether trucking companies have scheduled collection. | High (TAS integration) | Must use bookings made prior to landmark timestamp. |

---

## 3. Immediate In-Repo Feature Engineering for Stage 02 (Zero External Dependencies)

Before requesting external IT integrations, Stage 02 will exploit event telemetry already present in the raw event log (`data/interim/cleaned_event_log.parquet`) that was legally unobservable at $t_{\text{discharge}}$ but is fully legal at landmark horizons ($L_{12}$ and $L_{24}$):

```
+-----------------------------------------------------------------------------------------------+
|                      IN-REPO EVENT LOG MILESTONE EXTRACTION AT LANDMARK L                     |
+---------------------------------------------------+-------------------------------------------+
| Event Attribute                                   | Landmark Extraction Logic                 |
+---------------------------------------------------+-------------------------------------------+
| Delivery Order Release (JOB_DEL_DOCTYPE)          | event_timestamp <= t_discharge + 12h      |
| Customs Inspection Pad Transfer (BEHANDLE)        | event_timestamp <= t_discharge + 12h      |
| Cumulative Yard Reshuffle Count                   | count(activities) where ts <= t_discharge + 12h |
| Elapsed Time from Discharge to First Milestone    | min(ts) - t_discharge                     |
| Stagnation Flag (Zero events in first 12/24h)     | count(activities) == 0                    |
+---------------------------------------------------+-------------------------------------------+
```

### Feature Engineering Specifications:
1. **`has_delivery_order_at_12h`:** Binary indicator (0/1). True if any `JOB_DEL_DOCTYPE` activity has `Start Timestamp` $\le t_{\text{discharge\_start}} + 12.0\text{h}$.
2. **`has_delivery_order_at_24h`:** Binary indicator (0/1). True if any `JOB_DEL_DOCTYPE` activity has `Start Timestamp` $\le t_{\text{discharge\_start}} + 24.0\text{h}$.
3. **`hours_to_do_release_at_landmark`:** Continuous elapsed duration from discharge to DO release. If unreleased at landmark, encoded as $-1.0$ with a companion indicator `is_do_missing_at_landmark = 1`.
4. **`early_yard_activity_count`:** Integer count of all intermediate handling moves logged between $t_{\text{discharge}}$ and $t_L$.
5. **`is_completely_stagnant_at_landmark`:** Binary flag indicating that zero events or scans have been logged for this container since crane set-down.

---

## 4. External Data Acquisition Roadmap (Terminal IT Integrations)

For Phase 2B and beyond, external integrations should proceed in three structured phases:

### Phase 1: Terminal Operating System (TOS) Database Extract (Internal Port IT)
- **Target Systems:** CyberLogitec OPUS Terminal, Navis N4, or local TPS TOS database.
- **Data Extracted:**
  - Initial yard block and slot allocation at crane set-down.
  - Yard bay utilization and stacking height at drop-off time.
  - Reefer plug telemetry logs.
- **Estimated Integration Effort:** 3 to 4 weeks (SQL query engineering against historical TOS mirror).

### Phase 2: Gate System & SCADA Logs (Terminal Engineering)
- **Target Systems:** OCR Gate Camera Server, Weighbridge automated logging, Quay Crane SCADA telemetry.
- **Data Extracted:**
  - Trailing gate queue turnaround times.
  - Quay crane gross moves per hour (GMPH).
- **Estimated Integration Effort:** 4 to 6 weeks.

### Phase 3: Customs & External Port Community Systems (Government EDI)
- **Target Systems:** Indonesian Customs `CEISA` / National Single Window (INSW).
- **Data Extracted:**
  - Import declaration (`PPJK`) filing timestamp.
  - Channel clearance decision timestamp (Red/Yellow/Green).
  - Physical inspection quarantine holds (`Behandle`).
- **Estimated Integration Effort:** 8 to 12 weeks (Requires formal agency data sharing agreements).

---

## 5. Strict Data Hygiene & Anti-Leakage Protocol

Any candidate operational feature must pass the three-part **Temporal Validity Gate** before ingestion into Stage 02 feature matrices:

1. **Existence Test:** Did the underlying physical event occur before the prediction landmark timestamp?
   $$\text{event\_end\_ts} \le t_{\text{landmark}}$$
2. **Recording Test:** Was the event logged in the database before the prediction landmark timestamp?
   $$\text{database\_inserted\_ts} \le t_{\text{landmark}}$$
3. **Imputation Test:** If the milestone has not occurred by the landmark, is it represented using a clean absence indicator rather than a forward-looking default?
   - Example: Missing `JOB_DEL_DOCTYPE` must NOT be imputed with the container's future DO release date. It must be encoded as `DO_NOT_YET_ISSUED`.
