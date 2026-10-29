# Logistics Operations Field Guide: Container Dwell Bottleneck Triage

**Audience:** Supply Chain Planners, Port Dispatchers, & Freight Operations Leads
**Model Version:** Stage 01 Supervised Baseline (`XGBoost / LightGBM`)
**Prediction Horizon:** Container Discharge Commencement (`discharge_start_ts`)
**Scope:** One-Page Operational Reference Sheet

---

## 1. What the Model Predicts (and What It Does NOT)

- **What It Predicts:** The model estimates the probability that an inbound ocean container will experience normal, moderate, or severe yard dwell time between container discharge ($t_{\text{discharge}}$) and completed gate-out ($t_{\text{truck\_out}}$).
- **What It Does NOT Predict:**
  - It is **NOT** a pre-berthing or pre-arrival ETA model. In-yard slotting coordinates (`YARD_BLOCK`, `YARD_SLOT`) and discharge time are only assigned when the vessel commences discharge.
  - It does **NOT** predict the root cause (customs inspection, physical damage, payment dispute, or chassis shortage).
  - It is **NOT an autonomous rerouting command.** A predicted critical delay is a risk notification for human planner investigation, not an automatic directive to divert freight.

---

## 2. Target Classes & Operational Meaning

The model classifies containers into three operational tiers based on thresholds frozen from historical baseline training data:

| Tier | Status | Dwell Duration Threshold | Operational Context & Terminal Meaning |
| :---: | :---: | :---: | :--- |
| **Class 0** | **🟢 ON_TIME** | $\le 65.11\text{ hours}$ ($\le 2.71\text{ days}$) | Nominal terminal turnaround. Standard terminal free-time buffers intact. Standard rail/truck drayage dispatch. |
| **Class 1** | **🟡 AT_RISK** | $65.11\text{h} - 160.90\text{ hours}$ ($2.71 - 6.70\text{ days}$) | Moderate queue. Approaching carrier demurrage limits. Proactively monitor drayage appointment slots and commercial clearance. |
| **Class 2** | **🔴 CRITICAL_DELAY** | $> 160.90\text{ hours}$ ($> 6.70\text{ days}$) | Severe terminal congestion / bottleneck. High probability of compounding demurrage penalties and missed inland connections. |

---

## 3. Interpreting Output Scores & Alerts

When querying the prediction API (`POST /predict/container`), the service returns class scores:
```json
{
  "predicted_class_id": 2,
  "predicted_class_name": "CRITICAL_DELAY",
  "probabilities": {
    "P_ON_TIME": 0.2841,
    "P_AT_RISK": 0.3215,
    "P_CRITICAL_DELAY": 0.3944
  },
  "alert_triggered": true
}
```

> [!WARNING]
> **Probabilities are Soft Scoring Estimates, Not Guaranteed Frequencies:**
> While model probabilities sum to 1.0, they reflect empirical tree partition scores. Under out-of-time distribution shifts, raw scores require operational calibration. Treat $P(\text{CRITICAL\_DELAY}) > 0.25$ as a high-risk operational outlier (since baseline critical prevalence is only $\sim 7.4\%$).

---

## 4. What Actions the Model Justifies (and Does NOT Justify)

### Permitted / Recommended Actions:
- **Prioritize Drayage Booking:** Proactively alert local drayage carriers to schedule chassis for containers flagged as `AT_RISK` or `CRITICAL_DELAY`.
- **Pre-Clear Documentation:** Escalate bills of lading, phytosanitary manifests, and customs entries with freight forwarders prior to weekend yard closure.
- **Inland Warehouse Buffer Notification:** Alert downstream distribution centers to adjust labor shift scheduling if arrival is delayed by 3–5 days.

### Strictly FORBIDDEN Autonomous Actions:
- **DO NOT autonomously cancel bookings or reroute vessels.** Unnecessary emergency freight diversion wastes container slot capacity, incurs severe ocean carrier change-of-destination surcharges, and disrupts vessel stowage balances.
- **DO NOT rely on PortWatch external disruption flags.** External PortWatch weather indices are constant zero in the historical baseline extract; zero value does not prove absence of local rain or squalls.

---

## 5. False-Alarm vs. Missed-Delay Economic Trade-Offs

- **The Trade-off:**
  - **Prioritizing Recall (Catching Delays):** Catches $\sim 11.2\%$ of true bottlenecks at discharge start, but causes false alarms on some on-time units.
  - **Prioritizing Precision (Avoiding False Alarms):** Minimizes dispatcher distraction, but misses emerging yard gridlock.
- **No Inherent Optimal Threshold:** The exact alerting threshold ($\tau$) cannot be mathematically dictated by ML engineers without operational consensus. Logistics managers must establish the enterprise cost of an unnecessary yard check versus the detention/demurrage penalty of a missed bottleneck.

---

## 6. Critical Data & Generalization Limitations

1. **Single-Terminal Dataset:** Trained exclusively on historical operations at a single Indonesian container terminal (Surabaya, June–September 2021). **Do not assume identical model weights generalize to European, Indian, or North American transshipment ports.**
2. **Right-Censored Evaluation Window:** Stays arriving after September 12, 2021 in historical logs are subject to right-censoring survivorship bias (uncompleted long stays were cut off at extract end).
3. **Absence of Mid-Stay Milestones:** At discharge start, subsequent physical customs inspections (`has_behandle_inspection`) and delivery orders (`JOB_DEL_DOCTYPE`) have not occurred. Dynamic updates during in-yard storage will be addressed in Stage 02 sequential modeling.
