# Logistics Operations Field Guide: Container Dwell Bottleneck Triage (Audited Pass)

**Audience:** Supply Chain Planners, Port Dispatchers, & Freight Operations Leads
**Model Version:** Stage 01 Exploratory Baseline (`XGBoost / LightGBM`)
**Prediction Horizon:** Container Discharge Commencement (`discharge_start_ts`)
**Status:** Exploratory Triage Reference Sheet (NOT for Autonomous Execution)

---

## 1. What the Model Predicts (and What It Does NOT)

- **What It Predicts:** An exploratory estimate of container dwell duration risk between discharge ($t_{\text{discharge}}$) and gate-out ($t_{\text{truck\_out}}$).
- **What It Does NOT Predict:**
  - It is **NOT** a pre-berthing or pre-arrival ETA model. In-yard slotting coordinates (`YARD_BLOCK`, `YARD_SLOT`) and discharge time are only assigned when discharge begins.
  - It does **NOT** diagnose root causes (customs holds, paperwork discrepancies, chassis shortages).
  - It is **NOT an autonomous rerouting directive.** A high-risk score is an early operational flag for human investigation, never an automatic trigger to divert cargo.

---

## 2. Target Classes & Operational Meaning

| Tier | Status | Frozen Training Cutoff | Operational Context & Terminal Meaning |
| :---: | :---: | :---: | :--- |
| **Class 0** | **🟢 ON_TIME** | $\le 65.11\text{ hours}$ ($\le 2.71\text{ days}$) | Nominal terminal turnaround. Standard demurrage free-time buffers intact. |
| **Class 1** | **🟡 AT_RISK** | $65.11\text{h} - 160.90\text{ hours}$ ($2.71 - 6.70\text{ days}$) | Moderate queue. Approaching free-time limits. Monitor drayage appointments and customs status. |
| **Class 2** | **🔴 CRITICAL_DELAY** | $> 160.8993\text{ hours}$ ($> 6.70\text{ days}$) | Severe terminal congestion / bottleneck. High probability of demurrage penalties and missed rail links. |

---

## 3. Interpreting Model Scores & The Missed-Delay Reality

When querying the prediction API (`POST /predict/container`), the service returns decision scores:
```json
{
  "case_id": "AAAU9001220-2021-06-28",
  "predicted_class_id": 1,
  "predicted_class_name": "AT_RISK",
  "probabilities": {
    "P_ON_TIME": 0.2841,
    "P_AT_RISK": 0.5422,
    "P_CRITICAL_DELAY": 0.1737
  },
  "alert_threshold_tau": 0.25,
  "alert_triggered": false
}
```

> [!WARNING]
> **CRITICAL REALITY: Model Misses ~88.8% of Critical Delays at Default Decision Rule**
> - Under standard argmax classification, the model achieves only **11.18% recall** on `CRITICAL_DELAY` cases.
> - **Raw scores are UNCALIBRATED soft tree scores**, not true empirical probabilities.
> - To catch bottlenecks, planners must configure an operational alert threshold ($\tau$) rather than relying on the default predicted class.

---

## 4. Operational Alert Threshold Trade-Off Matrix

Dispatchers should choose an alert threshold ($\tau$) matching their operational capacity:

| Alert Threshold ($\tau$) | Terminal Stays Flagged | False Alarm Share | Bottlenecks Caught (Recall) | Recommended Operational Playbook |
| :---: | :---: | :---: | :---: | :--- |
| **$\tau \ge 0.10$** | **86.5%** | 91.7% | **97.6%** | **Audit Mode:** Catches almost all delays, but floods terminal dispatchers with false alarms. |
| **$\tau \ge 0.20$** | **67.8%** | 91.3% | **79.7%** | **High Surveillance Mode:** Useful during major port congestion surges (e.g. typhoons or strikes). |
| **$\tau \ge 0.25$** | **47.5%** | 90.9% | **58.6%** | **Balanced Triage:** Catches ~59% of bottlenecks while screening out over half of inbound freight. |
| **$\tau \ge 0.35$** | **10.7%** | 89.9% | **14.7%** | **Targeted Escalation:** Flags top 10% highest-risk containers for priority chassis reservation. |
| **$\tau \ge 0.50$** | **0.4%** | 78.9% | **1.2%** | **Emergency Intervention:** Only flags containers with overwhelming delay signals. Misses 98.8% of delays. |

---

## 5. What Actions the Model Justifies (and Does NOT Justify)

### Permitted / Recommended Actions:
- **Prioritize Drayage Booking:** Proactively alert drayage carriers to schedule chassis for containers flagged under the selected $\tau$ threshold.
- **Pre-Clear Commercial Documentation:** Escalate bills of lading, phytosanitary certificates, and customs entries with freight forwarders prior to weekend yard shutdown.
- **Inland Warehouse Buffer Notification:** Alert downstream distribution centers to adjust labor shift scheduling if arrival is delayed by 3–5 days.

### Strictly FORBIDDEN Autonomous Actions:
- **DO NOT autonomously cancel bookings or reroute vessels.** Unnecessary emergency freight diversion wastes container slot capacity, incurs severe ocean carrier change-of-destination surcharges, and disrupts vessel stowage balances.
- **DO NOT rely on PortWatch external disruption flags.** External PortWatch weather indices are constant zero in the historical baseline extract; zero value reflects catalog coverage limitations, not proven absence of local storms.

---

## 6. Critical Data & Generalization Limitations

1. **Single-Terminal Dataset:** Trained exclusively on historical operations at a single Indonesian container terminal (Surabaya, June–September 2021). **Do not assume model weights generalize to European, Indian, or North American ports.**
2. **Right-Censored Evaluation Window:** Stays arriving after September 12, 2021 in historical logs are subject to right-censoring survivorship bias (uncompleted long stays were cut off at extract end).
3. **Provisional Yard Feature Coordinates:** `YARD_BLOCK` and `YARD_SLOT` are static case attributes in the Mendeley extract; whether they represent initial discharge stacks or post-reshuffle locations remains unverified.
4. **Absence of Mid-Stay Milestones:** At discharge start, subsequent physical customs inspections (`has_behandle_inspection`) and delivery orders (`JOB_DEL_DOCTYPE`) have not occurred. Dynamic updates during in-yard storage will be addressed in Stage 02 sequential modeling.
