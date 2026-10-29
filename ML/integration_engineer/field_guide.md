# Logistics Operations Field Guide: Container Dwell Bottleneck Triage

**Audience:** Supply Chain Planners, Port Dispatchers, & Freight Operations Leads
**Model Version:** Stage 01 Supervised Baseline (`XGBoost / LightGBM`)
**Prediction Horizon:** Container Discharge Commencement (`discharge_start_ts`)
**Scope:** One-Page Operational Reference Sheet (Stage 01 Deliverable)

---

## 1. Quick-Triage Decision Matrix

| Predicted Class | Frozen Training Dwell Threshold | Recommended Action | Operational Playbook |
| :--- | :--- | :--- | :--- |
| **🟢 ON_TIME** | $\le 65.11\text{ hours}$ ($\le 2.71\text{ days}$) | **Nominal Monitoring** | Standard rail/truck drayage dispatch; standard demurrage free-time buffers intact. |
| **🟡 AT_RISK** | $65.11\text{h} - 160.90\text{ hours}$ ($2.71 - 6.70\text{ days}$) | **Priority Gate Clearance** | Pre-clear documentation with customs broker; proactively verify drayage appointment slots. |
| **🔴 CRITICAL_DELAY** | $> 160.90\text{ hours}$ ($> 6.70\text{ days}$) | **Escalated Intervention** | Alert downstream warehouse of schedule buffer; prioritize yard unstacking; investigate commercial holds. |

---

## 2. Top Bottleneck Early Indicators at Discharge

Planners should monitor these empirical signals available at container discharge ($t_{\text{discharge}}$):
1. **Discharge Day of Week:** Thursday and Friday discharges encounter severe weekend drayage restrictions and customs office closures.
2. **Container Gross Weight (`GROSS`):** Heavier units experience prolonged yard dwell due to specialized multi-axle chassis availability and road axle limits.
3. **Specialized Container Types (`CTR_TYPE`):** Reefer, Open-Top, and Tank containers require specialized plug-in bays, inspection ramps, or quarantine.
4. **Spatial Yard Location (`YARD_BLOCK`, `YARD_SLOT`):** Deep-stack terminal blocks require additional crane re-handling moves compared to high-turnover buffer slots.
5. **7-Day Trailing Terminal Delay Rate:** When rolling terminal delay rate exceeds 30%, systemic yard congestion compounds queue times.

---

## 3. Operational Trade-Offs & Threshold Setting

> [!WARNING]
> **No Universal Cost Ratio:** Triggering an unnecessary emergency freight diversion wastes container capacity and incurs substantial carrier surcharges.
> The alerting threshold ($\tau$) must be agreed with logistics operations based on the enterprise cost of an unnecessary inspection versus the demurrage penalty of a missed bottleneck. Do not treat model probabilities as autonomous rerouting directives.

---

## 4. API Operations Cheat Sheet

- **Endpoint:** `POST /predict/container`
- **Expected P95 Latency:** $< 25\text{ms}$
- **Response Format:**
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
  "is_anomaly": false,
  "alert_triggered": true,
  "recommended_action": "Proactively verify drayage appointment slots and bill of lading clearance"
}
```
