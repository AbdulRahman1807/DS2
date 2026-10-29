# Logistics Ops Field Guide: Port-to-Warehouse Bottleneck Triage
*1-Page Operational Reference Sheet for Supply Chain Planners (Stage 01 Deliverable)*

---

## 1. Quick-Triage Decision Matrix

| Predicted Class | Estimated Dwell | Recommended Action | Operational Playbook |
| :--- | :--- | :--- | :--- |
| **🟢 ON_TIME** | $\le 50^{\text{th}}$ percentile | **Nominal Monitoring** | Standard rail/truck dispatch; safety-stock buffers intact. |
| **🟡 AT_RISK** | $50^{\text{th}} - 85^{\text{th}}$ percentile | **Priority Gate Clearance** | Request expedited customs pre-clearance; notify regional warehouse of $\pm 24\text{h}$ schedule buffer. |
| **🔴 CRITICAL_DELAY** | $> 85^{\text{th}}$ percentile | **Emergency Triage / Reroute** | Trigger alternate inland rail hub or secondary terminal reroute; escalate with ocean carrier. |

---

## 2. Top Bottleneck Pre-Warning Indicators

Planners should watch for these early-warning signals before discharge gridlock occurs:
1. **Quarantine / Customs Holds (`SP_RED_FLAG_PIB`, `HAS_QUARANTINE_FLAG`):**
   - High correlation with prolonged yard dwell.
   - Immediate escalation to customs broker required if hold exceeds 24 hours.
2. **7-Day Trailing Delay Rate Spike:**
   - When the terminal delay rate exceeds **35%**, inbound containers experience compounding queue times.
3. **Extreme Weather / Regional Disruption (PortWatch Alert Level: RED):**
   - Pre-emptively hold downstream truck bookings to avoid detention and demurrage penalties.

---

## 3. False-Alarm Economic Trade-Offs

> **Planner Caution:** Triggering an unnecessary emergency reroute costs approximately **$1,000–$1,500** per container in carrier surcharges.  
> Only execute emergency reroutes when the **Critical Delay probability exceeds the squad-calibrated operating threshold ($\tau$)**.

---

## 4. API Operations Cheat Sheet

- **Endpoint:** `POST /predict/container`
- **Expected P95 Latency:** $< 25\text{ms}$
- **Response Format:**
```json
{
  "case_id": "AAAU9001220-2021-06-28",
  "predicted_class": "AT_RISK",
  "class_probabilities": {
    "ON_TIME": 0.18,
    "AT_RISK": 0.64,
    "CRITICAL_DELAY": 0.18
  },
  "is_anomaly": false,
  "recommended_action": "Request expedited customs pre-clearance"
}
```
