# Error Analysis: False Negatives vs. False Positives

**Role:** Evaluation Engineer  
**Evaluated Split:** Quarantined `test.parquet` ($N=13,959$)

---

## 1. Error Counts & Distribution

### Strategy D: Calibrated Decision Rule ($\tau^* = 0.30$)
- **True Positives Caught:** 20 (Recall: 5.19%)
- **Missed Bottlenecks (False Negatives):** 365 (94.81% missed)
- **Wasted Alerts (False Positives):** 105 (Precision: 16.00%)
- **Total Test Cost:** $1,126,500.00 (Net Saving: +$28,500 over Do Nothing)

### Strategy C: Default Argmax Rule
- **True Positives Caught:** 67 (Recall: 17.40%)
- **Missed Bottlenecks (False Negatives):** 318 (82.60% missed)
- **Wasted Alerts (False Positives):** 2,222 (Precision: 2.93%)
- **Total Test Cost:** $1,620,600.00 (Net Loss: -$465,600 worse than Do Nothing)

---

## 2. Key Failure Patterns (Plain Language for Port Operators)
1. **Missed Critical Delays (False Negatives):**
   - The model regularly misses standard dry 20ft/40ft containers with ordinary cargo weights that look completely normal upon arrival.
   - **Root Cause:** In container logistics, the majority of severe dwell delays are caused by **in-yard events occurring 24–72 hours AFTER discharge**: customs inspection holds (`SP_RED_FLAG_PIB`), delivery order (`JOB_DEL_DOCTYPE`) processing stalls, or consignee payment disputes. None of these events are observable at vessel crane unstacking ($t_{\text{discharge}}$).
2. **False Alarms (False Positives):**
   - High alert scores are disproportionately assigned to containers with heavy gross weights or rare carrier codes.
   - However, many shippers of heavy industrial cargo maintain dedicated chassis fleets and pre-cleared customs documentation, clearing the gate within 48 hours despite their anomalous physical profile.
