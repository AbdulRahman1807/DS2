# Stage 01 Evaluation Executive Briefing (Level-01 One-Pager)

**Project:** AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks (Project 07)  
**Stage:** Stage 01 Machine Learning Baseline Triage  
**Auditor:** Evaluation Engineer (Independent)  
**Date:** 2026-10-30  

**Final Deployment Verdict:** ⛔ **NO-GO FOR AUTOMATED THRESHOLD ALERTS | ⚠️ QUALIFIED GO AS OPERATOR RANKING QUEUE (TOP 1% ONLY)**

---

## 1. Five Foundational Empirical Audit Numbers

1. **Validation Baseline Reproduction:**
   - Critical Delay Recall at default argmax is **11.18%** (172 / 1,538 delays caught). Misses **88.82%** of true bottlenecks. *(ML Engineer claim verified).*
2. **Predictive Signal Strength at Arrival ($t_{\text{discharge}}$):**
   - Test Critical PR-AUC is **0.0415** vs 0.0276 base rate $\to$ **1.50x Signal Lift** (Bootstrap 95% CI: **[1.20x, 2.01x]**). Statistically significant, but far too weak for autonomous operations.
3. **Top 1% Review Queue Precision:**
   - Top 1% precision reaches **15.00% (5.44x lift, 95% CI: [8.57%, 20.71%])** on full test and **26.37% (6.26x lift, 95% CI: [17.58%, 35.16%])** on 7-day follow-up. Top 5% CI lower bound drops below base rate; queueing is viable **strictly for Top 1% only**.
4. **Economic Impact of Calibrated Strategy D ($\tau^* = 0.30$):**
   - Full Test Net Saving vs Do-Nothing: **+\$28,500** (Bootstrap 95% CI: **[+\$3,285, +\$55,200]**).
   - 7-Day Follow-Up Net Saving: **+\$45,600** (Bootstrap 95% CI: **[+\$21,278, +\$71,100]**). Both exclude zero.
5. **Probability Miscalibration under Regime Shift:**
   - Quarantined test Critical ECE is **0.0607** $\to$ approximately **2.2x the test set critical prevalence (2.76%)**.

---

## 2. Three Critical Operational Limitations

1. **Assumed Cost Model:** Financial figures reflect operational assumptions ($C_{\text{FA}} = \$300$, $C_{\text{FN}} = \$3,000$ USD/container). True port charges must be verified by terminal management.
2. **Right-Censoring Lower Bound:** Test critical prevalence (2.76%) is an empirical lower bound because active stays lacking gate-out by Sept 30 were quarantined. Direction of metric bias is not assumed.
3. **Signal Ceiling at Crane Unstacking:** Manifest attributes known at discharge lack visibility into subsequent in-yard customs holds and delivery stalls where severe delays actually originate.

---

## 3. Team Huddle Decision Points: Acceptable False-Alarm Trade-Offs

To align operations before Stage 02 integration, the cross-functional squad must resolve:
1. **Option A (No Automated Alerts - Recommended Baseline):** Rely exclusively on human inspection of the daily Top-1% ranking queue (~7 containers/day, catching ~1.1 delays/day with ~6.2 false alarms).
2. **Option B (Top-1% Queue + Hybrid Tier Audit):** Add an audit tier for containers flagged by BOTH the calibrated classifier ($\ge 0.30$) AND the Isolation Forest outlier detector (3 containers in test, 1 true critical delay, 33.3% precision).
3. **Option C (Dispatcher Actionability Boundary):** Dispatchers must successfully resolve at least **56.8% of alerted delays** to break even against false alarm intervention costs ($6.25 \times \$300 = \$1,875$/day). Is terminal staffing capable of this intervention rate?

---

## 4. Recommended Next Steps

- **Data Engineer:** Change Data Contract to ingest mid-stay inspection milestones (+12h to +24h post-discharge); remove `YARD_BLOCK`/`YARD_SLOT` from arrival contract.
- **ML Engineer:** Transition from 3-class classification to continuous LightGBM Quantile Regression ($p10, p50, p90$) on continuous dwell hours.
- **Integration Engineer:** Implement the daily Top-1% sorting queue in `api.py` using `reports/evaluation/models/calibrator_isotonic.joblib`.
