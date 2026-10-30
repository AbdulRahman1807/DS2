# Data Leakage, Observability, and Censoring Audit Report

**Role:** Evaluation Engineer  
**Date:** 2026-10-30  
**Evaluated Splits:** `train.parquet` (60,698 stays), `val.parquet` (20,879 stays), `test.parquet` (13,959 stays)

---

## 1. Feature Contracts, Target Isolation & Dataset Provenance
- **Dataset Provenance:** Real operational container terminal event log from Pelabuhan Tanjung Perak (Surabaya, Indonesia), source as recorded in `dataset_manifest.json` (`Dataset of Dwelling Time Process Fix.xlsx` / `mendeley_event_log.parquet`). Right-censored stays lacking TRUCK_OUT gate-out records were quarantined.
- **Target Isolation:** Verified. None of `{'dwell_days', 'truck_out_end_ts', 'target_class_name', 'dwell_hours', 'target_class'}` exist in active predictor set $X$.
- **Post-Arrival Milestone Isolation:** Verified. Post-discharge events (`JOB_DEL_DOCTYPE`, `has_behandle_inspection`, `total_events_in_stay`) are strictly quarantined.
- **Entity Disjointness:** Verified. Container ID overlap across all split pairs is **0**.
  - Train $\cap$ Val containers: `0`
  - Train $\cap$ Test containers: `0`
  - Val $\cap$ Test containers: `0`

---

## 2. Yard Coordinate Observability & Multi-Split Ablation (`YARD_BLOCK`, `YARD_SLOT`)
- **Observability Audit:** In `cleaner.py`, `YARD_BLOCK` and `YARD_SLOT` are extracted from any non-null event in the stay. In physical port operations, containers undergo multiple yard reshuffles before road gate-out.
- **Multi-Split Paired Bootstrap Ablation (1,000 resamples):**
  - **Validation Split ($N=20,879$):**
    - Full Model (With Yard): Critical PR-AUC = **0.0934**
    - Conservative Model (No Yard): Critical PR-AUC = **0.0906**
    - Difference $\Delta (\text{NoYard} - \text{Full}) = -0.0028$ (95% Paired CI: [-0.0055, -0.0001])
  - **Quarantined Test Split ($N=13,959$):**
    - Full Model (With Yard): Critical PR-AUC = **0.0415**
    - Conservative Model (No Yard): Critical PR-AUC = **0.0473**
    - Difference $\Delta (\text{NoYard} - \text{Full}) = +0.0059$ (95% Paired CI: [+0.0008, +0.0112])
- **Interpretation & Engineering Recommendation:**
  - **Performance Evidence is Mixed:** On validation data (closer in time to training), yard coordinates slightly helped ($\Delta = -0.0028$). On out-of-time test data, the conservative model without yard coordinates achieved higher PR-AUC ($\Delta = +0.0059$, 95% CI $[+0.0008, +0.0112]$). Because performance deltas exhibit opposite signs across splits, statistical performance alone does not decide the issue.
  - **Observability at Prediction Time ($t_{\text{discharge}}$):** Because physical yard reshuffles occur hours or days after unstacking, final yard coordinates are not reliably known at $t_{\text{discharge}}$. We therefore **recommend to the Data Engineer** that `YARD_BLOCK` and `YARD_SLOT` be removed from the production feature contract on observability grounds.
  - *Hypothesis:* The test improvement without yard features may reflect avoiding overfitting to stale spatial coordinates, but this remains an unconfirmed hypothesis.

---

## 3. Right-Censoring Mechanism & Pre-Extracted Delay Loss
- **Data Engineering Mechanism:** In `cleaner.py` line 65, any stay with `DISCHARGE` that lacks a completed `TRUCK_OUT` is classified as incomplete and **quarantined**.
- **Censoring Impact on Test Split:**
  - Overall Test Prevalence: **2.76%** (385 stays)
  - 7-Day Observable Follow-Up Subset ($N=9,087$): Prevalence rises to **4.21%** (383 stays).
  - Out of 385 total critical delays in test, **383 occurred before September 24**. Stays arriving in the final week with prolonged delays were dropped by quarantine.
- **Formal Caveats:**
  > [!IMPORTANT]
  > 1. Test critical delay prevalence is an **empirical lower bound** on true terminal delay prevalence. The direction of metric bias is not assumed.
  > 2. The **15.00% train prevalence was fixed by construction** (by defining the threshold as the empirical 85th percentile cutoff on training data).
- **Separation of Censoring vs. Genuine Regime Drift:**
  - Train critical prevalence is **15.00%**, Val is **7.37%**. Because the validation window (Aug 27 – Sept 11) had over 19 days of follow-up before the Sept 30 extract cutoff, this drop reflects a **genuine macro clearance regime shift** (July terminal congestion crisis vs late August recovery), supported by the empirical weekly dwell trend curve ([`reports/evaluation/weekly_dwell_trend.png`](file://reports/evaluation/weekly_dwell_trend.png)).

---

## 4. Open Questions for Data Engineering
1. Can initial crane unstacking block coordinates be isolated from post-reshuffle storage coordinates in the TOS event log?
2. In future releases, can incomplete stays be right-censored with survival analysis labels rather than discarded entirely?
