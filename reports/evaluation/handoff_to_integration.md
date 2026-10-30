# Handoff to Integration Engineer: Decision Rule, Artifacts & Service Integration

**Role:** Evaluation Engineer  
**Target Recipient:** Integration Engineer (`ML/integration_engineer/api.py`)  
**Final Deployment Verdict:** ⛔ **NO-GO FOR AUTOMATED THRESHOLD ALERTS | ⚠️ QUALIFIED GO AS OPERATOR RANKING QUEUE (TOP 1% ONLY)**

---

## 1. Executive Deployment Directive
1. **NO Automated Threshold Alerts / Autonomous Rerouting:**  
   Automated threshold alerting is an **ABSOLUTE NO-GO**. In live operations, automated alerting creates 85%–97% false alarms depending on the threshold (even at $\tau^* = 0.30$, 84% of alerts are false alarms and misses 95% of delays), inducing severe operational friction.
2. **Advisory Decision Support Only:**  
   Model predictions serve **strictly as review recommendations** to prioritize terminal dispatchers' daily visual inspection queue. Model outputs **NEVER authorize autonomous container rerouting, demurrage waivers, or off-dock drayage without human sign-off**.
3. **Calibrated Probabilities:**  
   Raw XGBoost outputs are unbalanced tree partition scores, NOT probabilities. The service must pass features through the multi-class Isotonic Calibrator fitted on `val_A`.

---

## 2. Serialized Production Artifacts & Cryptographic Provenance

| Artifact Role | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Supervised Model Pipeline** | `reports/evaluation/models/xgboost_weighted.joblib` | `b7c77bc50ecdd8a3c5e216cd281de9ba8aba0910f741e4e48b768aa90170c543` |
| **Isotonic Calibrator** | `reports/evaluation/models/calibrator_isotonic.joblib` | `aa77cfabd26719573bcd3c18f1af1f8bba7c9769a230f4bc53e73aaacf05c582` |
| **Reference Baseline Dataset** | `data/processed/dataset_manifest.json` | `975000a5b66a7ccdfcbdb78b6150a0693106158df4825b2bb4ba17450cac3614` |
| **Evaluated Threshold (Reference)** | $\tau^* = 0.30$ (Split-Val out-of-sample) | Bayes Theoretical: 0.0909 |

---

## 3. Human-in-the-Loop Daily Top-1% Ranking Protocol

Rather than comparing scores against a fixed scalar threshold $\tau^*$, the daily integration workflow operates as follows:

1. **Daily Batch Accumulation:** At the start of each operational shift (or rolling 24-hour cycle), accumulate all $N$ container arrivals at crane unstacking ($t_{\text{discharge}}$).
2. **Calibrated Scoring:** Generate calibrated probability $P(\text{CRITICAL\_DELAY})$ for each container.
3. **Cutoff Computation:** Compute the queue capacity $k = \lceil 0.01 \times N \rceil$. For typical terminal throughput ($N \approx 735$ containers/day), $k = \lceil 7.35 \rceil = 8$ containers.
4. **Stable Descending Sort & Tie-Handling:** Sort descending by $P(\text{CRITICAL\_DELAY})$ using a stable mergesort. If ties occur at the boundary, break ties using earliest `discharge_start_ts`.
5. **Dispatcher Review Queue:** Route only the top $k$ containers to the dispatcher review console.

---

## 4. Production Integration Code Snippet for `api.py`

```python
"""Production Serving Module for Container Bottleneck Ranking Queue."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Load serialized pipeline and calibrator
MODEL_PATH = Path("reports/evaluation/models/xgboost_weighted.joblib")
CALIB_PATH = Path("reports/evaluation/models/calibrator_isotonic.joblib")

artifact = joblib.load(MODEL_PATH)
preprocessor = artifact["preprocessor"]
model = artifact["model"]
calibrator = joblib.load(CALIB_PATH)

APPROVED_INPUT_FEATURES = [
    "gross_weight_kg",
    "gross_weight_kg_is_missing",
    "is_reefer",
    "is_tank",
    "is_empty",
    "carrier_code",
    "carrier_code_is_missing",
    "consignee_id",
    "consignee_id_is_missing",
    "vessel_name",
    "vessel_name_is_missing",
    "discharge_hour",
    "discharge_day_of_week",
    "trailing_7d_volume",
    "historical_delay_rate_7d",
]

def score_and_rank_daily_arrivals(arrivals_df: pd.DataFrame) -> pd.DataFrame:
    """Scores daily container arrivals and tags the top-1% operator inspection queue.
    
    Args:
        arrivals_df: DataFrame with container arrivals containing APPROVED_INPUT_FEATURES.
    Returns:
        DataFrame sorted by bottleneck priority with calibrated probabilities and queue flags.
    """
    assert all(col in arrivals_df.columns for col in APPROVED_INPUT_FEATURES), "Missing required features"
    
    X_proc = preprocessor.transform(arrivals_df[APPROVED_INPUT_FEATURES])
    cal_probs = calibrator.predict_proba(X_proc)
    
    out_df = arrivals_df.copy()
    out_df["p_on_time"] = np.round(cal_probs[:, 0], 4)
    out_df["p_at_risk"] = np.round(cal_probs[:, 1], 4)
    out_df["p_critical_delay"] = np.round(cal_probs[:, 2], 4)
    
    # Stable descending sort by critical delay risk
    out_df = out_df.sort_values(by=["p_critical_delay", "discharge_start_ts"], ascending=[False, True]).reset_index(drop=True)
    
    # Select top 1% queue
    n_total = len(out_df)
    k_queue = max(1, int(np.ceil(0.01 * n_total)))
    out_df["in_top_1pct_review_queue"] = False
    out_df.loc[:k_queue - 1, "in_top_1pct_review_queue"] = True
    
    out_df["operational_action"] = np.where(
        out_df["in_top_1pct_review_queue"],
        "DISPATCHER_PHYSICAL_REVIEW",
        "STANDARD_TERMINAL_FLOW"
    )
    return out_df
```

---

## 5. Telemetry & Ongoing Monitoring Requirements
The Integration Engineer must expose the following telemetry metrics to Prometheus/Grafana:
1. **Weekly Critical Delay Prevalence:** Alert if trailing 7-day prevalence drops below 2.0% or exceeds 10.0% (prevalence shift trigger).
2. **Calibration Drift / Brier Score Tracking:** Compute weekly multi-class Brier score on completed stays; trigger recalibration if Brier score > 0.45.
3. **Queue True Positive Rate:** Track retrospective 7-day dwell outcomes for containers placed in the top-1% review queue; trigger ML alert if precision drops below 10.0%.
4. **Daily Queue Volume:** Confirm daily review volume stays at $\sim 7$ to $8$ containers/day.
