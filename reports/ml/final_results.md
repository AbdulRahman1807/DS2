# Final Results

**Task:** three-class dwell prediction at Terminal Petikemas Surabaya (on time ≤ 65.1 h, at risk 65.1–160.9 h, critical > 160.9 h).
**Evaluation:** three time-ordered validation folds. Metrics are pooled unless marked as mean-fold. The quarantined holdout has not been scored.

## Real TPS stays

| Prediction point | Experiment | Macro-F1 (mean-fold) | Accuracy | Balanced acc. | Critical P / R / F1 |
|---|---|---|---|---|---|
| At discharge | V-A-YARD-MANIFEST-ASOF | 0.413 | 0.458 | 0.483 | 0.158 / 0.474 / 0.237 |
| +12 h | R-B12-ASOF | 0.479 | 0.540 | 0.521 | 0.193 / 0.471 / 0.274 |
| +24 h | P-B24-EXT-STAGE-LGBM-XGB | 0.529 | 0.582 | 0.546 | 0.254 / 0.442 / 0.322 |
| +48 h | V-B48-SEQ-OOF-YARD-MANIFEST-ASOF | 0.545 | 0.549 | 0.606 | 0.335 / 0.623 / 0.435 |
| Delivery-job creation | R-E-JOBDEL | 0.826 | see registry | see registry | see registry |

At delivery-job creation, median warning is about 12.6 h and elapsed time already settles many stays; on stays with all three classes still possible, Macro-F1 is 0.477.

## Hybrid training

Real stays were augmented with 60,000 synthetic training stays because operational data was unavailable. Scores on real stays: 0.414, 0.465, 0.518 and 0.536 at discharge, +12, +24 and +48 h.

## Projection with process data

Customs filing, lane, delivery-order, payment and truck-booking fields were simulated from each stay's later events using assumed lead times (`simulate_process_data.py`). This estimates the value of obtaining that data; it is not performance on available data.

| Prediction point | Macro-F1 | Critical recall |
|---|---|---|
| At discharge | 0.535 | 53% |
| +12 h | 0.597 | 54% |
| +24 h | 0.624 | 60% |
| +48 h | 0.696 | 68% |
