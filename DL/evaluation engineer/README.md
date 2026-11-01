# Evaluation Engineer Phase — Rigorous Metric Audit & Backtesting

**Phase:** Phase 2 — Model Evaluation, Backtesting & Benchmark Verification  
**Role:** Senior Evaluation Engineer  
**Status:** **AWAITING ML MODEL TRAINING RUNS**  

---

## 1. Scope & Objective

The Evaluation Engineer is responsible for independently evaluating all trained model checkpoints against strict econometric and supply chain criteria.

### Key Responsibilities:
1. **Multi-Horizon Metric Verification:**
   - Evaluates MAE, RMSE, WAPE (Weighted Absolute Percentage Error), and Pinball / Quantile loss ($P_{10}, P_{50}, P_{90}$) across the 7-day forecast horizon.
2. **Quarantined Test Set Unblinding:**
   - Unblinds the strictly quarantined Test partition (**2025 calendar year**, 741,335 sequences) only after models are frozen and hyperparameter tuning is complete.
3. **Cross-Resolution Validation:**
   - Backtests whether predicted 7-day vessel call spikes correlate with observed monthly AIS dwell expansions (`monthly_dwell_2015_2025.csv`) and weekly ADB turnaround times (`adb_port_traffic_weekly.parquet`).
4. **Bottleneck Regime Classification Audit:**
   - Evaluates binary bottleneck detection (Focal Loss, Precision-Recall AUC) when capacity pressure exceeds design thresholds ($\text{PressureRatio}_{14d} > 1.25$).

