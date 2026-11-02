# Evaluation Engineer — Prediction Artifacts Directory

This directory provides standardized access to the authoritative out-of-sample prediction artifacts generated during Phase 2 Deep Learning validation and blind-test evaluation.

To prevent repository bloat and comply with project file size governance, the canonical underlying binary Parquet files are stored in the ML Engineer artifact storage and linked here via relative filesystem references.

---

## 1. Prediction Artifact Inventory

| Artifact Alias | Canonical Source Path | Size | Row Count | Target Horizons | Cryptographic SHA-256 Digest |
|---|---|---:|---:|---|---|
| [`blind_test_predictions_2025.parquet`](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/artifacts/predictions/blind_test_predictions_2025.parquet) | `DL/ml engineer/artifacts/throughput_forecasting/blind_test_predictions_2025.parquet` | 18.94 MB | 741,335 | 7-day Future Container Calls & Cargo Tonnage ($t+1 \dots t+7$) | `57552e6cb469347ac9678d97bdbaef367c7c9a56d8d3343261a82201ffd05abf` |
| [`validation_predictions_full_2024.parquet`](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/artifacts/predictions/validation_predictions_full_2024.parquet) | `DL/ml engineer/artifacts/throughput_forecasting/validation_predictions_full_2024.parquet` | 18.45 MB | 743,400 | 7-day Future Container Calls & Cargo Tonnage ($t+1 \dots t+7$) | `f5aad11c457d17f88dc842189a1a14bd0d17c811e3059fea7b11ce6823e73379` |
| [`dwell_validation_predictions_2021.parquet`](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/artifacts/predictions/dwell_validation_predictions_2021.parquet) | `DL/ml engineer/artifacts/dwell_forecasting/validation_predictions.parquet` | 40.85 KB | 1,007 | Weekly Mean Anchorage Waiting Time ($H=1$ and $H=4$) | `9a59df1c8cbace4db6e3c351db6cbf733a1f72e98ad78772b36f7208589eea53` |
| [`dwell_test_predictions_2022.parquet`](file:///Users/abdul/Desktop/DS/DL/evaluation%20engineer/artifacts/predictions/dwell_test_predictions_2022.parquet) | `DL/ml engineer/artifacts/dwell_forecasting/test_predictions.parquet` | 37.70 KB | 931 | Weekly Mean Anchorage Waiting Time ($H=1$ and $H=4$) | `2b7ca8126cfae89b29e7e4ea05f222fb3a8ee49950c5b5780818bb6a25d30ecd` |

---

## 2. Parquet Schema Contracts

### 2.1 Direct Throughput Parquets (`blind_test_predictions_2025.parquet` and `validation_predictions_full_2024.parquet`)
- `sequence_id` (`string`): Unique identifier formatted as `{portwatch_id}_{reference_date}`.
- `portwatch_id` (`string`): UN/LOCODE or PortWatch facility identifier (e.g., `USLAX`, `CNSHA`).
- `portwatch_name` (`string`): Canonical port name.
- `country_iso3` (`string`): 3-letter ISO country code.
- `split_tag` (`string`): Partition identifier (`val` or `test`).
- `reference_date` (`string`): Observation origin timestamp $T$.
- `input_start_date` (`string`): Lookback origin timestamp $T-31$.
- `target_start_date` (`string`): Forecast window origin $T+1$.
- `target_end_date` (`string`): Forecast window terminus $T+7$.
- `actual_container_calls` (`float32`): Realized container-carrying vessel arrivals over $[T+1, T+7]$.
- `naive_container_calls` (`float32`): 7-day persistence baseline ($7.0 \times \text{roll7 container calls}$ at $T-2$).
- `pred_container_calls` (`float32`): Frozen residual Transformer prediction.
- `error_container_calls` (`float32`): Prediction error ($\hat{y} - y$).
- `actual_cargo_tonnage` (`float32`): Realized deadweight metric tons loaded/discharged over $[T+1, T+7]$.
- `naive_cargo_tonnage` (`float32`): 7-day persistence baseline ($7.0 \times \text{daily cargo tonnage}$ at $T-2$).
- `pred_cargo_tonnage` (`float32`): Frozen residual Transformer prediction.
- `error_cargo_tonnage` (`float32`): Prediction error ($\hat{y} - y$).

### 2.2 Weekly Anchorage Dwell Parquets (`dwell_*.parquet`)
- `port_id` (`string`): Port identifier across 19 mega-ports.
- `year_week` (`string`): ISO year-week format (`YYYY-WW`).
- `actual_waiting_time` (`float32`): Realized outer anchorage waiting time in hours.
- `naive_waiting_time` (`float32`): Trailing 1-week persistence baseline.
- `pred_waiting_time` (`float32`): Residual Dwell GRU forecast.
- `error_waiting_time` (`float32`): Residual error ($\hat{y} - y$).

