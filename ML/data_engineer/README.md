# Data Engineer Workspace - Stage 01

## Mission
Build reliable ingestion, sanitization, feature engineering, and leak-free splitting pipelines across multi-carrier shipment feeds and port telemetry.

## Completed Pipelines & Modules

1. [`ingest_mendeley.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/ingest_mendeley.py):
   - Ingests raw Excel event log (`Dataset of Dwelling Time Process Fix.xlsx`) and writes optimized columnar Parquet to `data/raw/mendeley_event_log.parquet`.
   - **Status:** Complete.

2. [`fetch_portwatch.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/fetch_portwatch.py):
   - Ingests IMF PortWatch global disruptions database (132 GDACS alerts) and global ports registry (2,000 ports).
   - **Status:** Complete.

3. [`ingest_indian_benchmarks.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/ingest_indian_benchmarks.py):
   - Compiles Ministry of Ports, Shipping and Waterways (MOPSW - TRW) Major Ports operational KPIs (TRT, PBWT, OSBD, berth occupancy) into `data/reference/indian_ports_benchmarks.parquet`.
   - **Status:** Complete.

4. [`cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py):
   - Normalizes activity codes (stripping whitespace like `'TRUCK_IN '`).
   - Extracts container numbers, carrier prefixes, and call dates.
   - Quarantines incomplete stays missing exit events (`CSNU6526902-2021-04-19`).
   - Pivots scattered attributes (`CTR_SIZE`, `CTR_TYPE`, `GROSS`, `YARD_BLOCK`, customs flags).
   - **Status:** Complete.

5. [`dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py):
   - Measures container dwell time (`DISCHARGE` start to `TRUCK_OUT` end).
   - Fits training percentiles ($p_{50}=54.98\text{h}$, $p_{85}=140.28\text{h}$) to avoid lookahead bias.
   - Assigns 3-tier ground-truth labels (`ON_TIME`: 50%, `AT_RISK`: 35%, `CRITICAL_DELAY`: 15%).
   - **Status:** Complete.

6. [`rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py):
   - Calculates 7-day trailing `historical_delay_rate_7d`, `trailing_7d_volume`, and `trailing_7d_mean_dwell`.
   - Strictly lagged by 1 day ($t-7$ to $t-1$) to guarantee zero lookahead leakage into current stay predictions.
   - **Status:** Complete.

7. [`splitter.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/splitter.py):
   - Isolates sparse burn-in call dates (< 2021-06-10).
   - Chronological split of 104 dense call dates into Train (72 dates), Val (16 dates), and Test (16 dates).
   - Reassigns multi-stay containers to ensure 100% disjoint container sets across splits.
   - **Status:** Complete.

8. [`build_pipeline.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/build_pipeline.py):
   - Master end-to-end orchestration pipeline producing final 32-feature datasets in `data/processed/`.
   - **Status:** Complete.

## Execution

```bash
# Run the complete end-to-end data engineering pipeline
python ML/data_engineer/build_pipeline.py
```
