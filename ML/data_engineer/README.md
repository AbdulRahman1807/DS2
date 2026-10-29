# Data Engineer Workspace - Stage 01

## Mission
Build reliable ingestion, sanitization, and splitting pipelines across multi-carrier shipment feeds and port telemetry.

## Key Files & Tasks
1. [`ingest_mendeley.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/ingest_mendeley.py):
   - Ingests raw Excel (`Dataset of Dwelling Time Process Fix.xlsx`) and writes optimized Parquet to `data/raw/mendeley_event_log.parquet`.
   - **Status:** Implemented & Verified.

2. [`fetch_portwatch.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/fetch_portwatch.py):
   - Queries IMF PortWatch ArcGIS REST APIs for global disruption alerts and port registry.
   - **Status:** Implemented & Verified.

3. [`cleaner.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/cleaner.py):
   - Parse timestamps, resolve formula errors (`=J2-I2`), pivot container attributes (`CTR_SIZE`, `CTR_TYPE`, `GROSS`, `YARD_BLOCK`), and quarantine incomplete stays (missing `TRUCK_OUT`).
   - **Status:** Stub generated.

4. [`dwell_clock.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/dwell_clock.py):
   - Compute dwell duration from `DISCHARGE` start to `TRUCK_OUT` end.
   - Fit percentiles strictly on the training set and assign 3-tier target labels (`ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`).
   - **Status:** Stub generated.

5. [`splitter.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/splitter.py):
   - Strict chronological + container-grouped splitting (`burn_in`, `train`, `val`, `test`).
   - Guarantees zero leakage across container IDs.
   - **Status:** Stub generated.

6. [`rolling_metrics.py`](file:///Users/abdul/Desktop/DS/ML/data_engineer/rolling_metrics.py):
   - Computes lagged 7-day trailing `historical_delay_rate` and port congestion index.
   - **Status:** Stub generated.
