# AI-Powered Prediction of Global Supply Chain Logistics Bottlenecks
*Autonomous Logistics Intelligence Engine for Port-to-Warehouse Bottleneck Triage & Mitigation*

Repository: [https://github.com/AbdulRahman1807/DS2.git](https://github.com/AbdulRahman1807/DS2.git)

---

## Stage 01: Machine Learning & Telemetry Ingestion
Stage 01 focuses on baseline statistical bottleneck triage across structured shipment tracking and port telemetry.

### Squad Architecture & Roles (`ML/`)

```text
ML/
├── data_engineer/             # Data Ingestion, Cleaning & Splitting
│   ├── ingest_mendeley.py     # Convert 63.8 MB Excel event log to Parquet
│   ├── fetch_portwatch.py     # Ingest IMF PortWatch disruptions & port directory
│   ├── cleaner.py             # Event log sanitization & incomplete stay quarantine
│   ├── dwell_clock.py         # DISCHARGE -> TRUCK_OUT dwell calculation & labeling
│   ├── splitter.py            # Leak-free temporal & container-grouped train/test split
│   └── rolling_metrics.py     # 7-day trailing delay rates & congestion indicators
│
├── eda_engineer/              # Exploratory Data Analysis & Quality Audit
│   ├── profiler.py            # Cardinality, activity counts, dwell percentiles
│   ├── drift_analysis.py      # Noise floor variations, reporting drift, call date gaps
│   └── eda_runner.py          # Diagnostic figures & reporting script
│
├── ml_engineer/               # Baseline Anomaly & Bottleneck Classifier
│   ├── anomaly_baseline.py    # Isolation Forest & One-Class SVM models
│   ├── classifier.py          # Multi-class XGBoost model (On-Time / At-Risk / Critical)
│   ├── feature_importance.py  # Feature Importance Matrix & SHAP values (Named Deliverable)
│   └── train.py               # Model training orchestrator
│
├── evaluation_engineer/       # Validation & False-Positive Trade-Offs
│   ├── metrics.py             # Confusion matrix, Precision/Recall, Macro F1
│   ├── cost_matrix.py         # False-alarm vs emergency reroute cost optimizer
│   └── evaluate.py            # Test split benchmark runner
│
└── integration_engineer/      # Serving API & Planners Field Guide
    ├── schemas.py             # Pydantic request / response contracts
    ├── api.py                 # Real-Time Bottleneck Prediction API (Named Deliverable)
    └── field_guide.md         # 1-Page Logistics Ops Field Guide (Named Deliverable)
```

---

## Data Pillars
1. **Container Tracking & Dwell:** Mendeley Container Dwell Event Log (`data/raw/mendeley_event_log.parquet`, 952k rows, 95k stays).
2. **Disruptions & Weather:** IMF PortWatch Global Disruptions (`data/raw/portwatch_disruptions.parquet`, 132 GDACS events).
3. **Ports Reference:** IMF PortWatch Ports Directory (`data/reference/portwatch_ports.parquet`, 2,000+ ports).

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest datasets (Data Engineer)
python ML/data_engineer/ingest_mendeley.py
python ML/data_engineer/fetch_portwatch.py
```
