# Integration Engineer Workspace — Stage 01 Production Interface

## Mission
Expose bottleneck predictions via a low-latency operations REST API (< 25ms SLA), author the operational field guide for supply chain planners and dispatchers, and deliver a state-of-the-art Control Tower Web Dashboard for live container triage and manifest batch scoring.

---

## Deliverables & Architecture

1. **[`schemas.py`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/schemas.py)**:
   - Strict Pydantic v2 validation contracts for all 15 pre-discharge arrival features.
   - Comprehensive response models for multi-class bottleneck scores (`ON_TIME`, `AT_RISK`, `CRITICAL_DELAY`).
   - Models for vectorized batch scoring, operational alert sensitivity (τ), empirical risk diagnosis, feature importance, and port benchmarks.
   - **Status:** **Completed & Validated**.

2. **[`api.py`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/api.py)**:
   - **Named Deliverable:** Real-Time Bottleneck Prediction API & Control Tower Service.
   - FastAPI microservice with sub-25ms SLA execution.
   - Core endpoints:
     * `GET  /health`: Model registry and service SLA health check.
     * `POST /predict/container`: Low-latency real-time single container risk triage.
     * `POST /predict/batch`: Vectorized batch scoring for vessel discharge manifests.
     * `GET  /api/v1/metrics/operational`: Alert threshold (τ) trade-off curves.
     * `GET  /api/v1/features/importance`: Ranked feature importance matrix with domain hypotheses.
     * `GET  /api/v1/benchmarks`: Indian Major Ports baseline benchmarks.
     * `GET  /api/v1/sample-cases`: Real container case telemetry catalog.
     * `GET  /api/v1/contracts`: Feature contract specifications.
     * `GET  /` and `GET /dashboard`: Serves the Interactive Control Tower UI.
   - **Status:** **Completed & Validated**.

3. **[`static/index.html`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/static/index.html), [`static/dashboard.css`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/static/dashboard.css), [`static/dashboard.js`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/static/dashboard.js)**:
   - **Named Deliverable:** APM Terminal Bottleneck Control Tower Web Dashboard.
   - Dark-mode glassmorphic interface with circular SVG risk gauges, live probability meters, interactive τ threshold calibration slider, manifest batch processor with CSV export, and empirical feature drivers.
   - **Status:** **Completed & Serving**.

4. **[`field_guide.md`](file:///Users/raamharishab/Desktop/DS2/ML/integration_engineer/field_guide.md)**:
   - **Named Deliverable:** 1-page Logistics Ops Field Guide for supply chain planners.
   - Quick-triage decision matrix, empirical arrival signals, and operational SOP directives.
   - **Status:** **Completed**.

---

## Running the API & Dashboard Locally

Start the FastAPI microservice and web dashboard:
```bash
python3 ML/integration_engineer/api.py
```
Or via Uvicorn with auto-reload:
```bash
uvicorn ML.integration_engineer.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Control Tower Dashboard:** `http://localhost:8000/` or `http://localhost:8000/dashboard`
- **Interactive OpenAPI / Swagger Documentation:** `http://localhost:8000/docs`
- **Service Health Check:** `http://localhost:8000/health`
