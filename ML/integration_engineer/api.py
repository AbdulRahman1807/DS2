"""Real-Time Bottleneck Prediction API Service.

Assigned Role: Integration Engineer
Named Deliverable: Real-Time Bottleneck Prediction API

Responsibilities:
- Expose low-latency REST service using FastAPI
- Endpoints:
  * GET  /health: Service and model registry health check
  * POST /predict/container: Real-time single container risk triage (< 25ms SLA)
  * POST /predict/batch: Vectorized batch scoring for operations dashboards
"""
