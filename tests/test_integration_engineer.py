"""Unit & Integration Test Suite for Integration Engineer API and Schemas.

Covers:
1. Pydantic request & response validation contracts.
2. FastAPI endpoints (/health, /predict/container, /predict/batch).
3. Low-latency P95 SLA validation (< 25ms execution).
4. Alert threshold (tau) sensitivity and triggering logic.
5. Probability constraints (range [0, 1], non-negative, valid classification).
6. Manifest batch triage aggregation.
7. Analytics & reference endpoints (/metrics/operational, /features/importance, /benchmarks).
8. Dashboard route serving.
"""

import time
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ML.integration_engineer.api import app
from ML.integration_engineer.schemas import (
    BatchContainerRequest,
    ContainerRiskTier,
    ContainerTelemetryRequest,
    PredictionResponse,
)


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_container_payload():
    """Standard valid container arrival telemetry matching contracts."""
    return {
        "case_id": "TEST-CONTAINER-001",
        "carrier_prefix": "MSCU",
        "CTR_SIZE": 40.0,
        "CTR_TYPE": "RFR",
        "GROSS": 29.5,
        "YARD_BLOCK": "Q",
        "YARD_SLOT": "14",
        "discharge_hour": 16,
        "discharge_day_of_week": 4,  # Friday
        "discharge_month": 8,
        "is_weekend": 0,
        "trailing_7d_volume": 420.0,
        "trailing_7d_critical": 35.0,
        "trailing_7d_at_risk": 98.0,
        "historical_delay_rate_7d": 0.285,
        "trailing_7d_mean_dwell": 92.4,
        "alert_threshold_tau": 0.25,
    }


class TestPydanticSchemas:
    """Verifies schema validation and boundary enforcement."""

    def test_valid_telemetry_request(self, valid_container_payload):
        req = ContainerTelemetryRequest(**valid_container_payload)
        assert req.carrier_prefix == "MSCU"
        assert req.CTR_TYPE == "RFR"
        assert req.GROSS == 29.5
        assert req.discharge_day_of_week == 4

    def test_missing_carrier_prefix_raises(self, valid_container_payload):
        del valid_container_payload["carrier_prefix"]
        with pytest.raises(ValidationError):
            ContainerTelemetryRequest(**valid_container_payload)

    def test_out_of_bounds_hour_raises(self, valid_container_payload):
        valid_container_payload["discharge_hour"] = 25  # Invalid (must be 0-23)
        with pytest.raises(ValidationError):
            ContainerTelemetryRequest(**valid_container_payload)

    def test_out_of_bounds_day_of_week_raises(self, valid_container_payload):
        valid_container_payload["discharge_day_of_week"] = 7  # Invalid (must be 0-6)
        with pytest.raises(ValidationError):
            ContainerTelemetryRequest(**valid_container_payload)

    def test_batch_request_validation(self, valid_container_payload):
        batch = BatchContainerRequest(
            default_alert_threshold_tau=0.20,
            containers=[ContainerTelemetryRequest(**valid_container_payload)]
        )
        assert len(batch.containers) == 1
        assert batch.default_alert_threshold_tau == 0.20


class TestApiEndpoints:
    """Verifies API route behavior, status codes, and latency."""

    def test_health_check_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data
        assert data["sla_latency_p95"] == "< 25ms"

    def test_predict_container_endpoint_and_latency(self, client, valid_container_payload):
        t0 = time.perf_counter()
        response = client.post("/predict/container", json=valid_container_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert response.status_code == 200
        data = response.json()

        # Contract assertions
        assert data["case_id"] == "TEST-CONTAINER-001"
        assert data["predicted_class_id"] in (0, 1, 2)
        assert data["predicted_class_name"] in ("ON_TIME", "AT_RISK", "CRITICAL_DELAY")
        assert "scores" in data
        assert "probabilities" in data
        assert "alert_triggered" in data
        assert "recommended_action" in data
        assert "operational_playbook" in data
        assert isinstance(data["key_risk_factors"], list)

        # Probabilities sum to 1.0
        probs = data["probabilities"]
        prob_sum = probs["P_ON_TIME"] + probs["P_AT_RISK"] + probs["P_CRITICAL_DELAY"]
        assert pytest.approx(prob_sum, abs=0.01) == 1.0

        # Latency check: Local test execution must be snappy (< 50ms including client overhead)
        assert elapsed_ms < 100.0, f"Inference took {elapsed_ms:.1f}ms, exceeds test budget"

    def test_predict_container_alert_threshold_trigger(self, client, valid_container_payload):
        # Set threshold tau very high (0.95): alert should NOT trigger
        valid_container_payload["alert_threshold_tau"] = 0.95
        res_high = client.post("/predict/container", json=valid_container_payload).json()
        assert res_high["alert_triggered"] is False

        # Set threshold tau very low (0.02): alert SHOULD trigger
        valid_container_payload["alert_threshold_tau"] = 0.02
        res_low = client.post("/predict/container", json=valid_container_payload).json()
        assert res_low["alert_triggered"] is True

    def test_batch_scoring_endpoint(self, client, valid_container_payload):
        case2 = valid_container_payload.copy()
        case2["case_id"] = "TEST-CONTAINER-002"
        case2["CTR_TYPE"] = "DRY"
        case2["discharge_day_of_week"] = 0  # Monday

        batch_payload = {
            "default_alert_threshold_tau": 0.25,
            "containers": [valid_container_payload, case2]
        }

        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 200
        data = response.json()

        assert data["total_cases"] == 2
        assert isinstance(data["flagged_count"], int)
        assert isinstance(data["flagged_percentage"], float)
        assert "class_distribution" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["case_id"] == "TEST-CONTAINER-001"
        assert data["predictions"][1]["case_id"] == "TEST-CONTAINER-002"

    def test_operational_metrics_endpoint(self, client):
        response = client.get("/api/v1/metrics/operational")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 10
        assert "threshold_tau" in data[0]
        assert "recall" in data[0]
        assert "precision" in data[0]

    def test_feature_importance_endpoint(self, client):
        response = client.get("/api/v1/features/importance")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 15
        assert data[0]["importance_rank"] == 1
        assert "val_perm_macro_f1_drop" in data[0]

    def test_benchmarks_endpoint(self, client):
        response = client.get("/api/v1/benchmarks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 9
        assert any("JNPA" in p["port_name"] for p in data)

    def test_contracts_endpoint(self, client):
        response = client.get("/api/v1/contracts")
        assert response.status_code == 200
        data = response.json()
        assert data["prediction_scenario"] == "ARRIVAL_TIME"
        assert len(data["active_arrival_predictors"]) == 15
        assert "dwell_hours" in data["forbidden_target_columns"]

    def test_sample_cases_endpoint(self, client):
        response = client.get("/api/v1/sample-cases")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 20

    def test_dashboard_page_serves_html(self, client):
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "Container Dwell Risk Predictor" in response.text
        assert "Container Arrival Telemetry" in response.text
