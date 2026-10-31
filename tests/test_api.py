"""Unit and Integration Tests for Real-Time Bottleneck Prediction API & Dashboard.

Assigned Role: Integration Engineer
"""

import pytest
from fastapi.testclient import TestClient
from ML.integration_engineer.api import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"].upper() == "HEALTHY"
    assert data["model_loaded"] is True
    assert "LGBMClassifier" in data["model_type"] or "XGBoost" in data["model_type"]
    assert data["active_features_count"] in (13, 15)


def test_predict_single_container():
    payload = {
        "case_id": "MSKU9012345-2021-07-28",
        "container_id": "MSKU9012345",
        "carrier_prefix": "MSK",
        "CTR_TYPE": "DRY",
        "CTR_SIZE": 40,
        "GROSS": 24.5,
        "discharge_hour": 14,
        "discharge_day_of_week": 2,
        "discharge_month": 7,
        "is_weekend": 0,
        "trailing_7d_volume": 1200.0,
        "trailing_7d_critical": 150.0,
        "trailing_7d_at_risk": 350.0,
        "historical_delay_rate_7d": 0.125,
        "trailing_7d_mean_dwell": 68.4,
        "alert_threshold_tau": 0.25,
    }
    response = client.post("/predict/container", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class_id"] in [0, 1, 2]
    assert data["predicted_class_name"] in ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
    
    probs = data["probabilities"]
    assert "P_ON_TIME" in probs
    assert "P_AT_RISK" in probs
    assert "P_CRITICAL_DELAY" in probs
    total_prob = sum(probs.values())
    assert 0.99 <= total_prob <= 1.01

    assert isinstance(data["alert_triggered"], bool)
    assert data["inference_latency_ms"] < 35.0  # Must satisfy SLA budget
    assert len(data["recommended_action"]) > 5


def test_predict_batch():
    container_1 = {
        "case_id": "MSKU9012345-2021-07-28",
        "container_id": "MSKU9012345",
        "carrier_prefix": "MSK",
        "CTR_TYPE": "DRY",
        "CTR_SIZE": 40,
        "GROSS": 24.5,
        "discharge_hour": 14,
        "discharge_day_of_week": 2,
        "discharge_month": 7,
        "is_weekend": 0,
        "trailing_7d_volume": 1200.0,
        "trailing_7d_critical": 150.0,
        "trailing_7d_at_risk": 350.0,
        "historical_delay_rate_7d": 0.125,
        "trailing_7d_mean_dwell": 68.4,
        "alert_threshold_tau": 0.25,
    }
    container_2 = dict(container_1, container_id="CMAU8492019", carrier_prefix="CMA", CTR_TYPE="REEFER", GROSS=31.2)
    
    batch_payload = {"containers": [container_1, container_2]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
    assert data["total_batch_latency_ms"] > 0.0


def test_dashboard_stats_endpoint():
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_active_monitored"] > 0
    assert "top_bottlenecks_queue" in data
    assert len(data["top_bottlenecks_queue"]) > 0


def test_dashboard_html_view():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Port Operations Control Tower" in response.text
    assert "Real-Time Triage Simulator" in response.text

