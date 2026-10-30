"""API Data Schemas & Validation Contracts.

Assigned Role: Integration Engineer
Named Deliverable: Real-Time Bottleneck Prediction API & Operations Dashboard

Responsibilities:
- Define Pydantic request models for inbound container telemetry streams
- Define response models for bottleneck risk tiers, probabilities, and anomaly scores
- Ensure contract compatibility with operations dashboard clients and control towers
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ContainerTelemetry(BaseModel):
    case_id: Optional[str] = Field(
        default="AAAU9001220-2021-06-28", description="Unique container stay or case identifier"
    )
    container_id: Optional[str] = Field(
        default="AAAU9001220", description="11-character ISO container identifier"
    )
    carrier_prefix: str = Field(
        default="MSK", description="3-letter carrier prefix code (e.g. MSK, CMA, ONE)"
    )
    CTR_TYPE: str = Field(
        default="DRY", description="Container equipment type (e.g. DRY, REEFER, TANK)"
    )
    CTR_SIZE: int = Field(
        default=40, description="Container nominal length in feet (20 or 40)"
    )
    GROSS: float = Field(
        default=24.5, description="Gross weight in metric tonnes"
    )
    discharge_hour: int = Field(
        default=14, ge=0, le=23, description="Hour of crane discharge (0-23)"
    )
    discharge_day_of_week: int = Field(
        default=2, ge=0, le=6, description="Discharge day of week (0=Mon, 6=Sun)"
    )
    discharge_month: int = Field(
        default=7, ge=1, le=12, description="Discharge calendar month (1-12)"
    )
    is_weekend: int = Field(
        default=0, ge=0, le=1, description="1 if discharged on weekend, else 0"
    )
    trailing_7d_volume: float = Field(
        default=1200.0, ge=0, description="Total completed departures in trailing 7 days"
    )
    trailing_7d_critical: float = Field(
        default=150.0, ge=0, description="Critical delay departures in trailing 7 days"
    )
    trailing_7d_at_risk: float = Field(
        default=350.0, ge=0, description="At-risk departures in trailing 7 days"
    )
    historical_delay_rate_7d: float = Field(
        default=0.125, ge=0.0, le=1.0, description="Trailing 7-day terminal delay rate"
    )
    trailing_7d_mean_dwell: float = Field(
        default=68.4, ge=0.0, description="Trailing 7-day mean dwell time in hours"
    )
    alert_threshold_tau: Optional[float] = Field(
        default=0.25, ge=0.05, le=0.95, description="Operational decision cutoff for CRITICAL_DELAY"
    )


class ContainerTriageResponse(BaseModel):
    case_id: Optional[str]
    container_id: Optional[str]
    carrier_prefix: Optional[str] = None
    CTR_TYPE: Optional[str] = None
    GROSS: Optional[float] = None
    predicted_class_id: int = Field(description="0: ON_TIME, 1: AT_RISK, 2: CRITICAL_DELAY")
    predicted_class_name: str = Field(description="ON_TIME, AT_RISK, or CRITICAL_DELAY")
    probabilities: Dict[str, float] = Field(
        description="Class probabilities: P_ON_TIME, P_AT_RISK, P_CRITICAL_DELAY"
    )
    critical_delay_risk_score: float = Field(
        description="Probability score for CRITICAL_DELAY (> 160.9 hours)"
    )
    alert_threshold_tau: float = Field(
        description="Operational decision threshold cutoff used"
    )
    alert_triggered: bool = Field(
        description="True if critical_delay_risk_score >= alert_threshold_tau"
    )
    recommended_action: str = Field(
        description="Operational playbook intervention from Logistics Ops Field Guide"
    )
    inference_latency_ms: float = Field(
        description="Model scoring latency in milliseconds"
    )


class BatchTriageRequest(BaseModel):
    containers: List[ContainerTelemetry] = Field(
        ..., min_length=1, max_length=1000, description="List of containers to score"
    )


class BatchTriageResponse(BaseModel):
    total_processed: int
    bottlenecks_flagged: int
    high_risk_share: float
    results: List[ContainerTriageResponse]
    total_batch_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    model_type: str
    active_features_count: int
    latency_sla: str


class DashboardStats(BaseModel):
    total_active_monitored: int
    on_time_count: int
    at_risk_count: int
    critical_delay_count: int
    critical_delay_percentage: float
    terminal_7d_mean_dwell_hours: float
    terminal_7d_delay_rate: float
    top_bottlenecks_queue: List[ContainerTriageResponse]
    operational_threshold_tau: float
