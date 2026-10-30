"""API Data Schemas & Validation Contracts.

Assigned Role: Integration Engineer
Deliverables:
- Strict Pydantic models for incoming container tracking telemetry.
- Response models for bottleneck risk tiers, probabilities, and anomaly scores.
- Models for operational trade-offs, feature importance, benchmarks, and batch requests.
- Contract compatibility with operations dashboard clients and terminal control towers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ContainerRiskTier(str, Enum):
    """3-Tier operational bottleneck classification categories."""
    ON_TIME = "ON_TIME"                # Class 0: <= 65.11 hours (<= 2.71 days)
    AT_RISK = "AT_RISK"                # Class 1: 65.11h - 160.90 hours (2.71 - 6.70 days)
    CRITICAL_DELAY = "CRITICAL_DELAY"  # Class 2: > 160.90 hours (> 6.70 days)


class UrgencyLevel(str, Enum):
    """Action priority rating for terminal dispatchers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ContainerTelemetryRequest(BaseModel):
    """Inbound container arrival telemetry request model.
    
    Contains the 15 active arrival-time predictors approved under the feature contract
    (strictly observable at or before container discharge t_discharge).
    """
    model_config = ConfigDict(extra="ignore", json_schema_extra={
        "example": {
            "case_id": "AAAU9001220-2021-06-28",
            "carrier_prefix": "AAAU",
            "CTR_SIZE": 40.0,
            "CTR_TYPE": "RFR",
            "GROSS": 28.5,
            "YARD_BLOCK": "Q",
            "YARD_SLOT": "14",
            "discharge_hour": 14,
            "discharge_day_of_week": 4,
            "discharge_month": 6,
            "is_weekend": 0,
            "trailing_7d_volume": 420.0,
            "trailing_7d_critical": 35.0,
            "trailing_7d_at_risk": 98.0,
            "historical_delay_rate_7d": 0.285,
            "trailing_7d_mean_dwell": 92.4,
            "alert_threshold_tau": 0.25
        }
    })

    case_id: Optional[str] = Field(
        default=None,
        description="Unique container stay identifier (e.g., 'AAAU9001220-2021-06-28'). If omitted, a synthetic ID is assigned."
    )
    carrier_prefix: str = Field(
        ...,
        description="First 4 characters of container ISO 6346 code identifying the carrier/shipping line (e.g. 'AAAU', 'MSCU')."
    )
    CTR_SIZE: float = Field(
        default=20.0,
        description="Container length in feet (e.g. 20.0, 40.0, 45.0)."
    )
    CTR_TYPE: str = Field(
        default="DRY",
        description="Container equipment classification (e.g. 'DRY', 'RFR' for Reefer, 'OVD' for Overdimensional, 'TANK')."
    )
    GROSS: float = Field(
        default=21.4,
        ge=0.0,
        le=80.0,
        description="Container gross weight in metric tonnes from VGM declaration."
    )
    YARD_BLOCK: str = Field(
        default="UNKNOWN",
        description="Provisional yard stack block assignment (e.g. 'B', 'N', 'Q')."
    )
    YARD_SLOT: str = Field(
        default="UNKNOWN",
        description="Provisional yard stack bay/slot coordinate (e.g. '01', '14')."
    )
    discharge_hour: int = Field(
        default=12,
        ge=0,
        le=23,
        description="Hour of container discharge milestone in local terminal time (0-23)."
    )
    discharge_day_of_week: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Day of week of discharge milestone (0=Monday, 6=Sunday)."
    )
    discharge_month: int = Field(
        default=7,
        ge=1,
        le=12,
        description="Month of discharge milestone (1-12)."
    )
    is_weekend: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary flag indicating weekend discharge (1 if Saturday or Sunday, else 0)."
    )
    trailing_7d_volume: float = Field(
        default=350.0,
        ge=0.0,
        description="Total completed departures through terminal during historical window [t-7, t-1]."
    )
    trailing_7d_critical: float = Field(
        default=25.0,
        ge=0.0,
        description="Count of severe bottlenecks (>160.9h dwell) completed during [t-7, t-1]."
    )
    trailing_7d_at_risk: float = Field(
        default=80.0,
        ge=0.0,
        description="Count of at-risk stays (65.1h-160.9h) completed during [t-7, t-1]."
    )
    historical_delay_rate_7d: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Ratio of delayed departures to total volume during trailing 7 days."
    )
    trailing_7d_mean_dwell: float = Field(
        default=75.0,
        ge=0.0,
        description="Average dwell duration in hours of completed stays during [t-7, t-1]."
    )
    alert_threshold_tau: Optional[float] = Field(
        default=0.25,
        ge=0.01,
        le=0.99,
        description="Configurable operational critical alert threshold tau. Defaults to 0.25."
    )

    @classmethod
    def from_flexible_dict(cls, data: Dict[str, Any]) -> "ContainerTelemetryRequest":
        """Sanitizes inputs and coerces percentages or missing fields gracefully."""
        d = dict(data)
        # Handle delay rate entered as 0-100 percentage
        if "historical_delay_rate_7d" in d and d["historical_delay_rate_7d"] is not None:
            try:
                rate = float(d["historical_delay_rate_7d"])
                if rate > 1.0 and rate <= 100.0:
                    d["historical_delay_rate_7d"] = round(rate / 100.0, 4)
                elif rate > 1.0:
                    d["historical_delay_rate_7d"] = 1.0
                elif rate < 0.0:
                    d["historical_delay_rate_7d"] = 0.0
            except (ValueError, TypeError):
                d["historical_delay_rate_7d"] = 0.20

        # Handle carrier prefix
        carrier = str(d.get("carrier_prefix", "")).strip().upper()
        d["carrier_prefix"] = carrier[:4] if carrier else "MSCU"

        # Coerce numeric bounds
        try:
            d["discharge_day_of_week"] = max(0, min(6, int(d.get("discharge_day_of_week", 0))))
        except Exception:
            d["discharge_day_of_week"] = 0

        try:
            d["discharge_hour"] = max(0, min(23, int(d.get("discharge_hour", 12))))
        except Exception:
            d["discharge_hour"] = 12

        try:
            d["discharge_month"] = max(1, min(12, int(d.get("discharge_month", 7))))
        except Exception:
            d["discharge_month"] = 7

        try:
            d["is_weekend"] = 1 if int(d.get("discharge_day_of_week", 0)) >= 5 else int(d.get("is_weekend", 0))
        except Exception:
            d["is_weekend"] = 0

        try:
            d["GROSS"] = max(1.0, float(d.get("GROSS", 20.0)))
        except Exception:
            d["GROSS"] = 20.0

        return cls(**d)


class BatchContainerRequest(BaseModel):
    """Vectorized batch request containing multiple container arrival items."""
    default_alert_threshold_tau: float = Field(
        default=0.25,
        ge=0.01,
        le=0.99,
        description="Default operational alert threshold applied to all items unless overridden."
    )
    containers: List[ContainerTelemetryRequest] = Field(
        ...,
        description="List of container telemetry objects for batch scoring."
    )


class ClassProbabilities(BaseModel):
    """Backward-compatible probability scores alias."""
    P_ON_TIME: float = Field(..., description="Decision score for ON_TIME tier (<= 65.11h)")
    P_AT_RISK: float = Field(..., description="Decision score for AT_RISK tier (65.11h - 160.90h)")
    P_CRITICAL_DELAY: float = Field(..., description="Decision score for CRITICAL_DELAY tier (> 160.90h)")


class ClassScores(BaseModel):
    """Standard model output decision partition scores."""
    score_on_time: float
    score_at_risk: float
    score_critical_delay: float


class RiskFactor(BaseModel):
    """Empirically verified risk contributor for the scored container."""
    name: str = Field(..., description="Name of the contributing feature.")
    domain: str = Field(..., description="Feature domain category.")
    impact: str = Field(..., description="Risk impact direction ('ELEVATED', 'MODERATE', 'NOMINAL').")
    value: Any = Field(..., description="Observed feature value.")
    interpretation: str = Field(..., description="Operational logistics explanation.")


class PredictionResponse(BaseModel):
    """Real-time container bottleneck prediction response."""
    case_id: str = Field(..., description="Container stay identifier.")
    predicted_class_id: int = Field(..., description="Predicted class index (0: ON_TIME, 1: AT_RISK, 2: CRITICAL_DELAY).")
    predicted_class_name: str = Field(..., description="Predicted class label.")
    urgency_level: UrgencyLevel = Field(..., description="Operational dispatch urgency tier.")
    scores: ClassScores = Field(..., description="Raw model decision partition scores.")
    probabilities: ClassProbabilities = Field(..., description="Probability alias object matching contracts.")
    alert_threshold_tau: float = Field(..., description="Operational alert cutoff tau used for triage.")
    alert_triggered: bool = Field(..., description="True if score_critical_delay >= alert_threshold_tau.")
    recommended_action: str = Field(..., description="Immediate operational triage directive.")
    operational_playbook: str = Field(..., description="Detailed standard operating procedure (SOP).")
    key_risk_factors: List[RiskFactor] = Field(default_factory=list, description="Primary empirical risk factors.")
    latency_ms: float = Field(..., description="Server inference execution time in milliseconds.")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp.")
    operational_note: str = Field(
        default="Scores represent uncalibrated gradient boosting decision scores. At default argmax, ~88.8% of critical delays are missed. Use alert_triggered with threshold_tau to adjust sensitivity.",
        description="Methodological operational disclaimer."
    )


class BatchPredictionResponse(BaseModel):
    """Vectorized batch triage results summary and items."""
    total_cases: int = Field(..., description="Total containers scored in batch.")
    flagged_count: int = Field(..., description="Number of containers triggering operational alert (score >= tau).")
    flagged_percentage: float = Field(..., description="Percentage of containers flagged for intervention.")
    class_distribution: Dict[str, int] = Field(..., description="Breakdown of predictions by class name.")
    average_critical_score: float = Field(..., description="Mean critical delay score across the batch.")
    total_latency_ms: float = Field(..., description="Total batch execution time in milliseconds.")
    avg_latency_per_case_ms: float = Field(..., description="Average inference latency per container.")
    predictions: List[PredictionResponse] = Field(..., description="Individual container prediction results.")


class HealthResponse(BaseModel):
    """System and model registry status payload."""
    status: str = Field(..., description="Overall service status ('HEALTHY', 'DEGRADED').")
    service: str = Field(default="Container Dwell Bottleneck Prediction API")
    version: str = Field(default="Stage 01 Final (Audited Pass)")
    model_loaded: bool = Field(..., description="Whether the ML model pipeline is active.")
    model_name: str = Field(default="EXP-BASE-01-OPT-MACRO (XGBoost Classifier)")
    sla_latency_p95: str = Field(default="< 25ms")
    active_features_count: int = Field(default=15)
    uptime_seconds: float = Field(...)
    timestamp: str = Field(...)


class OperationalTradeoffItem(BaseModel):
    """Sensitivity calibration entry for operational alert threshold tau."""
    threshold_tau: float
    flagged_count: int
    flagged_pct: float
    alerts_per_day: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float


class FeatureImportanceItem(BaseModel):
    """Ranked feature importance entry with operational provenance."""
    importance_rank: int
    feature_name: str
    feature_domain: str
    val_perm_macro_f1_drop: float
    xgb_gain_importance: float
    lgb_gain_importance: float
    evidence_and_operational_interpretation: str


class PortBenchmarkItem(BaseModel):
    """Indian Major Port official operational benchmark KPI entry."""
    port_name: str
    locode: str
    state: str
    coast: str
    annual_container_teu: int
    avg_turnaround_time_trt_hours: float
    pre_berthing_waiting_time_pbwt_hours: float
    avg_output_per_ship_berth_day_tonnes: float
    berth_occupancy_pct: float
    container_share_pct: float
