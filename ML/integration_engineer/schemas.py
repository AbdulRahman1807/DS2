"""API Schemas for Real-Time Bottleneck Prediction.

Assigned Role: Integration Engineer
Deliverables:
- Pydantic models for incoming container shipment telemetry
- Strict response contracts providing probability breakdown, triage class, and anomaly flags
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ContainerTelemetryRequest(BaseModel):
    case_id: str = Field(..., description="Unique container case identifier (e.g. AAAU9001220-2021-06-28)")
    container_id: str = Field(..., description="11-character ISO container number")
    call_date: str = Field(..., description="Vessel call date (YYYY-MM-DD)")
    ctr_size: Optional[int] = Field(None, description="Container size (20 or 40 feet)")
    ctr_type: Optional[str] = Field("DRY", description="Container category (DRY, RFR)")
    gross_weight: Optional[float] = Field(None, description="Gross cargo weight in tonnes")
    yard_block: Optional[str] = Field(None, description="Assigned yard block")
    has_customs_hold: int = Field(0, description="Binary flag indicating customs hold / red flag PIB")
    elapsed_dwell_hours: float = Field(..., description="Current elapsed hours since DISCHARGE")
    trailing_7d_delay_rate: float = Field(..., description="Trailing 7-day terminal delay rate")


class BottleneckPredictionResponse(BaseModel):
    case_id: str
    predicted_class: str = Field(..., description="'ON_TIME', 'AT_RISK', or 'CRITICAL_DELAY'")
    class_probabilities: Dict[str, float] = Field(
        ..., description="Calibrated probabilities for each operational tier"
    )
    is_anomaly: bool = Field(..., description="True if Isolation Forest flags stay as statistical outlier")
    anomaly_score: float = Field(..., description="Continuous decision score from baseline anomaly detector")
    recommended_action: str = Field(
        ..., description="Recommended planner triage action (e.g. Expedite Customs, Queue Reroute, Nominal Monitoring)"
    )
    inference_latency_ms: float = Field(..., description="API inference latency in milliseconds")


class BatchTelemetryRequest(BaseModel):
    containers: List[ContainerTelemetryRequest]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[BottleneckPredictionResponse]
