"""Feature Registry and Validation Specification for Milestone 3.1.

Defines the explicit, ordered 20-feature universal operational core for deep learning
sequence modeling. Validates that incoming dataset schemas and tensors conform
strictly to operational availability and data contracts.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import pyarrow as pa
import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    """Specification of a single input predictor feature."""
    name: str
    source_column: str
    feature_group: str  # 'operational', 'trailing_rolling', 'calendar'
    dtype: str          # 'float32'
    availability_policy: str  # 'portwatch_t_minus_2' or 'deterministic_calendar'
    is_derived: bool
    transformation_rule: str
    imputation_rule: str
    description: str


# -----------------------------------------------------------------------------
# Explicit, Ordered 20-Feature Universal Operational Core Registry
# -----------------------------------------------------------------------------
CORE_20_FEATURE_REGISTRY: Tuple[FeatureDefinition, ...] = (
    # Group 1: Operational Ground Truth Vessel Calls & Cargo (12 Features)
    FeatureDefinition(
        name="portcalls",
        source_column="portcalls",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Total commercial vessel arrivals on observation day d.",
    ),
    FeatureDefinition(
        name="portcalls_container",
        source_column="portcalls_container",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Container vessel arrivals on observation day d.",
    ),
    FeatureDefinition(
        name="portcalls_dry_bulk",
        source_column="portcalls_dry_bulk",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Dry bulk vessel arrivals on observation day d.",
    ),
    FeatureDefinition(
        name="portcalls_general_cargo",
        source_column="portcalls_general_cargo",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="General cargo vessel arrivals on observation day d.",
    ),
    FeatureDefinition(
        name="portcalls_roro",
        source_column="portcalls_roro",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Roll-on/Roll-off vehicle carrier vessel arrivals on observation day d.",
    ),
    FeatureDefinition(
        name="portcalls_tanker",
        source_column="portcalls_tanker",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Liquid tanker vessel arrivals on observation day d.",
    ),
    FeatureDefinition(
        name="portcalls_cargo",
        source_column="portcalls_cargo",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Combined commercial cargo vessel arrivals on observation day d.",
    ),
    FeatureDefinition(
        name="import_container",
        source_column="import_container",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Estimated laden import container volume on observation day d.",
    ),
    FeatureDefinition(
        name="import_cargo",
        source_column="import_cargo",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Estimated import cargo metric tonnage on observation day d.",
    ),
    FeatureDefinition(
        name="export_container",
        source_column="export_container",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Estimated laden export container volume on observation day d.",
    ),
    FeatureDefinition(
        name="export_cargo",
        source_column="export_cargo",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=False,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Estimated export cargo metric tonnage on observation day d.",
    ),
    FeatureDefinition(
        name="total_cargo_tonnage",
        source_column="total_cargo_tonnage",
        feature_group="operational",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Sum of import and export waterborne cargo tonnage on observation day d.",
    ),

    # Group 2: Trailing Rolling Operational Features (5 Features)
    FeatureDefinition(
        name="pw_calls_rolling_7d_mean",
        source_column="pw_calls_rolling_7d_mean",
        feature_group="trailing_rolling",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Strictly trailing 7-day backward mean of total portcalls [d-6, d].",
    ),
    FeatureDefinition(
        name="pw_calls_rolling_14d_mean",
        source_column="pw_calls_rolling_14d_mean",
        feature_group="trailing_rolling",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Strictly trailing 14-day backward mean of total portcalls [d-13, d].",
    ),
    FeatureDefinition(
        name="pw_calls_rolling_30d_mean",
        source_column="pw_calls_rolling_30d_mean",
        feature_group="trailing_rolling",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Strictly trailing 30-day backward mean of total portcalls [d-29, d].",
    ),
    FeatureDefinition(
        name="pw_container_rolling_7d_mean",
        source_column="pw_container_rolling_7d_mean",
        feature_group="trailing_rolling",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Strictly trailing 7-day backward mean of container vessel calls [d-6, d].",
    ),
    FeatureDefinition(
        name="pw_container_rolling_14d_mean",
        source_column="pw_container_rolling_14d_mean",
        feature_group="trailing_rolling",
        dtype="float32",
        availability_policy="portwatch_t_minus_2",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Strictly trailing 14-day backward mean of container vessel calls [d-13, d].",
    ),

    # Group 3: Calendar Features (3 Features)
    FeatureDefinition(
        name="day_of_week",
        source_column="day_of_week",
        feature_group="calendar",
        dtype="float32",
        availability_policy="deterministic_calendar",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Day of week index (0=Monday, 6=Sunday) for observation day d.",
    ),
    FeatureDefinition(
        name="is_weekend",
        source_column="is_weekend",
        feature_group="calendar",
        dtype="float32",
        availability_policy="deterministic_calendar",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Binary indicator {0, 1} whether observation day d is Saturday or Sunday.",
    ),
    FeatureDefinition(
        name="day_of_year",
        source_column="day_of_year",
        feature_group="calendar",
        dtype="float32",
        availability_policy="deterministic_calendar",
        is_derived=True,
        transformation_rule="standard_scale",
        imputation_rule="never_null",
        description="Julian day of year (1-366) for observation day d.",
    ),
)

# Ordered tuple of exactly 20 feature names
FEATURE_NAMES: Tuple[str, ...] = tuple(f.name for f in CORE_20_FEATURE_REGISTRY)
NUM_FEATURES: int = len(FEATURE_NAMES)
assert NUM_FEATURES == 20, f"Core registry must contain exactly 20 features, found {NUM_FEATURES}"


def get_feature_definition(feature_name: str) -> FeatureDefinition:
    """Retrieve the specification for a given feature name."""
    for feat in CORE_20_FEATURE_REGISTRY:
        if feat.name == feature_name:
            return feat
    raise KeyError(f"Feature '{feature_name}' not found in 20-feature core registry.")


def validate_feature_registry(available_columns: List[str]) -> Dict[str, Any]:
    """Validate that the source schema satisfies the 20-feature operational core contract.
    
    Args:
        available_columns: List of column names present in the data source.
        
    Returns:
        Dictionary summarizing validation status and column mappings.
        
    Raises:
        ValueError: If any expected feature is missing or prohibited columns are requested.
    """
    col_set = set(available_columns)
    missing_cols = [f.source_column for f in CORE_20_FEATURE_REGISTRY if f.source_column not in col_set]
    if missing_cols:
        raise ValueError(
            f"Missing {len(missing_cols)} required core features in source table: {missing_cols}"
        )

    # Disallowed leakage / identifier column check
    prohibited_prefixes = ("target_",)
    prohibited_exact = {"portwatch_id", "sequence_id", "portwatch_name", "country_name", "country_iso3"}
    detected_leaks = [f for f in FEATURE_NAMES if f.startswith(prohibited_prefixes) or f in prohibited_exact]
    if detected_leaks:
        raise ValueError(f"Prohibited target/identifier columns found in feature registry: {detected_leaks}")

    return {
        "status": "VALIDATED",
        "num_features": NUM_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "source_columns": [f.source_column for f in CORE_20_FEATURE_REGISTRY],
        "feature_groups": {
            "operational": [f.name for f in CORE_20_FEATURE_REGISTRY if f.feature_group == "operational"],
            "trailing_rolling": [f.name for f in CORE_20_FEATURE_REGISTRY if f.feature_group == "trailing_rolling"],
            "calendar": [f.name for f in CORE_20_FEATURE_REGISTRY if f.feature_group == "calendar"],
        },
    }
