"""Data engineering and sequence pipeline package for Milestone 3.1."""

from .feature_registry import (
    FeatureDefinition,
    CORE_20_FEATURE_REGISTRY,
    FEATURE_NAMES,
    NUM_FEATURES,
    get_feature_definition,
    validate_feature_registry,
)
from .preprocessing import (
    ScalerParameters,
    TrainingStandardScaler,
)
from .storage import (
    StorageManifest,
    MemmapStorageBuilder,
    MemmapStorageReader,
    STORAGE_DAT_FILENAME,
    MANIFEST_FILENAME,
    MARKER_FILENAME,
)
from .sequence_dataset import (
    PortSequenceDataset,
)
from .data_module import (
    PortSequenceDataModule,
    create_dataloaders,
)

__all__ = [
    "FeatureDefinition",
    "CORE_20_FEATURE_REGISTRY",
    "FEATURE_NAMES",
    "NUM_FEATURES",
    "get_feature_definition",
    "validate_feature_registry",
    "ScalerParameters",
    "TrainingStandardScaler",
    "StorageManifest",
    "MemmapStorageBuilder",
    "MemmapStorageReader",
    "STORAGE_DAT_FILENAME",
    "MANIFEST_FILENAME",
    "MARKER_FILENAME",
    "PortSequenceDataset",
    "PortSequenceDataModule",
    "create_dataloaders",
]
