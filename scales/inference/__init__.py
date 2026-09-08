"""Inference utilities for the SCALES models.

Exposes `SSMForecaster`, which loads a trained `DeepSSMPatternConditioned`
checkpoint together with its fitted scalers and turns a GMT timeseries plus an
observed tas context into a tas/pr forecast in physical units.
"""

from scales.inference.ssm_inference import (
    EXPECTED_DIMS,
    N_GMT_FEATURES,
    N_REGIONS,
    ForecastResult,
    SSMForecaster,
    forecast_from_checkpoint,
)

__all__ = [
    "EXPECTED_DIMS",
    "N_GMT_FEATURES",
    "N_REGIONS",
    "ForecastResult",
    "SSMForecaster",
    "forecast_from_checkpoint",
]
