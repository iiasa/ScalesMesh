"""Inference utilities for the SCALES models.

`SSMForecaster` loads a trained `DeepSSMPatternConditioned` checkpoint with its
fitted scalers and turns a GMT timeseries plus an observed tas context into a
tas/pr forecast in physical units.

`CnpForecaster` does the same for a CNP-conditioned model, loading everything
from one `train_cnp` run directory and additionally requiring a pr context,
from which the ESM embedding is inferred.
"""

from scales.inference.cnp_inference import (
    DEFAULT_COV_RANK,
    CnpForecaster,
    find_checkpoint,
    forecast_from_run_dir,
    infer_cnp_config,
)
from scales.inference.ssm_inference import (
    EXPECTED_DIMS,
    N_GMT_FEATURES,
    N_REGIONS,
    ForecastResult,
    SSMForecaster,
    forecast_from_checkpoint,
    infer_model_config,
)

__all__ = [
    "DEFAULT_COV_RANK",
    "EXPECTED_DIMS",
    "N_GMT_FEATURES",
    "N_REGIONS",
    "CnpForecaster",
    "ForecastResult",
    "SSMForecaster",
    "find_checkpoint",
    "forecast_from_checkpoint",
    "forecast_from_run_dir",
    "infer_cnp_config",
    "infer_model_config",
]
