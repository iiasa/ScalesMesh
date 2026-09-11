"""State-space models for SCALES, and the inference wrappers around them.

Modules
-------
ssm_model_utils
    Shared building blocks: `StandardScaler`, `MLP`, `diag_gaussian_kl`.
ssm_tas_pr
    `DeepSSMPatternConditioned` — a deep state-space model mapping a GMT
    forcing to regional tas and pr, with the sliding-window dataset, the linear
    pattern-scaling baseline and the DDP training loop `run_train`.
cnp_ssm
    `DeepCnpSsmforESM` — a Conditional Neural Process that infers an ESM
    embedding from a context set and FiLM-conditions the SSM's emission, plus
    the meta-learning task construction and training loop.
ssm_inference
    `SSMForecaster` — loads a trained SSM checkpoint with its scalers and turns
    a GMT timeseries plus a tas context into a forecast in physical units.
cnp_inference
    `CnpForecaster` — the same for a CNP-conditioned model, loading everything
    from one `train_cnp` run directory and additionally requiring a pr context.
"""

from scales.ssm.cnp_inference import (
    DEFAULT_COV_RANK,
    CnpForecaster,
    find_checkpoint,
    forecast_from_run_dir,
    infer_cnp_config,
)
from scales.ssm.cnp_ssm import (
    CnpLatentEncoder,
    ContextEncoderForCnp,
    DeepCnpSsmforESM,
    aggregate,
    build_task_dict,
    compute_task_loss,
    train_cnp,
)
from scales.ssm.ssm_inference import (
    EXPECTED_DIMS,
    N_GMT_FEATURES,
    N_REGIONS,
    ForecastResult,
    SSMForecaster,
    forecast_from_checkpoint,
    infer_model_config,
)
from scales.ssm.ssm_model_utils import MLP, StandardScaler, diag_gaussian_kl
from scales.ssm.ssm_tas_pr import (
    ALPHA_MAX,
    DeepSSMPatternConditioned,
    UnifiedWindowDataset,
    fit_control_mahalanobis,
    fit_ridge_D,
    load_into_ctrl_lin,
    mahalanobis_score,
    pick_threshold_from_val,
    run_train,
    sinh_arcsinh_flow_nll_conditional,
    sinh_arcsinh_forward,
)

__all__ = [
    # ssm_model_utils
    "MLP",
    "StandardScaler",
    "diag_gaussian_kl",
    # ssm_tas_pr
    "ALPHA_MAX",
    "DeepSSMPatternConditioned",
    "UnifiedWindowDataset",
    "fit_control_mahalanobis",
    "fit_ridge_D",
    "load_into_ctrl_lin",
    "mahalanobis_score",
    "pick_threshold_from_val",
    "run_train",
    "sinh_arcsinh_flow_nll_conditional",
    "sinh_arcsinh_forward",
    # cnp_ssm
    "CnpLatentEncoder",
    "ContextEncoderForCnp",
    "DeepCnpSsmforESM",
    "aggregate",
    "build_task_dict",
    "compute_task_loss",
    "train_cnp",
    # ssm_inference
    "EXPECTED_DIMS",
    "ForecastResult",
    "N_GMT_FEATURES",
    "N_REGIONS",
    "SSMForecaster",
    "forecast_from_checkpoint",
    "infer_model_config",
    # cnp_inference
    "CnpForecaster",
    "DEFAULT_COV_RANK",
    "find_checkpoint",
    "forecast_from_run_dir",
    "infer_cnp_config",
]
