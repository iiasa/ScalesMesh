"""Conditional Neural Process models for the SCALES SSM.

Exposes `DeepCnpSsmforESM`, which conditions a trained
`DeepSSMPatternConditioned` on an ESM embedding inferred from a context set,
together with the task construction and training helpers around it.
"""

from scales.cnp.cnp_ssm import (
    CnpLatentEncoder,
    ContextEncoderForCnp,
    DeepCnpSsmforESM,
    aggregate,
    build_task_dict,
    compute_task_loss,
    train_cnp,
)

__all__ = [
    "CnpLatentEncoder",
    "ContextEncoderForCnp",
    "DeepCnpSsmforESM",
    "aggregate",
    "build_task_dict",
    "compute_task_loss",
    "train_cnp",
]
