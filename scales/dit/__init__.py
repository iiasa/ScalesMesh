"""Conditional diffusion emulator for monthly regional tas / pr time series
from a GMT trajectory.

Quick start
-----------
    from scales.dit import Config, train_from_sims, ScenarioSampler

    cfg = Config()
    cfg.train.out_dir = "runs/v1"
    cfg.train.max_steps = 200_000
    out = train_from_sims(sims, cfg, groups=scenario_labels)

    s = ScenarioSampler.from_checkpoint("runs/v1/last.pt", device="cuda")
    ens = s.sample(new_gmt_monthly, n_members=20)   # (20, 116, N)
"""

from common.zenodo import download_from_zenodo

from .config import Config, DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from .data import CropDataset, Normalizer, check_data, group_split
from .diffusion import Diffusion
from .inference import load_gmt, run_inference
from .model import ScalesDiT, build_model
from .sample import ScenarioSampler
from .train import train_from_sims

__all__ = [
    "Config", "DataConfig", "ModelConfig", "DiffusionConfig", "TrainConfig",
    "Normalizer", "CropDataset", "check_data", "group_split",
    "Diffusion", "ScalesDiT", "build_model",
    "train_from_sims", "ScenarioSampler",
    "load_gmt", "run_inference",
    "download_from_zenodo",
]
