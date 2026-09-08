from __future__ import annotations

import math
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    """Zero-mean, unit-variance normaliser for 3-D arrays of shape [N, T, D].

    Statistics are computed jointly over the N and T axes so that each of the
    D features is normalised independently across all samples and time steps.
    The fitted mean and std are retained and can be persisted to disk, making
    it straightforward to apply the same normalisation at inference time.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps: float = eps
        # Set by fit(); None until then.
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        """Compute and store per-feature mean and std from x [N, T, D]."""
        mean = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
        self.mean_ = mean
        # Clamp std to eps to avoid division by zero for constant features.
        self.std_ = np.maximum(std, self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardise x using the fitted mean and std."""
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Map standardised values back to the original scale."""
        return x * self.std_ + self.mean_

    def save(self, filepath: str) -> None:
        """Serialise the fitted scaler to a pickle file at filepath."""
        assert self.mean_ is not None and self.std_ is not None, \
            "Cannot save an unfitted StandardScaler"

        data = {
            "eps": self.eps,
            "mean_": self.mean_,
            "std_": self.std_,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_file(cls, filepath: str) -> StandardScaler:
        """Load and validate a previously saved scaler from filepath."""
        assert os.path.exists(filepath), f"File does not exist: {filepath}"

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        assert isinstance(data, dict), "Saved scaler data must be a dictionary"
        assert "eps" in data and "mean_" in data and "std_" in data, \
            "Saved scaler file is missing required keys"

        scaler = cls(eps=data["eps"])
        scaler.mean_ = data["mean_"]
        scaler.std_ = data["std_"]

        assert scaler.mean_.shape == scaler.std_.shape, \
            "mean_ and std_ must have the same shape"
        assert np.all(scaler.std_ > 0), \
            "All std_ values must be positive"

        return scaler


class MLP(nn.Module):
    """Two-hidden-layer MLP with SiLU activations.

    Architecture: Linear -> SiLU -> Linear -> SiLU -> Linear
    Used as a building block for the transition and emission networks inside
    the SSM, where smooth, non-saturating activations help gradient flow.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass x through the MLP. x: [..., in_dim] -> [..., out_dim]."""
        return self.net(x)


def diag_gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL divergence KL(q || p) for diagonal Gaussians, summed over the latent dim.

    Args:
        mu_q:     Mean of the posterior q.     Shape [B, Z].
        logvar_q: Log-variance of q.           Shape [B, Z].
        mu_p:     Mean of the prior p.         Shape [B, Z].
        logvar_p: Log-variance of p.           Shape [B, Z].

    Returns:
        Per-sample KL divergence. Shape [B].
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    # Closed-form KL between two diagonal Gaussians:
    # 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q - mu_p)^2) / var_p - 1 )
    return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0).sum(-1)
