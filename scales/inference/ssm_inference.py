"""Checkpoint-driven inference for DeepSSMPatternConditioned.

The training pipeline (`scales.model.ssm_tas_pr.run_train`) persists a model
state dict plus three fitted `StandardScaler`s (tas, gmt, pr). This module
glues those artefacts back together for forecasting:

    1. Rebuild the model from a checkpoint, inferring the architecture
       hyper-parameters from the state-dict shapes so no separate config file
       is required.
    2. Standardise a GMT timeseries and an observed tas context with the
       scalers loaded from disk.
    3. Run `forecast` (or `forecast_deterministic`) and map the predictive mean
       and 10/90th-percentile bands back to physical units.

The model is written in generic state-space terms — the target is `y` and the
exogenous control is `u` — while this module speaks the climate naming used at
the call site: `y` is near-surface air temperature (tas) and `u` is global mean
temperature (gmt).

Typical use:

    fc = SSMForecaster.from_checkpoint(
        "run/checkpoints/model_epoch0050.pt",
        tas_scaler_path="run/y_scaler.out",
        gmt_scaler_path="run/u_scaler.out",
        pr_scaler_path="run/pr_scaler.out",
    )
    out = fc.forecast(gmt, tas_context)     # gmt covers context + horizon
    out.tas_mean                            # [H, Dy] in physical units
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from scales.model.ssm_model_utils import StandardScaler
from scales.model.ssm_tas_pr import DeepSSMPatternConditioned

# Feature counts of the SCALES setup: tas (and pr) are resolved over 58
# regions, and gmt is a single global scalar per time step. The time axis is
# unconstrained — any context length and horizon the checkpoint can serve is
# accepted. Pass expected_dims=None to load a checkpoint with other dimensions.
N_REGIONS = 58
N_GMT_FEATURES = 1
EXPECTED_DIMS = (N_REGIONS, N_GMT_FEATURES)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Forecast for near-surface air temperature (tas) and precipitation (pr).

    All arrays are in physical units (the scalers have been inverted) and have
    shape [B, H, Dy], or [H, Dy] when the request was made with un-batched
    inputs.

    Attributes:
        tas_mean:    Predictive mean of tas.
        tas_q10:     10th percentile of tas.
        tas_q90:     90th percentile of tas.
        pr_mean:     Predictive mean of precipitation.
        pr_q10:      10th percentile of precipitation.
        pr_q90:      90th percentile of precipitation.
        context_len: Number of context steps consumed, Tc.
        horizon:     Number of forecast steps produced, H.
    """

    tas_mean: np.ndarray
    tas_q10: np.ndarray
    tas_q90: np.ndarray
    pr_mean: np.ndarray
    pr_q10: np.ndarray
    pr_q90: np.ndarray
    context_len: int
    horizon: int


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """Pull a flat parameter dict out of whatever `torch.load` returned.

    Accepts a bare state dict, a wrapper dict (as written by common training
    loops) or a DDP-saved dict whose keys carry a "module." prefix.

    Args:
        obj: Object returned by `torch.load`.

    Returns:
        Mapping from parameter name to tensor, with any "module." prefix stripped.
    """
    state = obj
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            inner = state.get(key)
            if isinstance(inner, dict):
                state = inner
                break

    if not isinstance(state, dict) or not all(
        isinstance(v, torch.Tensor) for v in state.values()
    ):
        raise ValueError(
            "Checkpoint does not contain a state dict of tensors; got "
            f"{type(obj).__name__}"
        )

    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state.items()}


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Recover DeepSSMPatternConditioned constructor arguments from a state dict.

    Every architectural hyper-parameter is determined by the shapes of the
    saved tensors, so a checkpoint is self-describing. Keys are the model's own
    generic names (y = tas, u = gmt):

        y_dim            emit_pr output is 4 * y_dim
        u_dim            inference GRU input is y_dim + u_dim
        z_dim            q_head output is 2 * z_dim
        rnn_hidden       inference GRU hidden state
        u_rnn_hidden     control GRU hidden state
        mlp_hidden       first emit layer width
        cov_rank         emit output is y_dim * (2 + cov_rank)
        reservoir_dim    log_alpha is [y_dim, reservoir_dim]
        emission_uses_u  u_gru present
        use_linear_model ctrl_lin present

    Args:
        state_dict: Flat parameter dict from `_extract_state_dict`.

    Returns:
        Keyword arguments for the DeepSSMPatternConditioned constructor.
    """
    try:
        y_dim      = state_dict["emit_pr.net.4.weight"].shape[0] // 4
        rnn_in     = state_dict["gru.weight_ih_l0"].shape[1]
        rnn_hidden = state_dict["gru.weight_hh_l0"].shape[1]
        z_dim      = state_dict["q_head.weight"].shape[0] // 2
        mlp_hidden = state_dict["emit.net.0.weight"].shape[0]
        emit_out   = state_dict["emit.net.4.weight"].shape[0]
    except KeyError as exc:
        raise ValueError(
            f"Checkpoint is missing the expected parameter {exc.args[0]!r}; "
            "it does not look like a DeepSSMPatternConditioned state dict"
        ) from None

    cfg: dict[str, Any] = {
        "y_dim":            y_dim,
        "u_dim":            rnn_in - y_dim,
        "z_dim":            z_dim,
        "rnn_hidden":       rnn_hidden,
        "mlp_hidden":       mlp_hidden,
        "cov_rank":         emit_out // y_dim - 2,
        "emission_uses_u":  "u_gru.weight_ih_l0" in state_dict,
        "use_linear_model": "ctrl_lin.weight" in state_dict,
    }

    if cfg["emission_uses_u"]:
        cfg["u_rnn_hidden"]  = state_dict["u_gru.weight_hh_l0"].shape[1]
        cfg["reservoir_dim"] = state_dict["log_alpha"].shape[1]

    return cfg


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _as_batched(
    x: np.ndarray,
    n_features: int,
    name: str,
    feature_label: str,
) -> tuple[np.ndarray, bool]:
    """Assert an input array's shape and coerce it to [B, T, D].

    The time axis is free — any context length or horizon is accepted — but
    the trailing feature axis is fixed by the model: tas and pr carry one
    value per region, gmt one value per global series.

    Accepts [T] (only when n_features == 1), [T, D] and [B, T, D].

    Args:
        x:             Input array.
        n_features:    Required trailing dimension D.
        name:          Argument name, used in the assertion messages.
        feature_label: What the trailing axis counts, e.g. "regions".

    Returns:
        (batched array [B, T, D], was_unbatched)

    Raises:
        AssertionError: If x is not numeric, has the wrong rank, has a
            trailing axis other than n_features, or has an empty time axis.
    """
    arr = np.asarray(x)
    assert np.issubdtype(arr.dtype, np.number), (
        f"{name} must be a numeric array; got dtype {arr.dtype}"
    )
    arr = arr.astype(np.float32, copy=False)

    assert 1 <= arr.ndim <= 3, (
        f"{name} must be 1-, 2- or 3-D ([T], [T, {n_features}] or "
        f"[B, T, {n_features}]); got shape {arr.shape}"
    )

    if arr.ndim == 1:
        assert n_features == 1, (
            f"{name} is 1-D but the model expects {n_features} {feature_label}; "
            f"pass an array of shape [T, {n_features}]"
        )
        arr = arr[:, None]

    unbatched = arr.ndim == 2
    if unbatched:
        arr = arr[None, ...]

    assert arr.shape[-1] == n_features, (
        f"{name} has {arr.shape[-1]} {feature_label} but the model expects "
        f"{n_features}; expected shape [T, {n_features}] or [B, T, {n_features}], "
        f"got {np.shape(x)}"
    )
    assert arr.shape[1] > 0, f"{name} has an empty time axis; got shape {np.shape(x)}"
    assert arr.shape[0] > 0, f"{name} has an empty batch axis; got shape {np.shape(x)}"

    return arr, unbatched


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class SSMForecaster:
    """A trained DeepSSMPatternConditioned paired with its fitted scalers.

    The model is trained on standardised data, so every entry point here takes
    and returns physical units and applies the scalers internally.
    """

    def __init__(
        self,
        model: DeepSSMPatternConditioned,
        tas_scaler: StandardScaler,
        gmt_scaler: StandardScaler,
        pr_scaler: StandardScaler,
        device: torch.device | None = None,
        expected_dims: tuple[int, int] | None = EXPECTED_DIMS,
    ) -> None:
        """
        Args:
            model:         Model in eval mode; moved to `device`.
            tas_scaler:    Scaler fitted on the tas target (the model's y).
            gmt_scaler:    Scaler fitted on the GMT control (the model's u).
            pr_scaler:     Scaler fitted on the precipitation proxy pr.
            device:        Torch device; defaults to CUDA when available, else CPU.
            expected_dims: (n_tas_regions, n_gmt_features) the model must have,
                           defaulting to the SCALES setup. None accepts whatever
                           the checkpoint declares.

        Raises:
            AssertionError: If the model dimensions disagree with
                `expected_dims`, or if a scaler was fitted on a different
                number of features than the model expects.
        """
        if expected_dims is not None:
            n_tas, n_gmt = expected_dims
            assert model.y_dim == n_tas, (
                f"model expects {model.y_dim} tas regions, not the {n_tas} of this "
                f"configuration; pass expected_dims=None to accept the checkpoint"
            )
            assert model.u_dim == n_gmt, (
                f"model expects {model.u_dim} gmt features, not the {n_gmt} of this "
                f"configuration; pass expected_dims=None to accept the checkpoint"
            )

        # A scaler fitted on a different number of features than the model
        # consumes would silently normalise the wrong regions.
        for name, scaler, n_features in (
            ("tas_scaler", tas_scaler, model.y_dim),
            ("gmt_scaler", gmt_scaler, model.u_dim),
            ("pr_scaler",  pr_scaler,  model.y_dim),
        ):
            assert scaler.mean_ is not None, f"{name} is not fitted"
            assert scaler.mean_.shape[-1] == n_features, (
                f"{name} was fitted on {scaler.mean_.shape[-1]} features but the "
                f"model expects {n_features}"
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model      = model.to(self.device).eval()
        self.tas_scaler = tas_scaler
        self.gmt_scaler = gmt_scaler
        self.pr_scaler  = pr_scaler

    # -- construction -------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tas_scaler_path: str,
        gmt_scaler_path: str,
        pr_scaler_path: str,
        device: torch.device | None = None,
        expected_dims: tuple[int, int] | None = EXPECTED_DIMS,
        model_kwargs: dict[str, Any] | None = None,
        strict: bool = True,
    ) -> SSMForecaster:
        """Load a model checkpoint and the three scalers from disk.

        Args:
            checkpoint_path: Path to a state dict saved by `run_train`.
            tas_scaler_path: Path to the pickled tas StandardScaler
                             (written as y_scaler.out by `run_train`).
            gmt_scaler_path: Path to the pickled GMT StandardScaler
                             (written as u_scaler.out by `run_train`).
            pr_scaler_path:  Path to the pickled pr StandardScaler.
            device:          Target device; defaults to CUDA when available.
            expected_dims:   (n_tas_regions, n_gmt_features) the checkpoint must
                             declare; None accepts any dimensions.
            model_kwargs:    Overrides for the inferred constructor arguments.
            strict:          Passed to `load_state_dict`; set False to tolerate
                             checkpoints with extra or missing keys.

        Returns:
            A ready-to-use SSMForecaster.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        state_dict = _extract_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=False))

        cfg = infer_model_config(state_dict)
        if model_kwargs:
            cfg.update(model_kwargs)

        model = DeepSSMPatternConditioned(**cfg)
        model.load_state_dict(state_dict, strict=strict)

        return cls(
            model=model,
            tas_scaler=StandardScaler.from_file(tas_scaler_path),
            gmt_scaler=StandardScaler.from_file(gmt_scaler_path),
            pr_scaler=StandardScaler.from_file(pr_scaler_path),
            device=device,
            expected_dims=expected_dims,
        )

    # -- forecasting --------------------------------------------------------

    @torch.no_grad()
    def forecast(
        self,
        gmt: np.ndarray,
        tas_context: np.ndarray,
        horizon: int | None = None,
        n_samples: int = 50,
        deterministic: bool = False,
    ) -> ForecastResult:
        """Forecast tas and pr from a GMT timeseries and an observed tas context.

        The GMT series must span the context *and* the forecast window: its
        first Tc steps (Tc = length of `tas_context`) are used to condition the
        encoder, and the remainder supplies the known future controls.

        Args:
            gmt:           GMT timeseries covering context + horizon, in
                           physical units. Shape [T], [T, Du] or [B, T, Du]
                           with T >= Tc + 1.
            tas_context:   Observed tas context in physical units.
                           Shape [Tc, Dy] or [B, Tc, Dy].
            horizon:       Number of steps to forecast; defaults to all GMT
                           steps beyond the context.
            n_samples:     Monte Carlo samples used for the mean and bands.
            deterministic: If True, use `forecast_deterministic` (emission
                           means only) instead of full ancestral sampling.

        Returns:
            A ForecastResult in physical units. Batch axes are dropped when
            both inputs were passed un-batched.

        Raises:
            AssertionError: If either input has the wrong rank or feature count,
                if the batch sizes are incompatible, or if gmt does not cover
                the context plus the requested horizon.
        """
        gmt_all, gmt_unbatched = _as_batched(
            gmt, self.model.u_dim, "gmt", "gmt features")
        tas_ctx, tas_unbatched = _as_batched(
            tas_context, self.model.y_dim, "tas_context", "tas regions")

        if gmt_all.shape[0] != tas_ctx.shape[0]:
            if gmt_all.shape[0] == 1:
                gmt_all = np.repeat(gmt_all, tas_ctx.shape[0], axis=0)
            elif tas_ctx.shape[0] == 1:
                tas_ctx = np.repeat(tas_ctx, gmt_all.shape[0], axis=0)
            else:
                raise AssertionError(
                    f"gmt batch size {gmt_all.shape[0]} is incompatible with "
                    f"tas_context batch size {tas_ctx.shape[0]}; they must match "
                    f"or one of them must be 1"
                )

        Tc        = tas_ctx.shape[1]
        available = gmt_all.shape[1] - Tc
        assert available >= 1, (
            f"gmt has {gmt_all.shape[1]} steps but the context needs {Tc}; "
            "it must also cover at least one forecast step"
        )

        steps = available if horizon is None else int(horizon)
        assert 1 <= steps <= available, (
            f"horizon must be between 1 and {available} for this gmt series; got {steps}"
        )

        # Standardise with the training scalers, then split the GMT series into
        # the context part and the known future part.
        gmt_norm = self.gmt_scaler.transform(gmt_all)
        tas_norm = self.tas_scaler.transform(tas_ctx)

        tas_ctx_t = torch.as_tensor(tas_norm,                     dtype=torch.float32, device=self.device)
        gmt_ctx_t = torch.as_tensor(gmt_norm[:, :Tc],             dtype=torch.float32, device=self.device)
        gmt_fut_t = torch.as_tensor(gmt_norm[:, Tc:Tc + steps],   dtype=torch.float32, device=self.device)

        run = (self.model.forecast_deterministic if deterministic
               else self.model.forecast)
        tas_mean, tas_q10, tas_q90, pr_mean, pr_q10, pr_q90 = run(
            tas_ctx_t, gmt_ctx_t, gmt_fut_t, steps=steps, n_samples=n_samples)

        # Back to physical units; the affine inverse preserves quantile order.
        def to_tas(t: torch.Tensor) -> np.ndarray:
            return self.tas_scaler.inverse_transform(t.cpu().numpy())

        def to_pr(t: torch.Tensor) -> np.ndarray:
            return self.pr_scaler.inverse_transform(t.cpu().numpy())

        out = [to_tas(tas_mean), to_tas(tas_q10), to_tas(tas_q90),
               to_pr(pr_mean),   to_pr(pr_q10),   to_pr(pr_q90)]

        if gmt_unbatched and tas_unbatched:
            out = [a[0] for a in out]

        return ForecastResult(*out, context_len=Tc, horizon=steps)


def forecast_from_checkpoint(
    checkpoint_path: str,
    tas_scaler_path: str,
    gmt_scaler_path: str,
    pr_scaler_path: str,
    gmt: np.ndarray,
    tas_context: np.ndarray,
    horizon: int | None = None,
    n_samples: int = 50,
    deterministic: bool = False,
    device: torch.device | None = None,
    expected_dims: tuple[int, int] | None = EXPECTED_DIMS,
    model_kwargs: dict[str, Any] | None = None,
) -> ForecastResult:
    """One-shot convenience wrapper: load, normalise and forecast.

    Rebuilds the forecaster on every call, so prefer `SSMForecaster.from_checkpoint`
    when forecasting repeatedly from the same checkpoint.

    Args:
        checkpoint_path: Path to the model state dict.
        tas_scaler_path: Path to the pickled tas StandardScaler.
        gmt_scaler_path: Path to the pickled GMT StandardScaler.
        pr_scaler_path:  Path to the pickled pr StandardScaler.
        gmt:             GMT timeseries covering context + horizon.
        tas_context:     Observed tas context.
        horizon:         Forecast steps; defaults to all GMT steps past the context.
        n_samples:       Monte Carlo samples for the mean and bands.
        deterministic:   Use the mean-path forecast instead of sampling.
        device:          Target device; defaults to CUDA when available.
        expected_dims:   (n_tas_regions, n_gmt_features) the checkpoint must
                         declare; None accepts any dimensions.
        model_kwargs:    Overrides for the inferred constructor arguments.

    Returns:
        A ForecastResult in physical units.
    """
    forecaster = SSMForecaster.from_checkpoint(
        checkpoint_path,
        tas_scaler_path=tas_scaler_path,
        gmt_scaler_path=gmt_scaler_path,
        pr_scaler_path=pr_scaler_path,
        device=device,
        expected_dims=expected_dims,
        model_kwargs=model_kwargs,
    )
    return forecaster.forecast(
        gmt, tas_context, horizon=horizon,
        n_samples=n_samples, deterministic=deterministic,
    )
