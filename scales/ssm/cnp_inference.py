"""Checkpoint-directory inference for DeepCnpSsmforESM.

`train_cnp` and `build_task_dict` write everything a forecast needs into one
run directory:

    run_dir/
        y_scaler.out            tas scaler   (from build_task_dict)
        u_scaler.out            gmt scaler
        pr_scaler.out           pr scaler
        cnp_model_out           final model  (from train_cnp)
        checkpoints/
            cnp_epoch0010.pt    periodic snapshots
            cnp_epoch0020.pt

so `CnpForecaster.from_run_dir(run_dir)` is enough to rebuild the model, its
wrapped SSM and all three scalers. The architecture is read back from the
state-dict shapes, exactly as `ssm_inference.infer_model_config` does for a
bare SSM, so no separate config file is needed.

Unlike the plain SSM, the CNP needs a **pr context** as well as a tas context:
the ESM embedding is inferred from (gmt, tas, pr) triples, so `pr_context` is
an explicit, required argument of `forecast`.

Typical use:

    fc  = CnpForecaster.from_run_dir("runs/scales_cnp_20260728")
    out = fc.forecast(gmt, tas_context, pr_context)   # gmt covers context + horizon
    out.tas_mean                                      # [H, Dy] in physical units
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import torch

from scales.ssm.cnp_ssm import DeepCnpSsmforESM
from scales.ssm.ssm_inference import (
    EXPECTED_DIMS,
    ForecastResult,
    _as_batched,
    _extract_state_dict,
    infer_model_config,
)
from scales.ssm.ssm_model_utils import StandardScaler
from scales.ssm.ssm_tas_pr import DeepSSMPatternConditioned

# Filenames written by build_task_dict / train_cnp, relative to the run dir.
TAS_SCALER_FILE = "y_scaler.out"
GMT_SCALER_FILE = "u_scaler.out"
PR_SCALER_FILE = "pr_scaler.out"
FINAL_MODEL_FILE = "cnp_model_out"
CHECKPOINT_GLOB = os.path.join("checkpoints", "cnp_epoch*.pt")

# The CNP is built on the diagonal-emission SSM, so that is the expected rank.
DEFAULT_COV_RANK = 0


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoint(run_dir: str, epoch: int | None = None) -> str:
    """Locate a CNP checkpoint inside a run directory.

    Args:
        run_dir: Directory written by `train_cnp`.
        epoch:   Specific epoch to load from `checkpoints/cnp_epoch<NNNN>.pt`.
                 When None, the final `cnp_model_out` is preferred, falling
                 back to the highest-numbered epoch snapshot.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If the run directory or the requested checkpoint
            does not exist.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if epoch is not None:
        path = os.path.join(run_dir, "checkpoints", f"cnp_epoch{epoch:04d}.pt")
        if not os.path.exists(path):
            available = sorted(os.path.basename(p)
                               for p in glob.glob(os.path.join(run_dir, CHECKPOINT_GLOB)))
            raise FileNotFoundError(
                f"No checkpoint for epoch {epoch} in {run_dir}; "
                f"available: {available or 'none'}"
            )
        return path

    final = os.path.join(run_dir, FINAL_MODEL_FILE)
    if os.path.exists(final):
        return final

    # Zero-padded epoch numbers sort lexicographically, so max() is the latest.
    snapshots = glob.glob(os.path.join(run_dir, CHECKPOINT_GLOB))
    if not snapshots:
        raise FileNotFoundError(
            f"No CNP checkpoint in {run_dir}: expected {FINAL_MODEL_FILE} or "
            f"{CHECKPOINT_GLOB}"
        )
    return max(snapshots)


def infer_cnp_config(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Recover DeepCnpSsmforESM constructor arguments from a state dict.

    The CNP-specific widths follow from the encoder shapes:

        z_cnp_dim       lat_encoder.to_mu is [z_cnp_dim, r_dim]
        r_dim           same tensor, input side
        encoder_hidden  widths of the ctx_encoder.mlp linear layers

    Args:
        state_dict: Flat CNP parameter dict, "ssm." prefixes included.

    Returns:
        Keyword arguments for DeepCnpSsmforESM, excluding `ssm_model`.

    Raises:
        ValueError: If the expected CNP parameters are absent.
    """
    try:
        z_cnp_dim, r_dim = state_dict["lat_encoder.to_mu.weight"].shape
    except KeyError as exc:
        raise ValueError(
            f"Checkpoint is missing the expected parameter {exc.args[0]!r}; "
            "it does not look like a DeepCnpSsmforESM state dict"
        ) from None

    # ctx_encoder.mlp is Linear/ReLU/Linear/ReLU/.../Linear; the hidden widths
    # are the output sizes of every layer but the last.
    layer_out = [
        (int(key.split(".")[2]), tensor.shape[0])
        for key, tensor in state_dict.items()
        if key.startswith("ctx_encoder.mlp.") and key.endswith(".weight")
    ]
    hidden = tuple(width for _, width in sorted(layer_out)[:-1])

    return {"r_dim": int(r_dim), "z_cnp_dim": int(z_cnp_dim), "encoder_hidden": hidden}


def _split_ssm_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract the wrapped SSM's parameters from a CNP state dict.

    Args:
        state_dict: Flat CNP parameter dict.

    Returns:
        The "ssm."-prefixed entries with the prefix stripped.

    Raises:
        ValueError: If no SSM parameters are present.
    """
    prefix = "ssm."
    ssm_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not ssm_state:
        raise ValueError(
            "Checkpoint contains no 'ssm.'-prefixed parameters; it does not "
            "look like a DeepCnpSsmforESM state dict"
        )
    return ssm_state


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class CnpForecaster:
    """A trained DeepCnpSsmforESM paired with its fitted scalers.

    Takes and returns physical units; the scalers are applied internally. The
    ESM embedding is inferred from the (gmt, tas, pr) context, so a pr context
    is required alongside the tas one.
    """

    def __init__(
        self,
        model: DeepCnpSsmforESM,
        tas_scaler: StandardScaler,
        gmt_scaler: StandardScaler,
        pr_scaler: StandardScaler,
        device: torch.device | None = None,
        expected_dims: tuple[int, int] | None = EXPECTED_DIMS,
        cov_rank: int | None = DEFAULT_COV_RANK,
    ) -> None:
        """
        Args:
            model:         CNP model; moved to `device` and put in eval mode.
            tas_scaler:    Scaler fitted on tas (the SSM's y).
            gmt_scaler:    Scaler fitted on the GMT forcing (the SSM's u).
            pr_scaler:     Scaler fitted on pr.
            device:        Torch device; defaults to CUDA when available.
            expected_dims: (n_tas_regions, n_gmt_features) the wrapped SSM must
                           have; None accepts whatever the checkpoint declares.
            cov_rank:      Covariance rank the wrapped SSM must have, 0 for the
                           diagonal emission the CNP is built on; None accepts
                           whatever the checkpoint declares.

        Raises:
            AssertionError: If the SSM dimensions or covariance rank disagree
                with what was asked for, or if a scaler was fitted on a
                different number of features than the model expects.
        """
        ssm = model.ssm

        if expected_dims is not None:
            n_tas, n_gmt = expected_dims
            assert ssm.y_dim == n_tas, (
                f"wrapped SSM expects {ssm.y_dim} tas regions, not the {n_tas} of "
                f"this configuration; pass expected_dims=None to accept the checkpoint"
            )
            assert ssm.u_dim == n_gmt, (
                f"wrapped SSM expects {ssm.u_dim} gmt features, not the {n_gmt} of "
                f"this configuration; pass expected_dims=None to accept the checkpoint"
            )

        if cov_rank is not None:
            assert ssm.cov_rank == cov_rank, (
                f"wrapped SSM has cov_rank={ssm.cov_rank}, not the {cov_rank} of this "
                f"configuration; pass cov_rank=None to accept the checkpoint"
            )

        # A scaler fitted on a different feature count than the model consumes
        # would silently normalise the wrong regions.
        for name, scaler, n_features in (
            ("tas_scaler", tas_scaler, ssm.y_dim),
            ("gmt_scaler", gmt_scaler, ssm.u_dim),
            ("pr_scaler",  pr_scaler,  ssm.y_dim),
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
    def from_run_dir(
        cls,
        run_dir: str,
        epoch: int | None = None,
        device: torch.device | None = None,
        expected_dims: tuple[int, int] | None = EXPECTED_DIMS,
        cov_rank: int | None = DEFAULT_COV_RANK,
        model_kwargs: dict[str, Any] | None = None,
        strict: bool = True,
    ) -> CnpForecaster:
        """Load a model and all three scalers from one `train_cnp` run directory.

        Args:
            run_dir:       Directory holding the scalers and checkpoints.
            epoch:         Load `checkpoints/cnp_epoch<NNNN>.pt` for this epoch;
                           None takes `cnp_model_out`, else the latest snapshot.
            device:        Target device; defaults to CUDA when available.
            expected_dims: (n_tas_regions, n_gmt_features) the SSM must declare.
            cov_rank:      Covariance rank the SSM must declare; defaults to 0.
            model_kwargs:  Overrides for the inferred SSM constructor arguments.
            strict:        Passed to `load_state_dict`; False tolerates extra or
                           missing keys.

        Returns:
            A ready-to-use CnpForecaster.

        Raises:
            FileNotFoundError: If the run directory, a scaler or the requested
                checkpoint is missing.
        """
        checkpoint_path = find_checkpoint(run_dir, epoch=epoch)

        paths = {
            "tas_scaler_path": os.path.join(run_dir, TAS_SCALER_FILE),
            "gmt_scaler_path": os.path.join(run_dir, GMT_SCALER_FILE),
            "pr_scaler_path":  os.path.join(run_dir, PR_SCALER_FILE),
        }
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Run directory is missing scaler file(s): "
                + ", ".join(os.path.basename(p) for p in missing)
                + f" (looked in {run_dir})"
            )

        return cls.from_checkpoint(
            checkpoint_path,
            device=device,
            expected_dims=expected_dims,
            cov_rank=cov_rank,
            model_kwargs=model_kwargs,
            strict=strict,
            **paths,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tas_scaler_path: str,
        gmt_scaler_path: str,
        pr_scaler_path: str,
        device: torch.device | None = None,
        expected_dims: tuple[int, int] | None = EXPECTED_DIMS,
        cov_rank: int | None = DEFAULT_COV_RANK,
        model_kwargs: dict[str, Any] | None = None,
        strict: bool = True,
    ) -> CnpForecaster:
        """Load a CNP checkpoint and three scalers from explicit paths.

        Use `from_run_dir` when the artefacts sit together in a `train_cnp` run
        directory; this is the escape hatch for scalers kept elsewhere.

        Args:
            checkpoint_path: Path to a DeepCnpSsmforESM state dict.
            tas_scaler_path: Path to the pickled tas StandardScaler.
            gmt_scaler_path: Path to the pickled GMT StandardScaler.
            pr_scaler_path:  Path to the pickled pr StandardScaler.
            device:          Target device; defaults to CUDA when available.
            expected_dims:   (n_tas_regions, n_gmt_features) the SSM must declare.
            cov_rank:        Covariance rank the SSM must declare; defaults to 0.
            model_kwargs:    Overrides for the inferred SSM constructor arguments.
            strict:          Passed to `load_state_dict`.

        Returns:
            A ready-to-use CnpForecaster.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        state_dict = _extract_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=False))

        # Rebuild the wrapped SSM from its own slice of the state dict, then the
        # CNP modules from theirs.
        ssm_cfg = infer_model_config(_split_ssm_state(state_dict))
        if model_kwargs:
            ssm_cfg.update(model_kwargs)

        model = DeepCnpSsmforESM(
            ssm_model=DeepSSMPatternConditioned(**ssm_cfg),
            **infer_cnp_config(state_dict),
        )
        model.load_state_dict(state_dict, strict=strict)

        return cls(
            model=model,
            tas_scaler=StandardScaler.from_file(tas_scaler_path),
            gmt_scaler=StandardScaler.from_file(gmt_scaler_path),
            pr_scaler=StandardScaler.from_file(pr_scaler_path),
            device=device,
            expected_dims=expected_dims,
            cov_rank=cov_rank,
        )

    # -- forecasting --------------------------------------------------------

    @torch.no_grad()
    def forecast(
        self,
        gmt: np.ndarray,
        tas_context: np.ndarray,
        pr_context: np.ndarray,
        horizon: int | None = None,
        n_samples: int = 50,
        deterministic: bool = False,
        override_esm: bool = False,
    ) -> ForecastResult:
        """Forecast tas and pr from a GMT series and an observed tas/pr context.

        The GMT series must span the context *and* the forecast window: its
        first Tc steps (Tc = length of `tas_context`) condition the SSM encoder
        and, together with the tas and pr contexts, the CNP encoder; the
        remainder supplies the known future forcings.

        Args:
            gmt:           GMT timeseries covering context + horizon, in
                           physical units. Shape [T], [T, Du] or [B, T, Du]
                           with T >= Tc + 1.
            tas_context:   Observed tas context.  Shape [Tc, Dy] or [B, Tc, Dy].
            pr_context:    Observed pr context, same shape as `tas_context`.
                           Required: the ESM embedding is inferred from
                           (gmt, tas, pr) triples.
            horizon:       Steps to forecast; defaults to every GMT step past
                           the context.
            n_samples:     Monte Carlo samples for the mean and bands.
            deterministic: Use the mean-path rollout instead of sampling the
                           emissions.
            override_esm:  Draw a fresh embedding z_cnp ~ N(0, I) per sample, so
                           the bands marginalise over ESM identity as well as
                           over the model's own stochasticity. The tas and pr
                           contexts are then unused by the CNP encoder.

        Returns:
            A ForecastResult in physical units. Batch axes are dropped when all
            inputs were passed un-batched.

        Raises:
            AssertionError: If an input has the wrong rank or feature count, if
                the contexts disagree in shape, if the batch sizes are
                incompatible, or if gmt does not cover the context plus horizon.
        """
        ssm = self.model.ssm

        gmt_all, gmt_unbatched = _as_batched(gmt,         ssm.u_dim, "gmt",         "gmt features")
        tas_ctx, tas_unbatched = _as_batched(tas_context, ssm.y_dim, "tas_context", "tas regions")
        pr_ctx,  pr_unbatched  = _as_batched(pr_context,  ssm.y_dim, "pr_context",  "pr regions")

        assert tas_ctx.shape == pr_ctx.shape, (
            f"tas_context and pr_context must have the same shape; got "
            f"{np.shape(tas_context)} and {np.shape(pr_context)}"
        )

        # Broadcast a single series over a batch of the other input.
        batch_sizes = {gmt_all.shape[0], tas_ctx.shape[0]}
        if len(batch_sizes) > 1:
            B = max(batch_sizes)
            assert batch_sizes == {1, B}, (
                f"gmt batch size {gmt_all.shape[0]} is incompatible with "
                f"tas_context batch size {tas_ctx.shape[0]}; they must match or "
                f"one of them must be 1"
            )
            if gmt_all.shape[0] == 1:
                gmt_all = np.repeat(gmt_all, B, axis=0)
            else:
                tas_ctx = np.repeat(tas_ctx, B, axis=0)
                pr_ctx  = np.repeat(pr_ctx,  B, axis=0)

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
        pr_norm  = self.pr_scaler.transform(pr_ctx)

        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a, dtype=torch.float32, device=self.device)

        run = (self.model.forecast_deterministic if deterministic
               else self.model.forecast)
        tas_mean, tas_q10, tas_q90, pr_mean, pr_q10, pr_q90 = run(
            tensor(tas_norm),
            tensor(gmt_norm[:, :Tc]),
            tensor(gmt_norm[:, Tc:Tc + steps]),
            steps=steps,
            n_samples=n_samples,
            pr_ctx=tensor(pr_norm),
            override_esm=override_esm,
        )

        # Back to physical units; the affine inverse preserves quantile order.
        def to_tas(t: torch.Tensor) -> np.ndarray:
            return self.tas_scaler.inverse_transform(t.cpu().numpy())

        def to_pr(t: torch.Tensor) -> np.ndarray:
            return self.pr_scaler.inverse_transform(t.cpu().numpy())

        out = [to_tas(tas_mean), to_tas(tas_q10), to_tas(tas_q90),
               to_pr(pr_mean),   to_pr(pr_q10),   to_pr(pr_q90)]

        if gmt_unbatched and tas_unbatched and pr_unbatched:
            out = [a[0] for a in out]

        return ForecastResult(*out, context_len=Tc, horizon=steps)


def forecast_from_run_dir(
    run_dir: str,
    gmt: np.ndarray,
    tas_context: np.ndarray,
    pr_context: np.ndarray,
    horizon: int | None = None,
    n_samples: int = 50,
    deterministic: bool = False,
    override_esm: bool = False,
    epoch: int | None = None,
    device: torch.device | None = None,
    expected_dims: tuple[int, int] | None = EXPECTED_DIMS,
    cov_rank: int | None = DEFAULT_COV_RANK,
    model_kwargs: dict[str, Any] | None = None,
) -> ForecastResult:
    """One-shot convenience wrapper: load a run directory, normalise and forecast.

    Rebuilds the forecaster on every call, so prefer `CnpForecaster.from_run_dir`
    when forecasting repeatedly from the same run.

    Args:
        run_dir:       Directory written by `train_cnp`.
        gmt:           GMT timeseries covering context + horizon.
        tas_context:   Observed tas context.
        pr_context:    Observed pr context, same shape as `tas_context`.
        horizon:       Forecast steps; defaults to all GMT steps past the context.
        n_samples:     Monte Carlo samples for the mean and bands.
        deterministic: Use the mean-path rollout instead of sampling.
        override_esm:  Marginalise over ESM identity instead of encoding the context.
        epoch:         Specific epoch snapshot to load; None takes the final model.
        device:        Target device; defaults to CUDA when available.
        expected_dims: (n_tas_regions, n_gmt_features) the SSM must declare.
        cov_rank:      Covariance rank the SSM must declare; defaults to 0.
        model_kwargs:  Overrides for the inferred SSM constructor arguments.

    Returns:
        A ForecastResult in physical units.
    """
    forecaster = CnpForecaster.from_run_dir(
        run_dir,
        epoch=epoch,
        device=device,
        expected_dims=expected_dims,
        cov_rank=cov_rank,
        model_kwargs=model_kwargs,
    )
    return forecaster.forecast(
        gmt, tas_context, pr_context,
        horizon=horizon, n_samples=n_samples,
        deterministic=deterministic, override_esm=override_esm,
    )
