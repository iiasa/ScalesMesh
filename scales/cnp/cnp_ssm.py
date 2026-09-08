"""Conditional Neural Process (CNP) wrapper around DeepSSMPatternConditioned.

The SSM in `scales.model.ssm_tas_pr` learns one shared tas/pr response to a GMT
forcing. Different Earth System Models (ESMs) respond differently to the same
forcing, so this module adds a CNP that infers an ESM embedding from a context
set of (u, tas, pr) points and uses it to modulate the SSM's emission:

    context set  --> ContextEncoderForCnp --> mean-aggregate --> CnpLatentEncoder
                                                                      |
                                                                   z_cnp
                                                                      |
    e_in (SSM emission input)  -->  gamma(z_cnp) * e_in + beta(z_cnp)  -->  emit

The FiLM (Feature-wise Linear Modulation) step lets the embedding both rescale
and shift how the emission networks read the current SSM state, without
touching the latent dynamics in z.

Training objective:

    nll_tas + nll_pr + kl_ssm_w * KL(q(z) || p(z)) + kl_cnp_w * KL(q(z_cnp) || N(0, I))
    + alpha * huber(tas rollout) + omega * huber(pr rollout)
    + gamma * MSE(yearly-mean tas)

Meta-learning across ESMs is set up by `build_task_dict`, which splits each
ESM's scenarios into a support (train) and query (validation) set and fits one
set of global scalers on the pooled support data, and driven by `train_cnp`.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import scales.model.ssm_model_utils as utils
from scales.model.ssm_tas_pr import (
    DeepSSMPatternConditioned,
    UnifiedWindowDataset,
    sinh_arcsinh_flow_nll_conditional,
    sinh_arcsinh_forward,
)

# One batch from UnifiedWindowDataset: (y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut).
Batch = tuple[torch.Tensor, ...]


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------

def build_task_dict(
    esm_data: dict[str, dict[str, Any]],
    context_len: int = 600,
    horizon: int = 1200,
    stride: int = 1,
    support_frac: float = 0.7,
    start_mode: str = "all",
    run_dir: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, utils.StandardScaler]]:
    """Build a meta-learning task dictionary from multi-ESM data.

    Each ESM becomes one task, split by scenario index into a support set (used
    for training) and a query set (used for validation). Scalers are fitted on
    the pooled support data across all ESMs, so every ESM is normalised
    consistently and no query data leaks into the statistics — the same
    train-only convention `run_train` uses.

    Args:
        esm_data:     Per-ESM data, keyed by ESM name. Each value holds
                      'y'  [N_scenarios, T, Dy] regional tas anomalies,
                      'pr' [N_scenarios, T, Dy] regional pr anomalies,
                      'u'  [N_scenarios, T, Du] GMT forcing, and an optional
                      'weight' (float) used during meta-learning. 2-D arrays
                      are promoted to a single scenario.
        context_len:  Context window length Tc for UnifiedWindowDataset.
        horizon:      Forecast horizon H for UnifiedWindowDataset.
        stride:       Step between consecutive window starts.
        support_frac: Fraction of scenarios (by index) forming the support set;
                      the remainder become the query set.
        start_mode:   Window sampling mode, "all" or "zero".
        run_dir:      If given, the three global scalers are written there as
                      y_scaler.out / pr_scaler.out / u_scaler.out, matching the
                      filenames `run_train` and `SSMForecaster` expect.

    Returns:
        tasks:   {esm_name: {'support': UnifiedWindowDataset,
                             'query':   UnifiedWindowDataset,
                             'weight':  float}} over normalised data.
        scalers: {'y': StandardScaler, 'pr': StandardScaler, 'u': StandardScaler}.

    Raises:
        ValueError: If an ESM has too few scenarios to yield a query set.
    """
    # ---- Pass 1: validate shapes and compute the support/query split ----
    raw: dict[str, dict[str, Any]] = {}
    for esm_name, data in esm_data.items():
        y  = np.asarray(data["y"],  dtype=np.float32)
        pr = np.asarray(data["pr"], dtype=np.float32)
        u  = np.asarray(data["u"],  dtype=np.float32)

        if y.ndim == 2:
            y, pr, u = y[None], pr[None], u[None]

        N         = y.shape[0]
        n_support = max(1, int(np.floor(support_frac * N)))

        if N - n_support < 1:
            raise ValueError(
                f"ESM '{esm_name}' has only {N} scenario(s); support_frac="
                f"{support_frac} leaves no scenarios for the query set."
            )

        raw[esm_name] = {
            "y": y, "pr": pr, "u": u,
            "weight": float(data.get("weight", 1.0)),
            "n_support": n_support,
        }

    # ---- Fit global scalers on the pooled support data from all ESMs ----
    y_scaler  = utils.StandardScaler().fit(
        np.concatenate([v["y"][:v["n_support"]]  for v in raw.values()], axis=0))
    pr_scaler = utils.StandardScaler().fit(
        np.concatenate([v["pr"][:v["n_support"]] for v in raw.values()], axis=0))
    u_scaler  = utils.StandardScaler().fit(
        np.concatenate([v["u"][:v["n_support"]]  for v in raw.values()], axis=0))

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        y_scaler.save(os.path.join(run_dir,  "y_scaler.out"))
        pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
        u_scaler.save(os.path.join(run_dir,  "u_scaler.out"))

    # ---- Pass 2: normalise and build the sliding-window datasets ----
    window_kwargs = {
        "context_len": context_len,
        "horizon":     horizon,
        "stride":      stride,
        "start_mode":  start_mode,
    }

    tasks: dict[str, dict[str, Any]] = {}
    for esm_name, d in raw.items():
        n_sup = d["n_support"]
        y_n   = y_scaler.transform(d["y"])
        pr_n  = pr_scaler.transform(d["pr"])
        u_n   = u_scaler.transform(d["u"])

        tasks[esm_name] = {
            "support": UnifiedWindowDataset(
                y_n[:n_sup], pr_n[:n_sup], u_n[:n_sup], **window_kwargs),
            "query": UnifiedWindowDataset(
                y_n[n_sup:], pr_n[n_sup:], u_n[n_sup:], **window_kwargs),
            "weight": d["weight"],
        }

    return tasks, {"y": y_scaler, "pr": pr_scaler, "u": u_scaler}


# ---------------------------------------------------------------------------
# CNP encoder stack
# ---------------------------------------------------------------------------

def _relu_mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """Build a ReLU MLP with the given hidden widths.

    Kept separate from `utils.MLP` (which is SiLU) so the CNP encoder keeps the
    activation it was trained with; the layer shapes are otherwise identical.

    Args:
        in_dim:  Input width.
        out_dim: Output width.
        hidden:  Hidden layer widths, outermost first.

    Returns:
        Linear -> ReLU -> ... -> Linear, with no activation on the output.
    """
    dims   = [in_dim, *hidden, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ContextEncoderForCnp(nn.Module):
    """Encode one (forcing, state) pair into a fixed-width representation r.

    Applied independently to every point in the context set — no attention and
    no ordering, which is what makes the aggregation permutation-invariant. The
    state is all climate variables concatenated, i.e. cat([tas, pr], dim=-1).
    """

    def __init__(self, x_dim: int, y_dim: int, r_dim: int,
                 hidden: tuple[int, ...] = (256, 256)) -> None:
        """
        Args:
            x_dim:  Forcing width (the SSM's u_dim).
            y_dim:  State width (2 * the SSM's y_dim for [tas, pr]).
            r_dim:  Representation width.
            hidden: Hidden layer widths of the encoder MLP.
        """
        super().__init__()
        self.mlp = _relu_mlp(x_dim + y_dim, r_dim, hidden)

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor) -> torch.Tensor:
        """Encode a context set.

        Args:
            x_context: Forcings.  Shape [B, N_ctx, x_dim].
            y_context: States.    Shape [B, N_ctx, y_dim].

        Returns:
            Per-point representations.  Shape [B, N_ctx, r_dim].
        """
        return self.mlp(torch.cat([x_context, y_context], dim=-1))


def aggregate(r_context: torch.Tensor) -> torch.Tensor:
    """Collapse N_ctx representations into one, permutation-invariantly.

    The mean is the canonical CNP choice: it is order-independent and handles a
    context set of any size.

    Args:
        r_context: Per-point representations.  Shape [B, N_ctx, r_dim].

    Returns:
        Aggregated representation.  Shape [B, r_dim].
    """
    return r_context.mean(dim=1)


class CnpLatentEncoder(nn.Module):
    """Map an aggregated representation to a reparameterised latent z_cnp."""

    def __init__(self, r_dim: int, z_dim: int) -> None:
        """
        Args:
            r_dim: Aggregated representation width.
            z_dim: Latent width.
        """
        super().__init__()
        self.to_mu    = nn.Linear(r_dim, z_dim)
        self.to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r_agg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample z_cnp ~ N(mu, sigma^2) with the reparameterisation trick.

        Args:
            r_agg: Aggregated representation.  Shape [B, r_dim].

        Returns:
            (z, mu, sigma) — each [B, z_dim]. sigma is floored at 0.1 so the
            posterior cannot collapse to a point mass early in training.
        """
        mu    = self.to_mu(r_agg)
        sigma = 0.1 + 0.9 * F.softplus(self.to_sigma(r_agg))
        z     = mu + sigma * torch.randn_like(sigma)
        return z, mu, sigma


# ---------------------------------------------------------------------------
# CNP-conditioned SSM
# ---------------------------------------------------------------------------

class DeepCnpSsmforESM(nn.Module):
    """A DeepSSMPatternConditioned whose emission is FiLM-conditioned on an ESM embedding.

    The wrapped SSM is used as a library of parts rather than called through its
    own `forward_elbo`: the rollout here has to inject the FiLM step between
    building `e_in` and calling `emit`, which the SSM's own loop does not expose.
    Emission distributions are still built by the SSM's `_parse_emit` and
    `_emission_dist`, so a diagonal (cov_rank=0) and a low-rank SSM both work
    and the density matches `ssm_tas_pr` exactly.
    """

    def __init__(
        self,
        ssm_model: DeepSSMPatternConditioned,
        r_dim: int,
        z_cnp_dim: int,
        encoder_hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        """
        Args:
            ssm_model:      The SSM to wrap. Its parameters are part of this module.
            r_dim:          Width of the per-point CNP representation.
            z_cnp_dim:      Width of the ESM embedding z_cnp.
            encoder_hidden: Hidden widths of the context encoder MLP.
        """
        super().__init__()
        self.ssm       = ssm_model
        self.z_cnp_dim = z_cnp_dim

        # The CNP sees the forcing u as x, and [tas, pr] concatenated as the state.
        self.ctx_encoder = ContextEncoderForCnp(
            x_dim=ssm_model.u_dim,
            y_dim=2 * ssm_model.y_dim,
            r_dim=r_dim,
            hidden=encoder_hidden,
        )
        self.lat_encoder = CnpLatentEncoder(r_dim=r_dim, z_dim=z_cnp_dim)

        # FiLM projections must match the width of the SSM's emission input.
        emit_in_dim = ssm_model.z_dim
        if ssm_model.emission_uses_u:
            emit_in_dim += ssm_model.u_rnn_hidden + ssm_model.y_dim * ssm_model.reservoir_dim
        self.emit_in_dim = emit_in_dim
        self.z_cnp_scale = nn.Linear(z_cnp_dim, emit_in_dim)
        self.z_cnp_shift = nn.Linear(z_cnp_dim, emit_in_dim)

    # -- internal helpers ---------------------------------------------------

    def _encode_context(
        self,
        u_cnp: torch.Tensor,
        y_cnp: torch.Tensor,
        pr_cnp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer the ESM embedding from a context set.

        Args:
            u_cnp:  Context forcings.       Shape [B, N_ctx, u_dim].
            y_cnp:  Context tas.            Shape [B, N_ctx, y_dim].
            pr_cnp: Context pr.             Shape [B, N_ctx, y_dim].

        Returns:
            (z_cnp, mu_cnp, sigma_cnp) — each [B, z_cnp_dim].
        """
        state = torch.cat([y_cnp, pr_cnp], dim=-1)
        return self.lat_encoder(aggregate(self.ctx_encoder(u_cnp, state)))

    @staticmethod
    def _kl_cnp(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """KL( N(mu, sigma^2) || N(0, I) ), summed over z_cnp_dim and averaged over B.

        Args:
            mu:    Posterior mean.   Shape [B, Z].
            sigma: Posterior scale.  Shape [B, Z].

        Returns:
            Scalar KL divergence.
        """
        return (-0.5 * (1.0 + 2.0 * sigma.log() - mu.pow(2) - sigma.pow(2))).sum(-1).mean()

    def _film(self, z_cnp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project z_cnp to the FiLM scale and shift for the emission input.

        Args:
            z_cnp: ESM embedding.  Shape [B, z_cnp_dim].

        Returns:
            (gamma, beta) — each [B, emit_in_dim].
        """
        return self.z_cnp_scale(z_cnp), self.z_cnp_shift(z_cnp)

    def _emission_input(
        self,
        z: torch.Tensor,
        uh_t: torch.Tensor | None,
        s: torch.Tensor | None,
        film: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Assemble the SSM emission input and apply FiLM conditioning.

        Args:
            z:    Latent state.                  Shape [B, z_dim].
            uh_t: Control-GRU hidden state, or None when emission_uses_u is False.
            s:    Reservoir state, or None.      Shape [B, y_dim, reservoir_dim].
            film: (gamma, beta) from `_film`.

        Returns:
            Conditioned emission input.  Shape [B, emit_in_dim].
        """
        if self.ssm.emission_uses_u:
            e_in = torch.cat([z, uh_t, s.reshape(z.shape[0], -1)], dim=-1)
        else:
            e_in = z
        gamma, beta = film
        return gamma * e_in + beta

    # -- training objective -------------------------------------------------

    def forward_elbo(
        self,
        y: torch.Tensor,
        pr: torch.Tensor,
        u: torch.Tensor,
        u_cnp: torch.Tensor,
        y_cnp: torch.Tensor,
        pr_cnp: torch.Tensor,
        kl_free_bits: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the ELBO terms for a target sequence given an ESM context set.

        Mirrors `DeepSSMPatternConditioned.forward_elbo`, with the ESM embedding
        applied to the emission input at every step and an extra KL pulling the
        embedding posterior towards N(0, I).

        Args:
            y:            Target tas sequence.   Shape [B, T, Dy].
            pr:           Target pr sequence.    Shape [B, T, Dy].
            u:            Forcing sequence.      Shape [B, T, Du].
            u_cnp:        Context forcings.      Shape [B, N_ctx, Du].
            y_cnp:        Context tas.           Shape [B, N_ctx, Dy].
            pr_cnp:       Context pr.            Shape [B, N_ctx, Dy].
            kl_free_bits: Per-step floor on the SSM KL, guarding against
                          posterior collapse early in training.

        Returns:
            (nll, kl_ssm, nll_pr, kl_cnp) — scalar tensors, batch-averaged.
        """
        ssm     = self.ssm
        B, T, _ = y.shape

        z_cnp, mu_cnp, sigma_cnp = self._encode_context(u_cnp, y_cnp, pr_cnp)
        kl_cnp = self._kl_cnp(mu_cnp, sigma_cnp)
        film   = self._film(z_cnp)

        # Inference GRU over the full sequence -> per-step posterior params.
        h, _           = ssm.gru(torch.cat([y, u], dim=-1))
        mu_q, logvar_q = torch.chunk(ssm.q_head(h), 2, dim=-1)
        logvar_q       = torch.clamp(logvar_q, -12.0, 6.0)

        # Standard Normal prior for z_0.
        mu_p0     = torch.zeros(B, ssm.z_dim, device=y.device)
        logvar_p0 = torch.zeros(B, ssm.z_dim, device=y.device)

        nll     = 0.0
        nll_pr  = 0.0
        kl_ssm  = 0.0
        uh      = None
        s       = None

        if ssm.emission_uses_u:
            uh, _ = ssm.u_gru(u)
            # Start the reservoir at the equilibrium for the first control step.
            s = ssm.omega_lin(uh[:, 0]).reshape(B, ssm.y_dim, ssm.reservoir_dim).detach()

        z_prev = None
        for t in range(T):
            z_t  = ssm.sample(mu_q[:, t], logvar_q[:, t])
            e_in = self._emission_input(z_t, uh[:, t] if uh is not None else None, s, film)

            # tas emission — same density as ssm_tas_pr, diagonal or low-rank.
            res, cov_factor, cov_diag = ssm._parse_emit(ssm.emit(e_in), B)
            y_hat  = ssm.ctrl_lin(u[:, t]) + res if ssm.use_linear_model else res
            dist_y = ssm._emission_dist(y_hat, cov_factor, cov_diag)
            nll    = nll + (-dist_y.log_prob(y[:, t]))

            # pr emission — Sinh-Arcsinh flow.
            mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(
                ssm.emit_pr(e_in), 4, dim=-1)
            nll_pr = nll_pr + sinh_arcsinh_flow_nll_conditional(
                pr[:, t], mu_t, log_sigma_t, eps_skew_t, log_delta_t, eps=ssm.eps)

            # KL: learned prior for t > 0, standard Normal at t = 0.
            if t == 0:
                kl_t = utils.diag_gaussian_kl(mu_q[:, 0], logvar_q[:, 0], mu_p0, logvar_p0)
            else:
                mu_p, logvar_p = torch.chunk(
                    ssm.trans(torch.cat([z_prev, u[:, t]], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                kl_t     = utils.diag_gaussian_kl(mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)

            kl_ssm = kl_ssm + torch.clamp(kl_t, min=kl_free_bits)

            if ssm.emission_uses_u:
                s = ssm._reservoir_step(s, uh[:, t])
            z_prev = z_t

        return nll.mean(), kl_ssm.mean(), nll_pr.mean(), kl_cnp

    # -- forecasting --------------------------------------------------------

    @torch.no_grad()
    def _rollout(
        self,
        y_ctx: torch.Tensor,
        u_ctx: torch.Tensor,
        u_fut: torch.Tensor,
        steps: int,
        n_samples: int,
        pr_ctx: torch.Tensor | None,
        override_esm: bool,
        deterministic: bool,
    ) -> tuple[torch.Tensor, ...]:
        """Shared ancestral-sampling rollout behind both public forecast methods.

        Args:
            y_ctx:         tas context.               Shape [B, Tc, Dy].
            u_ctx:         Forcing context.           Shape [B, Tc, Du].
            u_fut:         Known future forcings.     Shape [B, steps, Du].
            steps:         Forecast horizon.
            n_samples:     Monte Carlo trajectories.
            pr_ctx:        pr context for the CNP encoding, or None.
            override_esm:  Draw a fresh z_cnp ~ N(0, I) per trajectory.
            deterministic: Use emission means instead of sampling the emissions.

        Returns:
            (tas_mean, tas_q10, tas_q90, pr_mean, pr_q10, pr_q90) — each
            [B, steps, Dy].
        """
        ssm = self.ssm
        B   = y_ctx.shape[0]

        film: tuple[torch.Tensor, torch.Tensor] | None = None
        if not override_esm:
            if pr_ctx is not None:
                z_cnp, _, _ = self._encode_context(u_ctx, y_ctx, pr_ctx)
            else:
                # No pr context to encode: fall back to z_cnp = 0, which leaves
                # FiLM at the projections' biases rather than at the identity.
                z_cnp = torch.zeros(B, self.z_cnp_dim, device=y_ctx.device)
            film = self._film(z_cnp)

        # Encode the context; only the final hidden state seeds the rollout.
        h, _             = ssm.gru(torch.cat([y_ctx, u_ctx], dim=-1))
        mu_qT, logvar_qT = torch.chunk(ssm.q_head(h[:, -1:]).squeeze(1), 2, dim=-1)
        logvar_qT        = torch.clamp(logvar_qT, -12.0, 6.0)

        h_u   = None
        s_ctx = None
        if ssm.emission_uses_u:
            uh_ctx, h_u = ssm.u_gru(u_ctx)
            # Warm the reservoir up by replaying the whole context.
            s_ctx = torch.zeros(B, ssm.y_dim, ssm.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = ssm._reservoir_step(s_ctx, uh_ctx[:, k])

        samples, samples_pr = [], []
        for _ in range(n_samples):
            if override_esm:
                # Marginalise over ESM identity jointly with the SSM noise.
                film = self._film(torch.randn(B, self.z_cnp_dim, device=y_ctx.device))

            z     = ssm.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone()   if ssm.emission_uses_u else None
            s     = s_ctx.clone() if ssm.emission_uses_u else None

            preds, preds_pr = [], []
            for k in range(steps):
                u_t            = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(
                    ssm.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z        = ssm.sample(mu_p, logvar_p)

                uh_t = None
                if ssm.emission_uses_u:
                    uh_t, h_u_s = ssm.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t        = uh_t.squeeze(1)

                e_in = self._emission_input(z, uh_t, s, film)

                if ssm.emission_uses_u:
                    s = ssm._reservoir_step(s, uh_t)

                emit_out = ssm.emit(e_in)
                mu_pr, log_sigma_pr, eps_skew_t, log_delta_t = torch.chunk(
                    ssm.emit_pr(e_in), 4, dim=-1)

                if deterministic:
                    # Only the emission means are needed; skip the cov params.
                    res    = emit_out[:, :ssm.y_dim]
                    y_hat  = ssm.ctrl_lin(u_t) + res if ssm.use_linear_model else res
                    x_samp = mu_pr
                else:
                    res, cov_factor, cov_diag = ssm._parse_emit(emit_out, B)
                    y_hat = ssm.ctrl_lin(u_t) + res if ssm.use_linear_model else res
                    y_hat = ssm._emission_dist(y_hat, cov_factor, cov_diag).sample()
                    sigma_pr = torch.exp(log_sigma_pr) + ssm.eps
                    x_samp   = mu_pr + sigma_pr * torch.randn_like(mu_pr)

                preds.append(y_hat)
                preds_pr.append(
                    sinh_arcsinh_forward(x_samp, eps_skew_t, log_delta_t, eps=ssm.eps))

            samples.append(torch.stack(preds,       dim=1))
            samples_pr.append(torch.stack(preds_pr, dim=1))

        samp    = torch.stack(samples,    dim=0)   # [S, B, steps, Dy]
        samp_pr = torch.stack(samples_pr, dim=0)
        return (
            samp.mean(0),    samp.quantile(0.10, 0),    samp.quantile(0.90, 0),
            samp_pr.mean(0), samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0),
        )

    def forecast(
        self,
        y_ctx: torch.Tensor,
        u_ctx: torch.Tensor,
        u_fut: torch.Tensor,
        steps: int,
        n_samples: int = 50,
        pr_ctx: torch.Tensor | None = None,
        override_esm: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Probabilistic forecast via ancestral sampling.

        The time-series context doubles as the CNP context set, so the ESM
        embedding is inferred from the same window that warms up the SSM.

        Args:
            y_ctx:        tas context.            Shape [B, Tc, Dy].
            u_ctx:        Forcing context.        Shape [B, Tc, Du].
            u_fut:        Known future forcings.  Shape [B, steps, Du].
            steps:        Forecast horizon.
            n_samples:    Monte Carlo trajectories for the mean and bands.
            pr_ctx:       pr context. Without it the embedding falls back to
                          z_cnp = 0, i.e. no ESM-specific adaptation.
            override_esm: Draw a fresh z_cnp ~ N(0, I) per trajectory, so the
                          bands marginalise over ESM identity as well as over
                          the SSM's own stochasticity.

        Returns:
            (tas_mean, tas_q10, tas_q90, pr_mean, pr_q10, pr_q90) — each
            [B, steps, Dy].
        """
        return self._rollout(y_ctx, u_ctx, u_fut, steps, n_samples,
                             pr_ctx, override_esm, deterministic=False)

    def forecast_deterministic(
        self,
        y_ctx: torch.Tensor,
        u_ctx: torch.Tensor,
        u_fut: torch.Tensor,
        steps: int,
        n_samples: int = 50,
        pr_ctx: torch.Tensor | None = None,
        override_esm: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Mean-path forecast, averaged over latent samples.

        Identical to `forecast` but takes the emission means instead of drawing
        from the emission distributions, which is faster and lower-variance —
        used for the rollout loss terms during training.

        Args:
            y_ctx:        tas context.            Shape [B, Tc, Dy].
            u_ctx:        Forcing context.        Shape [B, Tc, Du].
            u_fut:        Known future forcings.  Shape [B, steps, Du].
            steps:        Forecast horizon.
            n_samples:    Latent samples to average over.
            pr_ctx:       pr context for the CNP encoding, or None.
            override_esm: Draw a fresh z_cnp ~ N(0, I) per trajectory.

        Returns:
            (tas_mean, tas_q10, tas_q90, pr_mean, pr_q10, pr_q90) — each
            [B, steps, Dy].
        """
        return self._rollout(y_ctx, u_ctx, u_fut, steps, n_samples,
                             pr_ctx, override_esm, deterministic=True)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_task_loss(
    model: DeepCnpSsmforESM,
    batch: Batch,
    device: torch.device | str,
    horizon: int,
    kl_w: float = 5.0,
    kl_cnp_w: float = 1.0,
    alpha_w: float = 10000.0,
    omega_w: float = 500.0,
    gamma_w: float = 80000.0,
    kl_free_bits: float = 0.2,
    n_rollout_samples: int = 30,
    stochastic_rollout: bool = False,
    huber: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the composite loss for one batch, matching `run_train`'s objective.

    Gradients flow only through the ELBO terms: both forecast methods are
    decorated `@torch.no_grad`, so the rollout terms contribute to the reported
    loss value but not to the gradient — the same behaviour as `run_train`.

    The warm-up part of the batch (y_ctx, pr_ctx, u_ctx) doubles as the CNP
    context set.

    Args:
            model:              The CNP-wrapped SSM.
            batch:              (y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut) from
                                UnifiedWindowDataset.
            device:             Device to move the batch to.
            horizon:            Forecast horizon; must match the dataset's H.
            kl_w:               Weight on the SSM KL.
            kl_cnp_w:           Weight on the ESM-embedding KL.
            alpha_w:            Weight on the tas rollout error.
            omega_w:            Weight on the pr rollout error.
            gamma_w:            Weight on the yearly-mean tas error; 0 drops the term.
            kl_free_bits:       Per-step floor on the SSM KL.
            n_rollout_samples:  Monte Carlo samples for the rollout.
            stochastic_rollout: Use `forecast` rather than `forecast_deterministic`.
            huber:              Use Huber loss for the rollout terms instead of MSE.

    Returns:
        (loss, metrics) where metrics holds the detached scalar components for
        logging: nll, nll_pr, kl_ssm, kl_cnp, mse, mse_pr, mse_yearly.
    """
    y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut = (t.float().to(device) for t in batch)

    y_full  = torch.cat([y_ctx,  y_fut],  dim=1)
    pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
    u_full  = torch.cat([u_ctx,  u_fut],  dim=1)
    B, _, Dy = y_full.shape

    nll, kl_ssm, nll_pr, kl_cnp = model.forward_elbo(
        y_full, pr_full, u_full,
        u_cnp=u_ctx, y_cnp=y_ctx, pr_cnp=pr_ctx,
        kl_free_bits=kl_free_bits,
    )

    run = model.forecast if stochastic_rollout else model.forecast_deterministic
    mean, _, _, mean_pr, _, _ = run(
        y_ctx, u_ctx, u_fut, steps=horizon,
        n_samples=n_rollout_samples, pr_ctx=pr_ctx,
    )

    if huber:
        roll_mse    = F.huber_loss(mean,    y_fut,  delta=1.0)
        roll_mse_pr = F.huber_loss(mean_pr, pr_fut, delta=1.0)
    else:
        roll_mse    = ((mean    - y_fut)  ** 2).mean()
        roll_mse_pr = ((mean_pr - pr_fut) ** 2).mean()

    # Yearly-average error penalises multi-year trend drift over whole years.
    n_complete_years = horizon // 12
    if n_complete_years > 0:
        H_yr       = n_complete_years * 12
        mean_yr    = mean[:,  :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
        y_fut_yr   = y_fut[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
        mse_yearly = ((mean_yr - y_fut_yr) ** 2).mean()
    else:
        mse_yearly = torch.zeros((), device=mean.device)

    loss = (
        nll + nll_pr
        + kl_w     * kl_ssm
        + kl_cnp_w * kl_cnp
        + alpha_w  * roll_mse
        + omega_w  * roll_mse_pr
        + gamma_w  * mse_yearly
    )

    metrics = {
        "nll":        nll.item(),
        "nll_pr":     nll_pr.item(),
        "kl_ssm":     kl_ssm.item(),
        "kl_cnp":     kl_cnp.item(),
        "mse":        roll_mse.item(),
        "mse_pr":     roll_mse_pr.item(),
        "mse_yearly": mse_yearly.item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnp(
    model: DeepCnpSsmforESM,
    task_dict: dict[str, dict[str, Any]],
    num_epochs: int,
    horizon: int,
    lr: float = 2e-3,
    batch_size: int = 64,
    kl_w: float = 5.0,
    alpha_w: float = 10000.0,
    omega_w: float = 500.0,
    gamma_w: float = 80000.0,
    kl_cnp_w: float = 1.0,
    run_dir: str | None = None,
    device: torch.device | str = "cuda",
    patience: int = 15,
    weights_file: str | None = None,
    unfreeze_frac: float = 0.3,
    lr_ssm_unfrozen: float = 2e-3,
    n_rollout_samples: int = 30,
) -> DeepCnpSsmforESM:
    """Train a DeepCnpSsmforESM across ESM tasks, with early stopping.

    Each ESM's support set is a training set and its query set a validation set.
    Every ESM takes the same number of steps per epoch — matching the richest
    ESM, with sparse ESMs cycling their loader — and contributes an
    inverse-frequency weighted loss, so dataset size does not decide influence.

    When `weights_file` is given, training starts from a pretrained SSM with
    everything but the emission heads frozen, letting the CNP modules and the
    emissions adapt first; all SSM weights are unfrozen at `unfreeze_frac` of
    the run, at the lower `lr_ssm_unfrozen`.

    `alpha_w` and `omega_w` are ramped linearly over the first 30 % of steps;
    `gamma_w` has no warm-up, as in `run_train`.

    Args:
        model:             The CNP-wrapped SSM to train.
        task_dict:         Output of `build_task_dict` — already normalised.
        num_epochs:        Maximum number of epochs.
        horizon:           Forecast horizon; must match the datasets.
        lr:                Learning rate for the trainable parameters.
        batch_size:        Batch size per ESM; capped at the support-set size.
        kl_w:              Weight on the SSM KL.
        alpha_w:           Final weight on the tas rollout error.
        omega_w:           Final weight on the pr rollout error.
        gamma_w:           Weight on the yearly-mean tas error.
        kl_cnp_w:          Weight on the ESM-embedding KL.
        run_dir:           If given, checkpoints every 10 epochs plus the final
                           model are written there.
        device:            Device to train on.
        patience:          Epochs without validation improvement before stopping.
        weights_file:      Pretrained DeepSSMPatternConditioned state dict.
        unfreeze_frac:     Fraction of epochs after which all SSM weights
                           unfreeze; only used when weights_file is given.
        lr_ssm_unfrozen:   Learning rate for the newly unfrozen SSM parameters,
                           typically below `lr` to avoid disrupting them.
        n_rollout_samples: Monte Carlo samples for the rollout loss terms.

    Returns:
        The model, restored to its best-validation weights.
    """
    model = model.to(device)

    ssm_frozen     = False
    unfreeze_epoch = int(unfreeze_frac * num_epochs) + 1

    if weights_file is not None:
        model.ssm.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"Loaded SSM weights from {weights_file}")
        # Emission heads stay trainable so they can absorb the FiLM conditioning.
        for name, p in model.ssm.named_parameters():
            p.requires_grad = name.startswith("emit")
        print(f"SSM frozen except emit and emit_pr; unfreezing all at epoch {unfreeze_epoch}.")
        ssm_frozen = True

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    train_loaders = {
        name: DataLoader(
            task["support"],
            batch_size=min(batch_size, len(task["support"])),
            shuffle=True,
            drop_last=True,
        )
        for name, task in task_dict.items()
    }
    val_loaders = {
        name: DataLoader(task["query"], batch_size=batch_size, shuffle=False)
        for name, task in task_dict.items()
    }

    # Inverse-frequency task weights, normalised to mean 1 so the overall loss
    # scale is unchanged while small ESMs still pull their weight.
    n_support    = {name: len(task["support"]) for name, task in task_dict.items()}
    max_n        = max(n_support.values())
    raw_weights  = {name: max_n / n for name, n in n_support.items()}
    w_mean       = sum(raw_weights.values()) / len(raw_weights)
    task_weights = {name: w / w_mean for name, w in raw_weights.items()}
    print("Inverse-frequency task weights:",
          {name: f"{w:.2f}" for name, w in task_weights.items()})

    steps_per_esm = max(len(dl) for dl in train_loaders.values())
    total_steps   = num_epochs * steps_per_esm * len(task_dict)
    warmup_steps  = max(1, int(0.3 * total_steps))

    best_val      = float("inf")
    best_state    = None
    patience_left = patience
    global_step   = 0
    # Annealed weights persist out of the training loop into validation.
    alpha = omega = 0.0

    for epoch in range(1, num_epochs + 1):
        if ssm_frozen and epoch == unfreeze_epoch:
            newly_unfrozen = [p for p in model.ssm.parameters() if not p.requires_grad]
            for p in model.ssm.parameters():
                p.requires_grad = True
            opt.add_param_group({"params": newly_unfrozen, "lr": lr_ssm_unfrozen})
            ssm_frozen = False
            print(f"Epoch {epoch}: unfroze all SSM weights (lr={lr_ssm_unfrozen}).")

        model.train()
        tr_losses = []

        # Fresh iterators each epoch — no state carried across epoch boundaries.
        train_iters = {name: iter(dl) for name, dl in train_loaders.items()}

        for _ in range(steps_per_esm):
            for esm_name in task_dict:
                try:
                    batch = next(train_iters[esm_name])
                except StopIteration:
                    train_iters[esm_name] = iter(train_loaders[esm_name])
                    batch = next(train_iters[esm_name])

                global_step += 1
                frac  = min(1.0, global_step / warmup_steps)
                alpha = alpha_w * frac
                omega = omega_w * frac

                loss, metrics = compute_task_loss(
                    model, batch, device, horizon,
                    kl_w=kl_w, kl_cnp_w=kl_cnp_w,
                    alpha_w=alpha, omega_w=omega, gamma_w=gamma_w,
                    n_rollout_samples=n_rollout_samples,
                )
                loss = task_weights[esm_name] * loss

                if global_step % 100 == 0:
                    print(
                        f"[{esm_name}] step {global_step:05d} | "
                        + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                    )

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_losses.append(loss.item())

        # ---- Validation on the query sets ----
        model.eval()
        va_losses, va_mses = [], []

        with torch.no_grad():
            for esm_name, loader in val_loaders.items():
                weight = task_dict[esm_name].get("weight", 1.0)
                for batch in loader:
                    # Validation mirrors the prototype: a stochastic rollout
                    # scored with plain MSE and no yearly-trend term.
                    val_loss, val_metrics = compute_task_loss(
                        model, batch, device, horizon,
                        kl_w=kl_w, kl_cnp_w=kl_cnp_w,
                        alpha_w=alpha, omega_w=omega, gamma_w=0.0,
                        n_rollout_samples=n_rollout_samples,
                        stochastic_rollout=True, huber=False,
                    )
                    va_losses.append(weight * val_loss.item())
                    va_mses.append(val_metrics["mse"])

        tr     = float(np.mean(tr_losses))
        va     = float(np.mean(va_losses))
        mse_va = float(np.mean(va_mses))
        print(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | val_mse {mse_va:.4f}")

        if run_dir is not None and epoch % 10 == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"cnp_epoch{epoch:04d}.pt"))

        if va < best_val - 1e-4:
            best_val      = va
            best_state    = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}: no improvement in {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(run_dir, "cnp_model_out"))

    return model
