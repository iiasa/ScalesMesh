from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import scales.ssm.ssm_model_utils as utils

# Maximum EMA decay rate for the emission reservoir, after the sigmoid on
# log_alpha. Fixed rather than configurable: it rescales the learned log_alpha,
# so a value that differs from the one used at training time silently changes
# the meaning of a checkpoint's weights.
ALPHA_MAX = 0.002


# ---------------------------------------------------------------------------
# Emission distributions
# ---------------------------------------------------------------------------

def sinh_arcsinh_flow_nll_conditional(
    y: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    eps_skew: torch.Tensor,
    log_delta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Negative log-likelihood under a conditional Sinh-Arcsinh normalising flow.

    The flow maps a base Normal(mu, sigma) through the inverse Sinh-Arcsinh
    transform so the resulting distribution can capture skewness and heavy/light
    tails — useful for precipitation, which is typically right-skewed.

    Inverse transform (observation -> base space):
        x = sinh(delta * asinh(y) - eps_skew)

    Change-of-variables log|dx/dy|:
        log cosh(t) + log delta - 0.5 * log(1 + y^2),  where t = delta*asinh(y) - eps_skew

    Args:
        y:         Observed values.                Shape [B, Dy].
        mu:        Base Normal mean.               Shape [B, Dy].
        log_sigma: Base Normal log-scale.          Shape [B, Dy].
        eps_skew:  Skewness parameter ε.           Shape [B, Dy].
        log_delta: Unconstrained tail parameter;   Shape [B, Dy].
                   delta = softplus(log_delta) + eps > 0.
        eps:       Numerical stability floor.

    Returns:
        Per-sample NLL summed over Dy.  Shape [B].
    """
    sigma = torch.exp(torch.clamp(log_sigma, -8.0, 6.0)) + eps
    delta = F.softplus(log_delta) + eps

    # Inverse transform: x = sinh(delta * asinh(y) - eps_skew)
    a = torch.asinh(y)
    t = delta * a - eps_skew
    x = torch.sinh(t)

    # Log absolute Jacobian of the inverse transform: dx/dy = cosh(t)*delta / sqrt(1+y^2)
    log_abs_det = torch.log(torch.cosh(t) + eps) + torch.log(delta) - 0.5 * torch.log1p(y * y)

    # Log-prob of the base Normal evaluated at the pre-image x
    r = (x - mu) / sigma
    logp_base = -0.5 * (r * r) - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)

    # Change-of-variables: log p(y) = log p_base(x) + log|dx/dy|
    logp = logp_base + log_abs_det
    return (-logp).sum(dim=-1)


def sinh_arcsinh_forward(
    x: torch.Tensor,
    eps_skew: torch.Tensor,
    log_delta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Forward Sinh-Arcsinh transform: maps base-space samples to observation space.

    y = sinh( (asinh(x) + eps_skew) / delta )

    Used at forecast time to push Normal samples through the flow and produce
    samples in the original precipitation scale.

    Args:
        x:         Samples from base Normal.   Shape [..., Dy].
        eps_skew:  Skewness parameter.         Shape [..., Dy].
        log_delta: Unconstrained tail param.   Shape [..., Dy].
        eps:       Numerical stability floor.

    Returns:
        Transformed samples in observation space, same shape as x.
    """
    delta = F.softplus(log_delta) + eps
    return torch.sinh((torch.asinh(x) + eps_skew) / delta)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnifiedWindowDataset(Dataset):
    """Sliding-window dataset supporting a single series or a batch of series.

    Accepts either:
      - A single series: y [T, Dy], pr [T, Dy], u [T, Du]
      - Many series:     y [N, T, Dy], pr [N, T, Dy], u [N, T, Du]

    Each item is a (context, horizon) window drawn from one series. The window
    covers both the target variable y and the precipitation proxy pr, together
    with exogenous controls u.

    Item contents:
        y_ctx:  [Tc, Dy]  tas context
        pr_ctx: [Tc, Dy]  precipitation context
        u_ctx:  [Tc, Du]  control context (GMT)
        u_fut:  [H,  Du]  control horizon (assumed known at forecast time)
        pr_fut: [H,  Dy]  precipitation horizon
        y_fut:  [H,  Dy]  tas horizon (label)
    """

    def __init__(
        self,
        y: np.ndarray,
        pr: np.ndarray,
        u: np.ndarray,
        context_len: int = 40,
        horizon: int = 12,
        stride: int = 1,
        start_mode: str = "all",
    ) -> None:
        """
        Args:
            y:           Target observations.   Shape [T, Dy] or [N, T, Dy].
            pr:          Precipitation proxy.   Same leading shape as y.
            u:           Exogenous controls.    Shape [T, Du] or [N, T, Du].
            context_len: Context window length Tc.
            horizon:     Forecast horizon H.
            stride:      Step between consecutive window starts; stride > 1
                         reduces the number of overlapping windows.
            start_mode:  "all"  — sliding window over every valid start index,
                         "zero" — single window per series starting at t=0.
        """
        y  = np.asarray(y,  dtype=np.float32)
        pr = np.asarray(pr, dtype=np.float32)
        u  = np.asarray(u,  dtype=np.float32)

        # Normalise 2-D inputs to [N, T, D] so the rest of the code is uniform.
        if y.ndim  == 2: y  = y[None, ...]
        if pr.ndim == 2: pr = pr[None, ...]
        if u.ndim  == 2: u  = u[None, ...]

        assert y.shape[0]  == u.shape[0]  and y.shape[1]  == u.shape[1]
        assert pr.shape[0] == u.shape[0]  and pr.shape[1] == u.shape[1]
        assert pr.shape[2] == y.shape[2]

        self.y  = y
        self.u  = u
        self.pr = pr
        self.N, self.T, self.Dy = y.shape
        self.Du     = u.shape[2]
        self.Tc     = int(context_len)
        self.H      = int(horizon)
        self.stride = int(stride)

        if self.Tc + self.H > self.T:
            raise ValueError("context_len + horizon must be <= T")

        # Build a flat index of (series_idx, window_start) pairs.
        self.index: list[tuple[int, int]] = []
        max_start = self.T - (self.Tc + self.H)
        for s in range(self.N):
            if start_mode == "zero":
                self.index.append((s, 0))
            elif start_mode == "all":
                for start in range(0, max_start + 1, self.stride):
                    self.index.append((s, start))
            else:
                raise ValueError("start_mode must be 'all' or 'zero'")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Return (y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut) for window idx."""
        s, start = self.index[idx]
        Tc, H = self.Tc, self.H

        y  = self.y[s]
        u  = self.u[s]
        pr = self.pr[s]

        y_ctx  = y [start      : start + Tc]
        u_ctx  = u [start      : start + Tc]
        pr_ctx = pr[start      : start + Tc]
        u_fut  = u [start + Tc : start + Tc + H]
        y_fut  = y [start + Tc : start + Tc + H]
        pr_fut = pr[start + Tc : start + Tc + H]

        return y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeepSSMPatternConditioned(nn.Module):
    """Deep State-Space Model with pattern-conditioned emission.

    Inference network (approximate posterior):
        q(z_t | y_{1:t}, u_{1:t})  —  GRU encoder over [y, u] -> (mu_q, logvar_q)

    Prior (learned transition):
        p(z_t | z_{t-1}, u_t)  —  MLP([z_{t-1}, u_t]) -> (mu_p, logvar_p)

    Emission for target y:
        p(y_t | z_t)  —  MLP(z_t) -> Gaussian over the D output channels.
                         cov_rank > 0: low-rank Multivariate Normal
                             parameterised by (mean, cov_factor [D, r], cov_diag [D]),
                             capturing cross-channel correlation.
                         cov_rank = 0: diagonal Normal parameterised by
                             (mean, cov_diag [D]); channels are conditionally
                             independent given z.

    Emission for precipitation pr:
        p(pr_t | z_t)  —  MLP(z_t) -> Sinh-Arcsinh flow params
                          (mu, log_sigma, eps_skew, log_delta)

    Optional pattern-conditioning (emission_uses_u=True):
        A secondary GRU processes u and feeds an EMA reservoir state into the
        emission network, letting the model condition on the control trajectory
        without conflating it with the latent dynamics in z.

    Optional linear control baseline (use_linear_model=True):
        A ridge-initialised linear map ctrl_lin: u_t -> y_dim is added to the
        residual emission, separating the linear control response from the
        non-linear SSM residual. run_train freezes ctrl_lin at its ridge
        solution for the whole run, so the SSM only ever learns the residual on
        top of a fixed linear baseline.
    """

    def __init__(
        self,
        y_dim: int,
        u_dim: int,
        z_dim: int = 16,
        rnn_hidden: int = 62,
        u_rnn_hidden: int = 64,
        mlp_hidden: int = 128,
        emission_uses_u: bool = False,
        use_linear_model: bool = True,
        reservoir_dim: int = 2,
        init_alpha: float = 0.01,
        init_omega: float = 1.0,
        cov_rank: int = 5,
    ) -> None:
        """
        Args:
            y_dim:            Dimensionality of the target y.
            u_dim:            Dimensionality of the exogenous control u.
            z_dim:            Latent state dimensionality.
            rnn_hidden:       Hidden size of the inference GRU.
            u_rnn_hidden:     Hidden size of the control GRU (emission_uses_u only).
            mlp_hidden:       Hidden size for all MLP modules.
            emission_uses_u:  If True, the emission network receives control-derived
                              features from the EMA reservoir in addition to z.
            use_linear_model: If True, adds a linear ctrl_lin: u -> y as a pattern
                              baseline initialised via ridge regression.
            reservoir_dim:    Number of EMA reservoir states per output channel.
            init_alpha:       Initial EMA decay rate (before sigmoid scaling).
            init_omega:       Unused placeholder (reserved for future use).
            cov_rank:         Rank r of the low-rank covariance factor for y.
                              0 selects a purely diagonal emission, which drops
                              the factor block from the emit head entirely.
        """
        super().__init__()
        self.y_dim            = y_dim
        self.u_dim            = u_dim
        self.z_dim            = z_dim
        self.emission_uses_u  = emission_uses_u
        self.use_linear_model = use_linear_model
        self.u_rnn_hidden     = u_rnn_hidden
        self.reservoir_dim    = reservoir_dim
        self.alpha_max        = ALPHA_MAX
        self.cov_rank         = cov_rank

        # Inference network: encodes [y_t, u_t] sequence -> per-step posterior params.
        self.gru    = nn.GRU(input_size=y_dim + u_dim, hidden_size=rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        # Prior transition network: p(z_t | z_{t-1}, u_t).
        self.trans = utils.MLP(z_dim + u_dim, 2 * z_dim, hidden=mlp_hidden)

        if emission_uses_u:
            # Secondary GRU summarises the control trajectory for the emission.
            self.u_gru = nn.GRU(input_size=u_dim, hidden_size=u_rnn_hidden, batch_first=True)
            # log_alpha parameterises per-channel EMA rates in (0, alpha_max).
            # Initialised so sigmoid(log_alpha) ≈ init_alpha.
            log_alpha_init = math.log(init_alpha / (1.0 - init_alpha))
            self.log_alpha = nn.Parameter(torch.full((y_dim, reservoir_dim), log_alpha_init))
            # Projects u-GRU hidden state to per-channel reservoir targets.
            self.omega_lin = nn.Linear(u_rnn_hidden, y_dim * reservoir_dim, bias=False)

        # Input size to emission MLPs depends on whether control features are used.
        emit_in = z_dim + (u_rnn_hidden + y_dim * reservoir_dim if emission_uses_u else 0)

        # y emission: outputs mean (D) + log_cov_diag (D) + cov_factor (D*r).
        # With cov_rank=0 the factor block has width 0, leaving a 2*D diagonal head.
        self.emit    = utils.MLP(emit_in, y_dim * (2 + cov_rank), hidden=mlp_hidden)
        # pr emission: outputs 4 Sinh-Arcsinh parameters per output channel.
        self.emit_pr = utils.MLP(emit_in, 4 * y_dim, hidden=230)

        if self.use_linear_model:
            # Linear pattern baseline; ridge-initialised and frozen via
            # load_into_ctrl_lin.
            self.ctrl_lin = nn.Linear(u_dim, y_dim, bias=True)

        self.eps = 1e-6

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterised sample: z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)."""
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def _reservoir_step(self, s: torch.Tensor, uh_t: torch.Tensor) -> torch.Tensor:
        """Advance the per-channel EMA reservoir by one time step.

        Implements:  s_{t+1} = s_t + alpha * (target_t - s_t)
        where target_t = omega_lin(uh_t) reshaped to [B, y_dim, reservoir_dim].

        The bounded, learnable alpha ensures the reservoir tracks the control
        signal smoothly without dominating gradients from z.

        Args:
            s:    Current reservoir state.        Shape [B, y_dim, reservoir_dim].
            uh_t: u-GRU hidden state at step t.   Shape [B, u_rnn_hidden].

        Returns:
            Updated reservoir state.  Shape [B, y_dim, reservoir_dim].
        """
        # alpha in (0, alpha_max) — bounded so the EMA cannot collapse to identity.
        alpha  = torch.sigmoid(self.log_alpha) * self.alpha_max
        target = self.omega_lin(uh_t).reshape(uh_t.shape[0], self.y_dim, self.reservoir_dim)
        return s + alpha[None] * (target - s)

    def _parse_emit(
        self, emit_out: torch.Tensor, B: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Split the raw y-emission MLP output into (mean, cov_factor, cov_diag).

        The output layout is fixed as [mean (D) | log_cov_diag (D) | factor (D*r)]
        so a checkpoint's emit head keeps its meaning across ranks.

        Args:
            emit_out: Raw MLP output.  Shape [B, D*(2+r)].
            B:        Batch size.

        Returns:
            res:        Residual mean.            Shape [B, D].
            cov_factor: Low-rank factor L.        Shape [B, D, r],
                        or None when cov_rank == 0 (diagonal emission).
            cov_diag:   Positive diagonal term.   Shape [B, D].
        """
        D, r       = self.y_dim, self.cov_rank
        res        = emit_out[:, :D]
        log_diag   = emit_out[:, D:2*D]
        cov_diag   = F.softplus(log_diag) + self.eps   # enforce positivity
        # Rank 0 leaves no factor block to read; the emission is diagonal.
        cov_factor = emit_out[:, 2*D:].reshape(B, D, r) if r > 0 else None
        return res, cov_factor, cov_diag

    def _emission_dist(
        self,
        y_hat: torch.Tensor,
        cov_factor: torch.Tensor | None,
        cov_diag: torch.Tensor,
    ) -> torch.distributions.Distribution:
        """Build the Gaussian emission p(y_t | z_t) for the configured cov_rank.

        cov_rank > 0 gives a low-rank Multivariate Normal with covariance
        L L^T + diag(cov_diag), which models correlation between output
        channels. cov_rank = 0 gives an independent Normal per channel; the
        `Independent` wrapper reinterprets the channel axis as part of the
        event, so `log_prob` sums over D and returns [B] — matching the
        low-rank branch, as does `sample()` returning [B, D].

        Args:
            y_hat:      Emission mean.               Shape [B, D].
            cov_factor: Low-rank factor, or None.    Shape [B, D, r].
            cov_diag:   Positive diagonal variance.  Shape [B, D].

        Returns:
            A distribution over [B, D] whose log_prob has shape [B].
        """
        if self.cov_rank > 0:
            return torch.distributions.LowRankMultivariateNormal(
                loc=y_hat, cov_factor=cov_factor, cov_diag=cov_diag)

        # cov_diag holds variances, so the Normal scale is its square root.
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=y_hat, scale=torch.sqrt(cov_diag)),
            reinterpreted_batch_ndims=1,
        )

    def forward_elbo(
        self,
        y: torch.Tensor,
        pr: torch.Tensor,
        u: torch.Tensor,
        kl_free_bits: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the ELBO loss terms for a full sequence.

        Runs the inference network over the full sequence, samples z_t from q
        at each step, and accumulates:
          - NLL of y under the low-rank Gaussian emission.
          - NLL of pr under the Sinh-Arcsinh flow emission.
          - KL divergence KL(q || p) with a free-bits floor to prevent
            posterior collapse during early training.

        Args:
            y:             Target sequence.        Shape [B, T, Dy].
            pr:            Precipitation sequence. Shape [B, T, Dy].
            u:             Control sequence.       Shape [B, T, Du].
            kl_free_bits:  Minimum KL per sample; gradients below this are
                           zeroed to avoid over-regularisation of the posterior.

        Returns:
            (nll, kl, nll_pr) — scalar tensors, each averaged over the batch.
        """
        B, T, _ = y.shape

        # Run inference GRU to obtain per-step posterior parameters.
        rnn_in         = torch.cat([y, u], dim=-1)
        h, _           = self.gru(rnn_in)
        q_params       = self.q_head(h)
        mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)
        # Clamp logvar to a stable range; bounds are not tuned precisely.
        logvar_q = torch.clamp(logvar_q, -12.0, 6.0)

        # Standard Normal prior for the initial latent z_0.
        mu_p0     = torch.zeros(B, self.z_dim, device=y.device)
        logvar_p0 = torch.zeros(B, self.z_dim, device=y.device)

        nll    = 0.0
        nll_pr = 0.0
        kl     = 0.0

        if self.emission_uses_u:
            uh, _ = self.u_gru(u)
            # Initialise reservoir at the analytical equilibrium for the first u step.
            s = self.omega_lin(uh[:, 0]).reshape(B, self.y_dim, self.reservoir_dim).detach()

        z_prev = None
        for t in range(T):
            z_t = self.sample(mu_q[:, t], logvar_q[:, t])

            if self.use_linear_model:
                ctrl = self.ctrl_lin(u[:, t])

            e_in = torch.cat([z_t, uh[:, t], s.reshape(B, -1)], dim=-1) \
                   if self.emission_uses_u else z_t

            # y emission: low-rank Multivariate Normal.
            emit_out            = self.emit(e_in)
            res, cov_factor, cov_diag = self._parse_emit(emit_out, B)
            y_hat               = ctrl + res if self.use_linear_model else res
            dist_y              = self._emission_dist(y_hat, cov_factor, cov_diag)
            nll = nll + (-dist_y.log_prob(y[:, t]))

            # pr emission: Sinh-Arcsinh flow.
            out                                             = self.emit_pr(e_in)
            mu_t, log_sigma_t, eps_skew_t, log_delta_t     = torch.chunk(out, 4, dim=-1)
            nll_pr = nll_pr + sinh_arcsinh_flow_nll_conditional(
                pr[:, t], mu_t, log_sigma_t, eps_skew_t, log_delta_t, eps=self.eps)

            # KL: use learned prior for t > 0, standard Normal for t = 0.
            if t == 0:
                kl_t = utils.diag_gaussian_kl(mu_q[:, 0], logvar_q[:, 0], mu_p0, logvar_p0)
            else:
                trans_in        = torch.cat([z_prev, u[:, t]], dim=-1)
                mu_p, logvar_p  = torch.chunk(self.trans(trans_in), 2, dim=-1)
                logvar_p        = torch.clamp(logvar_p, -12.0, 6.0)
                kl_t            = utils.diag_gaussian_kl(mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)

            # Free-bits: clamp KL from below to prevent posterior collapse.
            kl = kl + torch.clamp(kl_t, min=kl_free_bits)

            if self.emission_uses_u:
                s = self._reservoir_step(s, uh[:, t])
            z_prev = z_t

        return nll.mean(), kl.mean(), nll_pr.mean()

    @torch.no_grad()
    def forecast(
        self,
        y_ctx: torch.Tensor,
        u_ctx: torch.Tensor,
        u_fut: torch.Tensor,
        steps: int,
        n_samples: int = 50,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Probabilistic forecast via ancestral sampling.

        Encodes the context to obtain q(z_T | context), then unrolls the
        prior p(z_t | z_{t-1}, u_t) for `steps` steps drawing `n_samples`
        independent trajectories. Returns the mean and 10/90th-percentile
        bands for both y and pr.

        Args:
            y_ctx:     Observed target context.    Shape [B, Tc, Dy].
            u_ctx:     Control context.            Shape [B, Tc, Du].
            u_fut:     Future controls (known).    Shape [B, H,  Du].
            steps:     Forecast horizon H.
            n_samples: Number of Monte Carlo samples for uncertainty bands.

        Returns:
            (y_mean, y_q10, y_q90, pr_mean, pr_q10, pr_q90) — each [B, H, Dy].
        """
        B, Tc, _ = y_ctx.shape

        # Encode context; only the final hidden state is used as the posterior.
        rnn_in           = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _             = self.gru(rnn_in)
        q_params         = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT        = torch.clamp(logvar_qT, -12.0, 6.0)

        if self.emission_uses_u:
            uh_ctx, h_u = self.u_gru(u_ctx)
            # Warm up the reservoir by replaying the full context sequence.
            s_ctx = torch.zeros(B, self.y_dim, self.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = self._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps, ysamps_pr = [], []
        for _ in range(n_samples):
            z     = self.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone()   if self.emission_uses_u else None
            s     = s_ctx.clone() if self.emission_uses_u else None

            preds, preds_pr = [], []
            for k in range(steps):
                u_t              = u_fut[:, k]
                mu_p, logvar_p   = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p         = torch.clamp(logvar_p, -12.0, 6.0)
                z                = self.sample(mu_p, logvar_p)

                if self.use_linear_model:
                    ctrl = self.ctrl_lin(u_t)

                if self.emission_uses_u:
                    uh_t, h_u_s = self.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t        = uh_t.squeeze(1)
                    e_in        = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s           = self._reservoir_step(s, uh_t)
                else:
                    e_in = z

                emit_out              = self.emit(e_in)
                res, cov_factor, cov_diag = self._parse_emit(emit_out, B)
                y_hat                 = ctrl + res if self.use_linear_model else res

                # pr: sample from base Normal then push through the forward flow.
                out                                         = self.emit_pr(e_in)
                mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)
                sigma_pr = torch.exp(log_sigma_t) + self.eps
                x_samp   = mu_t + sigma_pr * torch.randn_like(mu_t)

                dist_y = self._emission_dist(y_hat, cov_factor, cov_diag)
                preds.append(dist_y.sample())
                preds_pr.append(sinh_arcsinh_forward(x_samp, eps_skew_t, log_delta_t, eps=self.eps))

            ysamps.append(torch.stack(preds,    dim=1))
            ysamps_pr.append(torch.stack(preds_pr, dim=1))

        samp    = torch.stack(ysamps,    dim=0)   # [S, B, H, Dy]
        samp_pr = torch.stack(ysamps_pr, dim=0)
        return (
            samp.mean(0),    samp.quantile(0.10, 0),    samp.quantile(0.90, 0),
            samp_pr.mean(0), samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0),
        )

    @torch.no_grad()
    def forecast_deterministic(
        self,
        y_ctx: torch.Tensor,
        u_ctx: torch.Tensor,
        u_fut: torch.Tensor,
        steps: int,
        n_samples: int = 50,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Deterministic (mean-path) forecast, averaged over latent samples.

        Identical to `forecast` but uses the emission mean instead of drawing
        from the full distribution, making it faster and lower-variance. This
        is used for the rollout MSE term during training.

        Args:
            y_ctx:     Observed target context.    Shape [B, Tc, Dy].
            u_ctx:     Control context.            Shape [B, Tc, Du].
            u_fut:     Future controls (known).    Shape [B, H,  Du].
            steps:     Forecast horizon H.
            n_samples: Number of latent-space samples to average over.

        Returns:
            (y_mean, y_q10, y_q90, pr_mean, pr_q10, pr_q90) — each [B, H, Dy].
        """
        B, Tc, _ = y_ctx.shape

        rnn_in           = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _             = self.gru(rnn_in)
        q_params         = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT        = torch.clamp(logvar_qT, -12.0, 6.0)

        if self.emission_uses_u:
            uh_ctx, h_u = self.u_gru(u_ctx)
            s_ctx = torch.zeros(B, self.y_dim, self.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = self._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps, ysamps_pr = [], []
        for _ in range(n_samples):
            z     = self.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone()   if self.emission_uses_u else None
            s     = s_ctx.clone() if self.emission_uses_u else None

            preds, preds_pr = [], []
            for k in range(steps):
                u_t            = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p       = torch.clamp(logvar_p, -12.0, 6.0)
                z              = self.sample(mu_p, logvar_p)

                if self.use_linear_model:
                    ctrl = self.ctrl_lin(u_t)

                if self.emission_uses_u:
                    uh_t, h_u_s = self.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t        = uh_t.squeeze(1)
                    e_in        = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s           = self._reservoir_step(s, uh_t)
                else:
                    e_in = z

                emit_out = self.emit(e_in)
                # Only the mean is needed; skip cov params for speed.
                res   = emit_out[:, :self.y_dim]
                y_hat = ctrl + res if self.use_linear_model else res

                out                                         = self.emit_pr(e_in)
                mu_t, _, eps_skew_t, log_delta_t            = torch.chunk(out, 4, dim=-1)
                y_s_pr = sinh_arcsinh_forward(mu_t, eps_skew_t, log_delta_t, eps=self.eps)

                preds.append(y_hat)
                preds_pr.append(y_s_pr)

            ysamps.append(torch.stack(preds,    dim=1))
            ysamps_pr.append(torch.stack(preds_pr, dim=1))

        samp    = torch.stack(ysamps,    dim=0)
        samp_pr = torch.stack(ysamps_pr, dim=0)
        return (
            samp.mean(0),    samp.quantile(0.10, 0),    samp.quantile(0.90, 0),
            samp_pr.mean(0), samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0),
        )


# ---------------------------------------------------------------------------
# Control distribution utilities (OOD detection)
# ---------------------------------------------------------------------------

def fit_control_mahalanobis(
    u_train_norm: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Mahalanobis distance model to the training control distribution.

    Used to flag out-of-distribution control inputs at inference time.

    Args:
        u_train_norm: Standardised training controls.  Shape [N, T, Du].
        eps:          Ridge added to the covariance diagonal for invertibility.

    Returns:
        mu:      Per-feature mean.            Shape [Du].
        inv_cov: Inverse of the covariance.   Shape [Du, Du].
    """
    U       = u_train_norm.reshape(-1, u_train_norm.shape[-1])
    mu      = U.mean(axis=0)
    cov     = np.cov(U.T) + eps * np.eye(U.shape[1])
    inv_cov = np.linalg.inv(cov)
    return mu.astype(np.float32), inv_cov.astype(np.float32)


def mahalanobis_score(
    u_fut_norm: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
) -> np.ndarray:
    """Per-step Mahalanobis distance of future controls from the training distribution.

    Args:
        u_fut_norm: Standardised future controls.  Shape [B, H, Du].
        mu:         Training mean.                 Shape [Du].
        inv_cov:    Inverse training covariance.   Shape [Du, Du].

    Returns:
        Mahalanobis scores.  Shape [B, H].
    """
    diff = u_fut_norm - mu[None, None, :]
    # Vectorised quadratic form: score[b, h] = diff[b,h]^T @ inv_cov @ diff[b,h]
    return np.einsum("bhd,dd,bhd->bh", diff, inv_cov, diff)


def pick_threshold_from_val(
    val_loader: DataLoader,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """Compute an OOD detection threshold from the validation-set score distribution.

    Args:
        val_loader:  Validation DataLoader; each batch must have u_fut at index 3.
        mu:          Training control mean.                Shape [Du].
        inv_cov:     Inverse training control covariance.  Shape [Du, Du].
        percentile:  Score percentile to use as the threshold.

    Returns:
        Scalar threshold; inputs whose score exceeds this are considered OOD.
    """
    all_scores = []
    for _, _, _, u_fut, _, _ in val_loader:
        s = mahalanobis_score(u_fut.numpy(), mu, inv_cov)
        all_scores.append(s.reshape(-1))
    return float(np.percentile(np.concatenate(all_scores), percentile))


# ---------------------------------------------------------------------------
# Linear baseline
# ---------------------------------------------------------------------------

def fit_ridge_D(
    u_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1e-2,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ridge regression y ≈ u @ W + b independently for each output channel.

    Provides the linear pattern baseline loaded into ctrl_lin before SSM
    training, separating the linear control response from the non-linear residual.

    Args:
        u_train:       Normalised control inputs.   Shape [N, T, Du].
        y_train:       Normalised target outputs.   Shape [N, T, Dy].
        alpha:         Ridge regularisation strength λ.
        fit_intercept: If True, augments X with a bias column and exempts it
                       from regularisation.

    Returns:
        W: Weight matrix.  Shape [Du, Dy].
        b: Bias vector.    Shape [Dy].  (zeros if fit_intercept=False)
    """
    U = u_train.reshape(-1, u_train.shape[-1])
    Y = y_train.reshape(-1, y_train.shape[-1])

    if fit_intercept:
        # Augment with a ones column so the bias is absorbed into the solve.
        ones = np.ones((U.shape[0], 1))
        X    = np.concatenate([U, ones], axis=1)
    else:
        X = U

    # Closed-form ridge: (X^T X + α I)^{-1} X^T Y
    XtX = X.T @ X
    I   = np.eye(XtX.shape[0])
    if fit_intercept:
        # Do not regularise the bias term.
        I[-1, -1] = 0.0

    Wb = np.linalg.solve(XtX + alpha * I, X.T @ Y)

    if fit_intercept:
        W, b = Wb[:-1, :], Wb[-1, :]
    else:
        W, b = Wb, np.zeros((Y.shape[1],), dtype=Y.dtype)

    return W.astype(np.float32), b.astype(np.float32)


def load_into_ctrl_lin(
    model: DeepSSMPatternConditioned,
    W: np.ndarray,
    b: np.ndarray,
    freeze: bool = True,
) -> None:
    """Copy ridge weights into model.ctrl_lin and optionally freeze the parameters.

    Freezing pins the linear baseline at its ridge solution so the SSM learns
    only the residual on top of it. Pass freeze=False to let the baseline adapt
    during training.

    Args:
        model:  Model instance that has ctrl_lin = nn.Linear(u_dim, y_dim).
        W:      Ridge weight matrix.  Shape [Du, Dy].
        b:      Ridge bias vector.    Shape [Dy].
        freeze: If True, disables gradients for ctrl_lin after copying.
    """
    assert hasattr(model, "ctrl_lin"), "Model must have ctrl_lin = nn.Linear(u_dim, y_dim)"
    Du, Dy = W.shape
    assert model.ctrl_lin.in_features  == Du
    assert model.ctrl_lin.out_features == Dy

    with torch.no_grad():
        # nn.Linear stores weight as [out_features, in_features] = [Dy, Du].
        model.ctrl_lin.weight.copy_(torch.from_numpy(W.T))
        model.ctrl_lin.bias.copy_(torch.from_numpy(b))

    if freeze:
        for p in model.ctrl_lin.parameters():
            p.requires_grad = False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_train(
    y_np: np.ndarray,
    pr_np: np.ndarray,
    u_np: np.ndarray,
    context_len: int = 40,
    horizon: int = 12,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 2e-3,
    z_dim: int = 16,
    rnn_hidden: int = 62,
    use_linear_model: bool = True,
    resevoir_dim: int = 2,
    cov_rank: int = 5,
    run_dir: str | None = None,
    weights_file: str | None = None,
) -> tuple:
    """Full DDP training pipeline with early stopping.

    Pipeline:
      1. Initialise the distributed process group (NCCL for GPU, Gloo for CPU).
      2. Split data 80/20 into train/val by series index.
      3. Fit per-feature StandardScalers on the training split only to avoid leakage.
      4. Optionally fit a ridge baseline and load it into ctrl_lin, frozen for
         the whole run.
      5. Build sliding-window DataLoaders.
      6. Train with a composite loss:
             ELBO  = NLL_y + NLL_pr + kl_w * KL
             + alpha * rollout Huber-MSE for y
             + omega * rollout Huber-MSE for pr
             + gamma * yearly-average MSE for y   (penalises multi-year trend errors)
         kl_w, alpha, omega are linearly annealed over the first 30 % of steps.
         gamma has no warmup — trend accuracy is prioritised throughout.
      7. Checkpoint every 10 epochs if run_dir is provided.
      8. Early stopping on validation ELBO with patience=15 epochs.

    Args:
        y_np:             Target array.            Shape [N, T, Dy].
        pr_np:            Precipitation array.     Shape [N, T, Dy].
        u_np:             Control array.           Shape [N, T, Du].
        context_len:      Context window length Tc.
        horizon:          Forecast horizon H.
        batch_size:       Training batch size.
        epochs:           Maximum number of training epochs.
        lr:               Initial AdamW learning rate.
        z_dim:            Latent state dimensionality.
        rnn_hidden:       Inference GRU hidden size.
        use_linear_model: Whether to include the ridge-initialised linear head.
        resevoir_dim:     EMA reservoir dimension per output channel.
        cov_rank:         Rank of the low-rank covariance factor; 0 for a
                          diagonal y emission.
        run_dir:          If provided, scalers and checkpoints are saved here.
        weights_file:     If provided, model weights are warm-started from this path.

    Returns:
        (model, y_scaler, u_scaler, pr_scaler) — trained model and fitted scalers.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda   = torch.cuda.is_available()
    backend    = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend)
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")

    # 80/20 random split by series index.
    N                  = y_np.shape[0]
    idx                = np.random.permutation(N)
    n_train            = int(0.8 * N)
    tr_idx, va_idx     = idx[:n_train], idx[n_train:]
    y_tr,  pr_tr,  u_tr  = y_np[tr_idx],  pr_np[tr_idx],  u_np[tr_idx]
    y_va,  pr_va,  u_va  = y_np[va_idx],  pr_np[va_idx],  u_np[va_idx]

    print("training start")

    # Fit scalers on training data only to prevent leakage into validation.
    y_scaler  = utils.StandardScaler().fit(y_tr)
    u_scaler  = utils.StandardScaler().fit(u_tr)
    pr_scaler = utils.StandardScaler().fit(pr_tr)
    print("Created standard scaler")

    if run_dir is not None:
        y_scaler.save(os.path.join(run_dir,  "y_scaler.out"))
        u_scaler.save(os.path.join(run_dir,  "u_scaler.out"))
        pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))

    y_trn  = y_scaler.transform(y_tr);   y_van  = y_scaler.transform(y_va)
    pr_trn = pr_scaler.transform(pr_tr); pr_van = pr_scaler.transform(pr_va)
    u_trn  = u_scaler.transform(u_tr);   u_van  = u_scaler.transform(u_va)
    print("data standardised.")

    if use_linear_model:
        W, b = fit_ridge_D(u_trn, y_trn, alpha=1e-2, fit_intercept=True)
        print("ridge regression completed")

    train_ds = UnifiedWindowDataset(y_trn, pr_trn, u_trn, context_len=context_len, horizon=horizon)
    val_ds   = UnifiedWindowDataset(y_van, pr_van, u_van, context_len=context_len, horizon=horizon)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    print("dataloaders prepared")

    Dy, Du = y_np.shape[-1], u_np.shape[-1]
    raw_model = DeepSSMPatternConditioned(
        y_dim=Dy, u_dim=Du, z_dim=z_dim, rnn_hidden=rnn_hidden,
        use_linear_model=use_linear_model, emission_uses_u=True,
        reservoir_dim=resevoir_dim, cov_rank=cov_rank,
    ).to(device)

    if weights_file is not None:
        raw_model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"Loaded weights from {weights_file}")

    if use_linear_model:
        # Freeze ctrl_lin for the first phase of training.
        load_into_ctrl_lin(raw_model, W, b, freeze=True)

    ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if use_cuda else {}
    model = DDP(raw_model, **ddp_kwargs)

    # Only optimise parameters that have gradients (ctrl_lin stays frozen).
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)

    best_val, best_state        = float("inf"), None
    patience, patience_left     = 15, 15
    total_steps                 = epochs * len(train_dl)
    global_step                 = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = []

        for y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut in train_dl:
            y_ctx  = torch.tensor(y_ctx,  device=device)
            pr_ctx = torch.tensor(pr_ctx, device=device)
            u_ctx  = torch.tensor(u_ctx,  device=device)
            u_fut  = torch.tensor(u_fut,  device=device)
            y_fut  = torch.tensor(y_fut,  device=device)
            pr_fut = torch.tensor(pr_fut, device=device)

            # Train on the full context+horizon sequence so the model learns
            # dynamics across the context/forecast boundary.
            y_full  = torch.cat([y_ctx,  y_fut],  dim=1)
            pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
            u_full  = torch.cat([u_ctx,  u_fut],  dim=1)
            B, T, _ = y_full.shape

            nll, kl, nll_pr = raw_model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)
            mean, _, _, mean_pr, _, _ = raw_model.forecast_deterministic(
                y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
            roll_out_mse    = F.huber_loss(mean,    y_fut,  delta=1.0)
            roll_out_mse_pr = F.huber_loss(mean_pr, pr_fut, delta=1.0)

            if use_linear_model:
                lin_mean = raw_model.ctrl_lin(u_full.reshape(-1, Du)).reshape(B, T, Dy)
                lin_mse  = ((lin_mean - y_full) ** 2).mean()

            # Yearly-average loss: penalise trend errors over complete 12-month blocks.
            n_complete_years = horizon // 12
            if n_complete_years > 0:
                H_yr                = n_complete_years * 12
                mean_yr             = mean[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                y_fut_yr            = y_fut[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                roll_out_mse_yearly = ((mean_yr - y_fut_yr) ** 2).mean()
            else:
                roll_out_mse_yearly = torch.tensor(0.0, device=device)

            # Linear warmup of rollout loss weights over the first 30 % of steps.
            global_step += 1
            frac  = min(1.0, global_step / int(0.3 * total_steps))
            kl_w  = 5
            alpha = 10000 * frac
            omega = 500   * frac
            gamma = 80000          # yearly trend weight — no warmup needed

            loss = (nll + nll_pr + kl_w * kl + alpha * roll_out_mse
                    + omega * roll_out_mse_pr + gamma * roll_out_mse_yearly)

            if global_step % 100 == 0:
                if use_linear_model:
                    print("loss: ", nll.item(), kl_w, kl.item(), roll_out_mse.item(),
                          lin_mse.item(), nll_pr.item(), roll_out_mse_pr.item(), roll_out_mse_yearly.item())
                else:
                    print("loss: ", nll.item(), kl_w, kl.item(), roll_out_mse.item(),
                          nll_pr.item(), roll_out_mse_pr.item(), roll_out_mse_yearly.item())

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss.append(loss.item())

        # ---- Validation ----
        model.eval()
        va_loss, va_mse = [], []

        with torch.no_grad():
            for y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut in val_dl:
                y_ctx  = torch.tensor(y_ctx,  device=device)
                pr_ctx = torch.tensor(pr_ctx, device=device)
                u_ctx  = torch.tensor(u_ctx,  device=device)
                u_fut  = torch.tensor(u_fut,  device=device)
                y_fut  = torch.tensor(y_fut,  device=device)
                pr_fut = torch.tensor(pr_fut, device=device)

                y_full  = torch.cat([y_ctx,  y_fut],  dim=1)
                u_full  = torch.cat([u_ctx,  u_fut],  dim=1)
                pr_full = torch.cat([pr_ctx, pr_fut], dim=1)

                nll, kl, nll_pr = raw_model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)
                mean, _, _, mean_pr, _, _ = raw_model.forecast(
                    y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
                mse    = ((mean    - y_fut)  ** 2).mean().item()
                mse_pr = ((mean_pr - pr_fut) ** 2).mean().item()
                loss   = nll + nll_pr + kl_w * kl + alpha * mse + omega * mse_pr
                va_loss.append(loss.item())
                va_mse.append(mse)

        tr  = float(np.mean(tr_loss))
        va  = float(np.mean(va_loss))
        mse = float(np.mean(va_mse))
        print(f"epoch {epoch:03d} | train {tr:.4f} | val_elbo {va:.4f} | val_mse {mse:.4f}")

        if run_dir is not None and epoch % 10 == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, f"model_epoch{epoch:04d}.pt"))

        # Early stopping on validation ELBO.
        if va < best_val - 1e-4:
            best_val      = va
            best_state    = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

    if best_state is not None:
        raw_model.load_state_dict(best_state)

    return raw_model, y_scaler, u_scaler, pr_scaler
