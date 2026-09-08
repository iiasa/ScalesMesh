"""Tests for diag_gaussian_kl and DeepSSMPatternConditioned.

Focus:
  - KL(q || q) == 0 validates the full KL path through forward_elbo.
  - forward_elbo / forecast shape and sanity checks catch regressions in the
    model architecture without requiring a full training run.
"""

import pytest
import torch

from scales.model.ssm_model_utils import diag_gaussian_kl
from scales.model.ssm_tas_pr import DeepSSMPatternConditioned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_model(emission_uses_u: bool = True, use_linear_model: bool = True,
                cov_rank: int = 2):
    return DeepSSMPatternConditioned(
        y_dim=2, u_dim=3, z_dim=4,
        rnn_hidden=8, u_rnn_hidden=8, mlp_hidden=16,
        emission_uses_u=emission_uses_u,
        use_linear_model=use_linear_model,
        reservoir_dim=2, cov_rank=cov_rank,
    )


def random_batch(B: int = 4, T: int = 10, Dy: int = 2, Du: int = 3):
    return (
        torch.randn(B, T, Dy),
        torch.randn(B, T, Dy),
        torch.randn(B, T, Du),
    )


# ---------------------------------------------------------------------------
# diag_gaussian_kl
# ---------------------------------------------------------------------------

class TestDiagGaussianKL:
    def test_identical_distributions_is_zero(self):
        """KL(q || q) must be exactly 0 — strongest single correctness check."""
        B, Z   = 8, 6
        mu     = torch.randn(B, Z)
        logvar = torch.randn(B, Z)
        kl = diag_gaussian_kl(mu, logvar, mu, logvar)
        torch.testing.assert_close(kl, torch.zeros(B), atol=1e-5, rtol=0)

    def test_non_negative(self):
        """KL divergence is always >= 0."""
        B, Z = 16, 8
        kl = diag_gaussian_kl(
            torch.randn(B, Z), torch.randn(B, Z),
            torch.randn(B, Z), torch.randn(B, Z),
        )
        assert (kl >= 0).all(), f"Negative KL values found: {kl[kl < 0]}"

    def test_known_analytical_value(self):
        """KL(N(1, 1) || N(0, 1)) = 0.5 per dimension, summed over Z."""
        Z        = 4
        mu_q     = torch.ones(1, Z)
        logvar_q = torch.zeros(1, Z)   # var = 1
        mu_p     = torch.zeros(1, Z)
        logvar_p = torch.zeros(1, Z)   # var = 1
        kl = diag_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        # 0.5 * (0 - 0 + (1 + 1^2)/1 - 1) = 0.5 per dim, summed over Z
        torch.testing.assert_close(kl, torch.tensor([0.5 * Z]), atol=1e-5, rtol=0)

    def test_output_shape(self):
        B, Z = 5, 3
        kl = diag_gaussian_kl(
            torch.randn(B, Z), torch.randn(B, Z),
            torch.randn(B, Z), torch.randn(B, Z),
        )
        assert kl.shape == (B,)


# ---------------------------------------------------------------------------
# DeepSSMPatternConditioned — forward_elbo
# ---------------------------------------------------------------------------

class TestForwardElbo:
    def test_returns_finite_scalars(self):
        model    = small_model()
        y, pr, u = random_batch()
        nll, kl, nll_pr = model.forward_elbo(y, pr, u)
        for name, t in [("nll", nll), ("kl", kl), ("nll_pr", nll_pr)]:
            assert t.shape == (), f"{name} should be a scalar"
            assert torch.isfinite(t),  f"{name} is not finite: {t.item()}"

    def test_kl_non_negative(self):
        model    = small_model()
        y, pr, u = random_batch()
        _, kl, _ = model.forward_elbo(y, pr, u)
        assert kl.item() >= 0, f"KL is negative: {kl.item()}"

    @pytest.mark.parametrize("emission_uses_u,use_linear_model", [
        (False, False),
        (False, True),
        (True,  False),
        (True,  True),
    ])
    def test_all_flag_combinations(self, emission_uses_u, use_linear_model):
        model    = small_model(emission_uses_u=emission_uses_u, use_linear_model=use_linear_model)
        y, pr, u = random_batch()
        nll, kl, nll_pr = model.forward_elbo(y, pr, u)
        assert torch.isfinite(nll)
        assert torch.isfinite(kl)
        assert torch.isfinite(nll_pr)


# ---------------------------------------------------------------------------
# DeepSSMPatternConditioned — forecast
# ---------------------------------------------------------------------------

class TestForecast:
    B, Tc, H, Dy, Du = 4, 10, 5, 2, 3

    @pytest.fixture
    def ctx_and_fut(self):
        B, Tc, H, Dy, Du = self.B, self.Tc, self.H, self.Dy, self.Du
        return (
            torch.randn(B, Tc, Dy),
            torch.randn(B, Tc, Du),
            torch.randn(B, H,  Du),
        )

    def test_output_shapes(self, ctx_and_fut):
        model           = small_model()
        y_ctx, u_ctx, u_fut = ctx_and_fut
        outputs         = model.forecast(y_ctx, u_ctx, u_fut, steps=self.H, n_samples=10)
        assert len(outputs) == 6
        for t in outputs:
            assert t.shape == (self.B, self.H, self.Dy)

    def test_quantile_ordering(self, ctx_and_fut):
        """q10 <= q90 must hold element-wise for both y and pr."""
        model              = small_model()
        y_ctx, u_ctx, u_fut = ctx_and_fut
        _, q10, q90, _, q10_pr, q90_pr = model.forecast(
            y_ctx, u_ctx, u_fut, steps=self.H, n_samples=50)
        assert (q10 <= q90).all(),    "y q10 > q90 found"
        assert (q10_pr <= q90_pr).all(), "pr q10 > q90 found"

    def test_deterministic_shapes(self, ctx_and_fut):
        model              = small_model()
        y_ctx, u_ctx, u_fut = ctx_and_fut
        outputs            = model.forecast_deterministic(
            y_ctx, u_ctx, u_fut, steps=self.H, n_samples=10)
        assert len(outputs) == 6
        for t in outputs:
            assert t.shape == (self.B, self.H, self.Dy)


# ---------------------------------------------------------------------------
# y emission covariance — diagonal (cov_rank=0) vs low-rank (cov_rank>0)
# ---------------------------------------------------------------------------

class TestEmissionCovariance:
    """The emit head layout is [mean (D) | log_cov_diag (D) | factor (D*r)].

    cov_rank=0 must drop the factor block and fall back to a diagonal Normal
    without disturbing that layout, so checkpoints stay loadable either way.
    """

    Dy, Du, B = 2, 3, 4

    def test_head_width_scales_with_rank(self):
        for r in (0, 1, 2, 5):
            model = small_model(cov_rank=r)
            assert model.emit.net[-1].out_features == self.Dy * (2 + r)

    def test_diagonal_has_no_factor_block(self):
        model = small_model(cov_rank=0)
        emit_out = torch.randn(self.B, self.Dy * 2)
        res, cov_factor, cov_diag = model._parse_emit(emit_out, self.B)
        assert cov_factor is None, "rank 0 must not produce a covariance factor"
        assert res.shape == (self.B, self.Dy)
        assert cov_diag.shape == (self.B, self.Dy)
        assert (cov_diag > 0).all(), "cov_diag must be strictly positive"

    def test_low_rank_keeps_factor_block(self):
        r = 2
        model = small_model(cov_rank=r)
        emit_out = torch.randn(self.B, self.Dy * (2 + r))
        _, cov_factor, _ = model._parse_emit(emit_out, self.B)
        assert cov_factor.shape == (self.B, self.Dy, r)

    @pytest.mark.parametrize("cov_rank", [0, 1, 3])
    def test_distribution_contract_is_rank_independent(self, cov_rank):
        """log_prob must be [B] and sample [B, Dy] for every rank."""
        model = small_model(cov_rank=cov_rank)
        emit_out = torch.randn(self.B, self.Dy * (2 + cov_rank))
        res, cov_factor, cov_diag = model._parse_emit(emit_out, self.B)
        dist = model._emission_dist(res, cov_factor, cov_diag)
        assert dist.log_prob(torch.randn(self.B, self.Dy)).shape == (self.B,)
        assert dist.sample().shape == (self.B, self.Dy)

    def test_diagonal_matches_low_rank_with_zero_factor(self):
        """A rank-1 factor of zeros is exactly a diagonal covariance.

        This pins the diagonal branch to the same density as the low-rank one,
        including the sqrt on cov_diag (which holds variances, not scales).
        """
        model = small_model(cov_rank=0)
        emit_out = torch.randn(self.B, self.Dy * 2)
        res, _, cov_diag = model._parse_emit(emit_out, self.B)

        diagonal = model._emission_dist(res, None, cov_diag)
        low_rank = torch.distributions.LowRankMultivariateNormal(
            loc=res, cov_factor=torch.zeros(self.B, self.Dy, 1), cov_diag=cov_diag)

        y = torch.randn(self.B, self.Dy)
        torch.testing.assert_close(
            diagonal.log_prob(y), low_rank.log_prob(y), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(diagonal.variance, cov_diag, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("cov_rank", [0, 2])
    def test_forward_elbo_runs(self, cov_rank):
        model   = small_model(cov_rank=cov_rank)
        y, pr, u = random_batch()
        nll, kl, nll_pr = model.forward_elbo(y, pr, u)
        for name, t in (("nll", nll), ("kl", kl), ("nll_pr", nll_pr)):
            assert t.shape == () and torch.isfinite(t), f"{name} is not a finite scalar"

    @pytest.mark.parametrize("cov_rank", [0, 2])
    @pytest.mark.parametrize("method", ["forecast", "forecast_deterministic"])
    def test_forecast_runs(self, cov_rank, method):
        Tc, H = 6, 3
        model = small_model(cov_rank=cov_rank)
        outputs = getattr(model, method)(
            torch.randn(self.B, Tc, self.Dy),
            torch.randn(self.B, Tc, self.Du),
            torch.randn(self.B, H,  self.Du),
            steps=H, n_samples=5,
        )
        assert len(outputs) == 6
        for t in outputs:
            assert t.shape == (self.B, H, self.Dy)
            assert torch.isfinite(t).all()

    @pytest.mark.parametrize("cov_rank", [0, 2])
    def test_gradients_reach_the_emit_head(self, cov_rank):
        model    = small_model(cov_rank=cov_rank)
        y, pr, u = random_batch()
        model.zero_grad()
        model.forward_elbo(y, pr, u)[0].backward()
        grad = model.emit.net[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, "emit head received no gradient"

    @pytest.mark.parametrize("cov_rank", [0, 2, 5])
    def test_checkpoint_round_trip(self, cov_rank):
        """A state dict must reload strictly into a model of the same rank."""
        saved = small_model(cov_rank=cov_rank).state_dict()
        small_model(cov_rank=cov_rank).load_state_dict(saved, strict=True)

    def test_ranks_are_not_interchangeable(self):
        """Loading across ranks must fail loudly, not silently reinterpret weights."""
        saved = small_model(cov_rank=2).state_dict()
        with pytest.raises(RuntimeError, match="size mismatch"):
            small_model(cov_rank=0).load_state_dict(saved, strict=True)
