"""Tests for diag_gaussian_kl and DeepSSMPatternConditioned.

Focus:
  - KL(q || q) == 0 validates the full KL path through forward_elbo.
  - forward_elbo / forecast shape and sanity checks catch regressions in the
    model architecture without requiring a full training run.
"""

import pytest
import torch

from scales.model.model_utils import diag_gaussian_kl
from scales.model.ssm_tas_pr import DeepSSMPatternConditioned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_model(emission_uses_u: bool = True, use_linear_model: bool = True):
    return DeepSSMPatternConditioned(
        y_dim=2, u_dim=3, z_dim=4,
        rnn_hidden=8, u_rnn_hidden=8, mlp_hidden=16,
        emission_uses_u=emission_uses_u,
        use_linear_model=use_linear_model,
        reservoir_dim=2, cov_rank=2,
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
