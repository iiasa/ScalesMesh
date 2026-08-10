"""Tests for UnifiedWindowDataset.

Focus: window indexing correctness — the most common source of silent bugs
in sliding-window datasets (off-by-one counts, wrong slice offsets, shape errors).
"""

import numpy as np
import pytest

from scales.model.ssm_tas_pr import UnifiedWindowDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(N: int = 3, T: int = 60, Dy: int = 2, Du: int = 3):
    rng = np.random.default_rng(0)
    y   = rng.standard_normal((N, T, Dy)).astype(np.float32)
    pr  = rng.standard_normal((N, T, Dy)).astype(np.float32)
    u   = rng.standard_normal((N, T, Du)).astype(np.float32)
    return y, pr, u


Tc, H = 20, 10   # used throughout


# ---------------------------------------------------------------------------
# Window count
# ---------------------------------------------------------------------------

class TestWindowCount:
    def test_stride1_single_series(self):
        T = 60
        y, pr, u = make_data(N=1, T=T)
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H, stride=1)
        # valid starts: 0 .. T-Tc-H inclusive
        assert len(ds) == T - Tc - H + 1

    def test_stride1_multi_series(self):
        N, T = 4, 60
        y, pr, u = make_data(N=N, T=T)
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H, stride=1)
        assert len(ds) == N * (T - Tc - H + 1)

    def test_stride2(self):
        N, T = 3, 60
        y, pr, u = make_data(N=N, T=T)
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H, stride=2)
        max_start         = T - Tc - H
        windows_per_series = len(range(0, max_start + 1, 2))
        assert len(ds) == N * windows_per_series

    def test_start_mode_zero(self):
        N = 5
        y, pr, u = make_data(N=N)
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H, start_mode="zero")
        assert len(ds) == N


# ---------------------------------------------------------------------------
# Item shapes
# ---------------------------------------------------------------------------

class TestItemShapes:
    def test_output_shapes(self):
        Dy, Du = 2, 3
        y, pr, u = make_data(Dy=Dy, Du=Du)
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H)
        y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut = ds[0]
        assert y_ctx.shape  == (Tc, Dy)
        assert pr_ctx.shape == (Tc, Dy)
        assert u_ctx.shape  == (Tc, Du)
        assert u_fut.shape  == (H,  Du)
        assert pr_fut.shape == (H,  Dy)
        assert y_fut.shape  == (H,  Dy)


# ---------------------------------------------------------------------------
# Slice correctness — the core indexing invariant
# ---------------------------------------------------------------------------

class TestSliceCorrectness:
    """Each window must be an exact, correctly placed slice of the source array."""

    def test_context_and_horizon_match_source(self):
        N, T, Dy, Du = 2, 60, 2, 3
        y, pr, u = make_data(N=N, T=T, Dy=Dy, Du=Du)
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H)

        # Spot-check first, a middle, and last window.
        for idx in [0, len(ds) // 2, len(ds) - 1]:
            y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut = ds[idx]
            s, start = ds.index[idx]

            np.testing.assert_array_equal(y_ctx,  y[s,  start       : start + Tc])
            np.testing.assert_array_equal(y_fut,  y[s,  start + Tc  : start + Tc + H])
            np.testing.assert_array_equal(pr_ctx, pr[s, start       : start + Tc])
            np.testing.assert_array_equal(pr_fut, pr[s, start + Tc  : start + Tc + H])
            np.testing.assert_array_equal(u_ctx,  u[s,  start       : start + Tc])
            np.testing.assert_array_equal(u_fut,  u[s,  start + Tc  : start + Tc + H])

    def test_context_and_horizon_are_contiguous(self):
        """y_ctx immediately precedes y_fut in the original series."""
        y, pr, u = make_data()
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H)
        y_ctx, _, _, _, _, y_fut = ds[0]
        s, start = ds.index[0]
        full_window = y[s, start : start + Tc + H]
        np.testing.assert_array_equal(
            np.concatenate([y_ctx, y_fut], axis=0), full_window
        )


# ---------------------------------------------------------------------------
# 2-D input promotion
# ---------------------------------------------------------------------------

class TestInputPromotion:
    def test_2d_arrays_treated_as_single_series(self):
        T, Dy, Du = 60, 2, 3
        rng = np.random.default_rng(1)
        y   = rng.standard_normal((T, Dy)).astype(np.float32)
        pr  = rng.standard_normal((T, Dy)).astype(np.float32)
        u   = rng.standard_normal((T, Du)).astype(np.float32)
        ds = UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H)
        assert len(ds) == T - Tc - H + 1

    def test_2d_and_3d_identical_windows(self):
        """Passing y[0] (2-D) should give the same windows as y[[0]] (3-D)."""
        N, T, Dy, Du = 1, 60, 2, 3
        y, pr, u = make_data(N=N, T=T, Dy=Dy, Du=Du)

        ds_3d = UnifiedWindowDataset(y,      pr,      u,      context_len=Tc, horizon=H)
        ds_2d = UnifiedWindowDataset(y[0],   pr[0],   u[0],   context_len=Tc, horizon=H)

        assert len(ds_3d) == len(ds_2d)
        for i in range(len(ds_3d)):
            for a, b in zip(ds_3d[i], ds_2d[i]):
                np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_series_too_short_raises(self):
        y, pr, u = make_data(N=1, T=25)
        with pytest.raises(ValueError, match=r"context_len \+ horizon"):
            UnifiedWindowDataset(y, pr, u, context_len=20, horizon=10)

    def test_invalid_start_mode_raises(self):
        y, pr, u = make_data()
        with pytest.raises(ValueError, match="start_mode"):
            UnifiedWindowDataset(y, pr, u, context_len=Tc, horizon=H, start_mode="invalid")
