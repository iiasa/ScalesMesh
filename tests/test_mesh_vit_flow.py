"""Tests for mesh.mesh_vit_flow: tokenizers, attention blocks, the ViT model, EMA, and flow matching.

Focus:
  - patchify/unpatchify must round-trip exactly (a mismatch here silently
    scrambles pixels without ever raising).
  - ViTCondDiffusionSR's forward pass must actually depend on the LR
    conditioning tensor -- if it doesn't, the model has learned to ignore
    its only conditioning signal without any shape check catching it.
  - ModelEMA's update rule and state_dict round-trip (an EMA that doesn't
    actually move, or a save/load that swaps model weights for random init,
    fails silently in training and is only noticed much later).
  - FlowMatching / FlowMatchingLinear: sample_xt endpoints at t=0/1, and
    target_velocity checked against a finite-difference derivative of
    sample_xt rather than by re-deriving the same formula.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mesh.mesh_vit_flow import (
    AdaLNScaleShift,
    CrossAttention,
    FlowMatching,
    FlowMatchingLinear,
    GraphEncoder,
    ModelEMA,
    SelfAttention,
    SinusoidalTimeEmbedding,
    TransformerBlock,
    ViTCondDiffusionSR,
    patchify,
    set_seed,
    unpatchify,
)


def tiny_model(**overrides) -> ViTCondDiffusionSR:
    kwargs = {
        "varmax": (300.0, 5.0), "varmin": (250.0, 0.0),
        "in_ch": 2, "dim": 16, "depth": 2, "heads": 2, "patch": 2, "lr_patch": 2,
        "hr_hw": (4, 8), "lr_regions": 6,
    }
    kwargs.update(overrides)
    return ViTCondDiffusionSR(**kwargs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_set_seed_reproducible():
    set_seed(123)
    a = (torch.rand(3).clone(), np.random.rand(3).copy(), random.random())
    set_seed(123)
    b = (torch.rand(3).clone(), np.random.rand(3).copy(), random.random())
    torch.testing.assert_close(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    assert a[2] == b[2]


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


class TestPatchify:
    def test_shape(self):
        x = torch.randn(2, 3, 8, 8)
        patches = patchify(x, patch_size=4)
        assert patches.shape == (2, 4, 4 * 4 * 3)

    def test_roundtrip(self):
        x = torch.randn(2, 3, 8, 8)
        patches = patchify(x, patch_size=4)
        back = unpatchify(patches, patch_size=4, img_hw=(8, 8), out_ch=3)
        torch.testing.assert_close(back, x)

    def test_roundtrip_rectangular(self):
        x = torch.randn(2, 2, 4, 8)
        patches = patchify(x, patch_size=2)
        back = unpatchify(patches, patch_size=2, img_hw=(4, 8), out_ch=2)
        torch.testing.assert_close(back, x)


# ---------------------------------------------------------------------------
# Embeddings and attention blocks
# ---------------------------------------------------------------------------


class TestSinusoidalTimeEmbedding:
    def test_shape_and_t_zero_values(self):
        emb = SinusoidalTimeEmbedding(dim=16)
        out = emb(torch.tensor([0.0, 500.0, 999.0]))
        assert out.shape == (3, 16)
        half = 8
        torch.testing.assert_close(out[0, :half], torch.zeros(half), atol=1e-5, rtol=0)
        torch.testing.assert_close(out[0, half:], torch.ones(half), atol=1e-5, rtol=0)

    def test_odd_dim_is_zero_padded(self):
        emb = SinusoidalTimeEmbedding(dim=15)
        out = emb(torch.tensor([1.0]))
        assert out.shape == (1, 15)
        assert out[0, -1].item() == 0.0


class TestAdaLNScaleShift:
    def test_zero_cond_reduces_to_layernorm(self):
        layer = AdaLNScaleShift(dim=8, cond_dim=4)
        with torch.no_grad():
            layer.proj.weight.zero_()
            layer.proj.bias.zero_()
        x = torch.randn(2, 5, 8)
        out = layer(x, torch.zeros(2, 4))
        torch.testing.assert_close(out, layer.norm(x))


class TestAttention:
    def test_self_attention_preserves_shape(self):
        attn = SelfAttention(dim=16, heads=4).eval()
        x = torch.randn(2, 6, 16)
        assert attn(x).shape == x.shape

    def test_cross_attention_output_shape_follows_query(self):
        attn = CrossAttention(dim_q=16, dim_kv=16, heads=4).eval()
        q = torch.randn(2, 6, 16)
        kv = torch.randn(2, 9, 16)
        assert attn(q, kv).shape == q.shape

    def test_transformer_block_preserves_hr_shape(self):
        block = TransformerBlock(dim=16, heads=4, cond_dim=8).eval()
        x_hr = torch.randn(2, 5, 16)
        tokens_lr = torch.randn(2, 3, 16)
        cond = torch.randn(2, 8)
        assert block(x_hr, cond, tokens_lr).shape == x_hr.shape

    def test_graph_encoder_shape(self):
        enc = GraphEncoder(in_dim=2, d_model=16, n_heads=2, n_layers=1).eval()
        out = enc(torch.randn(3, 5, 2))
        assert out.shape == (3, 5, 16)


# ---------------------------------------------------------------------------
# ViTCondDiffusionSR
# ---------------------------------------------------------------------------


class TestViTCondDiffusionSR:
    def test_forward_shape(self):
        model = tiny_model().eval()
        B = 3
        x = torch.randn(B, 2, 4, 8)
        t = torch.rand(B)
        lr = torch.randn(B, 2, 6)
        with torch.no_grad():
            out = model(x, t, lr)
        assert out.shape == x.shape

    def test_varmax_varmin_registered_as_buffers(self):
        model = tiny_model()
        buffers = dict(model.named_buffers())
        torch.testing.assert_close(buffers["varmax"], torch.tensor([300.0, 5.0]))
        torch.testing.assert_close(buffers["varmin"], torch.tensor([250.0, 0.0]))

    def test_output_depends_on_lr_conditioning(self):
        """A model that ignores its only conditioning input is broken silently -- shape checks won't catch it."""
        torch.manual_seed(0)
        model = tiny_model().eval()
        x = torch.randn(2, 2, 4, 8)
        t = torch.rand(2)
        lr1, lr2 = torch.randn(2, 2, 6), torch.randn(2, 2, 6)
        with torch.no_grad():
            out1, out2 = model(x, t, lr1), model(x, t, lr2)
        assert not torch.allclose(out1, out2)

    def test_output_depends_on_timestep(self):
        torch.manual_seed(0)
        model = tiny_model().eval()
        x = torch.randn(2, 2, 4, 8)
        lr = torch.randn(2, 2, 6)
        with torch.no_grad():
            out1 = model(x, torch.zeros(2), lr)
            out2 = model(x, torch.ones(2), lr)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# ModelEMA
# ---------------------------------------------------------------------------


class TestModelEMA:
    def test_update_moves_toward_but_not_all_the_way_to_source(self):
        torch.manual_seed(0)
        model = tiny_model()
        ema = ModelEMA(model, decay=0.9)
        before = {n: p.clone() for n, p in ema.ema_model.named_parameters()}

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        src = dict(model.named_parameters())
        for n, p in ema.ema_model.named_parameters():
            moved = (p - before[n]).norm()
            full_gap = (src[n] - before[n]).norm()
            assert 0 < moved.item() < full_gap.item()

    def test_buffers_are_copied_verbatim_not_averaged(self):
        model = tiny_model(varmax=(300.0, 5.0), varmin=(250.0, 0.0))
        ema = ModelEMA(model, decay=0.5)
        model.varmax.fill_(999.0)
        ema.update(model)
        torch.testing.assert_close(ema.ema_model.varmax, model.varmax)

    def test_state_dict_roundtrip(self):
        model = tiny_model()
        ema = ModelEMA(model, decay=0.99)
        ema.update(model)
        sd = ema.state_dict()

        ema2 = ModelEMA(tiny_model(), decay=0.5)
        ema2.load_state_dict(sd)

        assert ema2.num_updates == ema.num_updates
        assert ema2.decay == 0.99
        for (n1, p1), (n2, p2) in zip(
            ema.ema_model.named_parameters(), ema2.ema_model.named_parameters(), strict=True
        ):
            assert n1 == n2
            torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------


FLOW_CLASSES = [FlowMatching, FlowMatchingLinear]


class TestFlowMatching:
    @pytest.mark.parametrize("cls", FLOW_CLASSES)
    def test_sample_xt_endpoints(self, cls):
        fm = cls(hr_hw=(4, 4), channels=2, device="cpu")
        x0 = torch.randn(3, 2, 4, 4)
        x1 = torch.randn(3, 2, 4, 4)
        torch.testing.assert_close(fm.sample_xt(x0, x1, torch.zeros(3)), x0, atol=1e-5, rtol=0)
        torch.testing.assert_close(fm.sample_xt(x0, x1, torch.ones(3)), x1, atol=1e-5, rtol=0)

    @pytest.mark.parametrize("cls", FLOW_CLASSES)
    def test_target_velocity_matches_finite_difference(self, cls):
        fm = cls(hr_hw=(4, 4), channels=1, device="cpu")
        x0 = torch.randn(2, 1, 4, 4)
        x1 = torch.randn(2, 1, 4, 4)
        t = torch.full((2,), 0.37)
        eps = 1e-4
        numerical = (fm.sample_xt(x0, x1, t + eps) - fm.sample_xt(x0, x1, t - eps)) / (2 * eps)
        analytic = fm.target_velocity(x0, x1, t)
        torch.testing.assert_close(analytic, numerical, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("cls", FLOW_CLASSES)
    @pytest.mark.parametrize("method", ["sample", "sample_heun"])
    def test_sampling_produces_finite_output_of_the_right_shape(self, cls, method):
        model = tiny_model(hr_hw=(4, 8), in_ch=2).eval()
        fm = cls(hr_hw=(4, 8), channels=2, device="cpu")
        lr = torch.randn(2, 2, 6)
        with torch.no_grad():
            out = getattr(fm, method)(model, lr, n_steps=3)
        assert out.shape == (2, 2, 4, 8)
        assert torch.isfinite(out).all()
