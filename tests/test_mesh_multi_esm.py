"""Tests for mesh.mesh_multi_esm: ESM-conditioned model, loss weights, per-ESM evaluation."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mesh.mesh_multi_esm import ViTCondDiffusionSRMultiESM, compute_esm_weights, evaluate_per_esm
from mesh.mesh_vit_flow import FlowMatchingLinear


def tiny_model(**overrides) -> ViTCondDiffusionSRMultiESM:
    kwargs = {
        "n_esms": 3, "varmax": (300.0, 5.0), "varmin": (250.0, 0.0),
        "in_ch": 2, "dim": 16, "depth": 2, "heads": 2, "patch": 2, "lr_patch": 2,
        "hr_hw": (4, 8), "lr_regions": 6,
    }  # fmt: skip
    kwargs.update(overrides)
    return ViTCondDiffusionSRMultiESM(**kwargs)


def _inputs(b: int = 2):
    return torch.randn(b, 2, 4, 8), torch.rand(b), torch.randn(b, 2, 6)


def test_forward_shape_and_depends_on_esm():
    torch.manual_seed(0)
    m = tiny_model().eval()
    x, t, lr = _inputs()
    out0 = m(x, t, lr, torch.tensor([0, 0]))
    out1 = m(x, t, lr, torch.tensor([1, 1]))
    assert out0.shape == x.shape
    assert not torch.allclose(out0, out1)  # the embedding must actually reach the output


def test_for_esm_matches_explicit_ids():
    m = tiny_model().eval()
    x, t, lr = _inputs()
    torch.testing.assert_close(m.for_esm(2)(x, t, lr), m(x, t, lr, torch.tensor([2, 2])))


def test_for_esm_works_with_flow_matching_sampler():
    m = tiny_model().eval()
    fm = FlowMatchingLinear(hr_hw=(4, 8), channels=2, device="cpu")
    out = fm.sample(m.for_esm(1), torch.randn(2, 2, 6), n_steps=2)
    assert out.shape == (2, 2, 4, 8)


def _tasks(sizes: dict[str, int], weights: dict[str, float] | None = None):
    weights = weights or {}
    return {e: {"support": range(n), "weight": weights.get(e, 1.0)} for e, n in sizes.items()}


class TestComputeEsmWeights:
    def test_alpha_zero_is_uniform(self):
        w = compute_esm_weights(_tasks({"A": 90, "B": 10}), alpha=0.0)
        assert w["A"] == pytest.approx(1.0) and w["B"] == pytest.approx(1.0)

    def test_alpha_one_equalises_esm_contribution(self):
        sizes = {"A": 90, "B": 10}
        w = compute_esm_weights(_tasks(sizes), alpha=1.0)
        assert w["A"] * sizes["A"] == pytest.approx(w["B"] * sizes["B"])
        assert w["B"] / w["A"] == pytest.approx(9.0)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_sample_averaged_weight_is_one(self, alpha):
        sizes = {"A": 90, "B": 10, "C": 40}
        w = compute_esm_weights(_tasks(sizes, {"C": 3.0}), alpha=alpha)
        assert sum(sizes[e] * w[e] for e in sizes) / sum(sizes.values()) == pytest.approx(1.0)

    def test_task_weight_scales_relative_weight(self):
        w = compute_esm_weights(_tasks({"A": 10, "B": 10}, {"B": 2.0}), alpha=0.5)
        assert w["B"] / w["A"] == pytest.approx(2.0)


def test_evaluate_per_esm_reports_each_esm():
    m = tiny_model()
    fm = FlowMatchingLinear(hr_hw=(4, 8), channels=2, device="cpu")
    loaders = {
        e: DataLoader(
            TensorDataset(torch.randn(4, 2, 6), torch.randn(4, 2, 4, 8), torch.full((4,), i, dtype=torch.long)),
            batch_size=2,
        )
        for i, e in enumerate(["A", "B"])
    }
    out = evaluate_per_esm(m, fm, loaders, torch.device("cpu"))
    assert set(out) == {"A", "B"} and all(v > 0 and v == v for v in out.values())
