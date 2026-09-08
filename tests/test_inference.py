"""Tests for scales.inference.ssm_inference.

Focus:
  - Architecture inference from a checkpoint must reconstruct the exact model
    that was saved — a wrong guess makes load_state_dict fail loudly, so this
    is the highest-value check.
  - Normalisation must be applied on the way in and inverted on the way out.
  - Context/horizon splitting of the GMT series (off-by-one prone).
"""

import numpy as np
import pytest
import torch

from scales.inference import (
    EXPECTED_DIMS,
    N_GMT_FEATURES,
    N_REGIONS,
    SSMForecaster,
    forecast_from_checkpoint,
)
from scales.inference.ssm_inference import infer_model_config
from scales.model.ssm_model_utils import StandardScaler
from scales.model.ssm_tas_pr import DeepSSMPatternConditioned


Dy, Du = 2, 1
Tc, H = 12, 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_model(**overrides) -> DeepSSMPatternConditioned:
    kwargs = dict(
        y_dim=Dy, u_dim=Du, z_dim=4,
        rnn_hidden=8, u_rnn_hidden=6, mlp_hidden=16,
        emission_uses_u=True, use_linear_model=True,
        reservoir_dim=2, cov_rank=2,
    )
    kwargs.update(overrides)
    return DeepSSMPatternConditioned(**kwargs)


@pytest.fixture
def artefacts(tmp_path):
    """Write a checkpoint plus y/u/pr scalers to tmp_path and return the paths."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    torch.save(small_model().state_dict(), tmp_path / "model.pt")

    # Deliberately non-trivial offsets/scales so a missing transform shows up.
    scalers = {
        "y":  rng.standard_normal((4, 30, Dy)).astype(np.float32) * 5.0 + 280.0,
        "u":  rng.standard_normal((4, 30, Du)).astype(np.float32) * 0.5 + 1.5,
        "pr": rng.standard_normal((4, 30, Dy)).astype(np.float32) * 2.0 + 50.0,
    }
    for name, data in scalers.items():
        StandardScaler().fit(data).save(str(tmp_path / f"{name}_scaler.out"))

    return {
        "checkpoint_path": str(tmp_path / "model.pt"),
        "tas_scaler_path": str(tmp_path / "y_scaler.out"),
        "gmt_scaler_path": str(tmp_path / "u_scaler.out"),
        "pr_scaler_path":  str(tmp_path / "pr_scaler.out"),
        # The test model is deliberately tiny, not the 58-region SCALES setup.
        "expected_dims":   (Dy, Du),
    }


def make_inputs(B: int | None = None, gmt_len: int = Tc + H):
    """GMT series (context + horizon) and a tas context, in physical units."""
    rng = np.random.default_rng(1)
    shape_u = (gmt_len, Du) if B is None else (B, gmt_len, Du)
    shape_y = (Tc, Dy)      if B is None else (B, Tc, Dy)
    gmt = rng.standard_normal(shape_u).astype(np.float32) * 0.5 + 1.5
    tas = rng.standard_normal(shape_y).astype(np.float32) * 5.0 + 280.0
    return gmt, tas


# ---------------------------------------------------------------------------
# Architecture inference
# ---------------------------------------------------------------------------

class TestInferModelConfig:
    def test_roundtrip_full_model(self):
        cfg = infer_model_config(small_model().state_dict())
        assert cfg == {
            "y_dim": Dy, "u_dim": Du, "z_dim": 4,
            "rnn_hidden": 8, "u_rnn_hidden": 6, "mlp_hidden": 16,
            "cov_rank": 2, "reservoir_dim": 2,
            "emission_uses_u": True, "use_linear_model": True,
        }

    @pytest.mark.parametrize("emission_uses_u", [True, False])
    @pytest.mark.parametrize("use_linear_model", [True, False])
    def test_inferred_config_reloads_state_dict(self, emission_uses_u, use_linear_model):
        """The inferred config must accept the saved weights strictly."""
        sd = small_model(emission_uses_u=emission_uses_u,
                         use_linear_model=use_linear_model).state_dict()
        rebuilt = DeepSSMPatternConditioned(**infer_model_config(sd))
        rebuilt.load_state_dict(sd, strict=True)   # raises on any mismatch

    @pytest.mark.parametrize("cov_rank", [0, 2, 5])
    def test_cov_rank_recovered_from_emit_head(self, cov_rank):
        """A diagonal (cov_rank=0) checkpoint must be recognised as such."""
        sd  = small_model(cov_rank=cov_rank).state_dict()
        cfg = infer_model_config(sd)
        assert cfg["cov_rank"] == cov_rank
        DeepSSMPatternConditioned(**cfg).load_state_dict(sd, strict=True)

    def test_rejects_foreign_checkpoint(self):
        with pytest.raises(ValueError, match="missing the expected parameter"):
            infer_model_config({"foo.weight": torch.zeros(3, 3)})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestFromCheckpoint:
    def test_loads_and_is_eval_mode(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        assert isinstance(fc.model, DeepSSMPatternConditioned)
        assert not fc.model.training
        assert fc.model.y_dim == Dy and fc.model.u_dim == Du

    def test_ddp_prefixed_checkpoint(self, artefacts, tmp_path):
        sd = {f"module.{k}": v for k, v in small_model().state_dict().items()}
        path = tmp_path / "ddp.pt"
        torch.save(sd, path)
        SSMForecaster.from_checkpoint(**{**artefacts, "checkpoint_path": str(path)})

    def test_wrapped_checkpoint(self, artefacts, tmp_path):
        path = tmp_path / "wrapped.pt"
        torch.save({"epoch": 7, "state_dict": small_model().state_dict()}, path)
        SSMForecaster.from_checkpoint(**{**artefacts, "checkpoint_path": str(path)})

    def test_missing_checkpoint(self, artefacts):
        with pytest.raises(FileNotFoundError):
            SSMForecaster.from_checkpoint(**{**artefacts, "checkpoint_path": "/nope.pt"})


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class TestForecast:
    def test_unbatched_shapes(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        out = fc.forecast(gmt, tas, n_samples=4)
        assert out.context_len == Tc and out.horizon == H
        for arr in (out.tas_mean, out.tas_q10, out.tas_q90,
                    out.pr_mean, out.pr_q10, out.pr_q90):
            assert arr.shape == (H, Dy)
            assert np.isfinite(arr).all()

    def test_batched_shapes(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs(B=3)
        out = fc.forecast(gmt, tas, n_samples=4)
        assert out.tas_mean.shape == (3, H, Dy)
        assert out.pr_q90.shape == (3, H, Dy)

    def test_1d_gmt_accepted_when_u_dim_is_one(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        out_1d = fc.forecast(gmt[:, 0], tas, n_samples=4, deterministic=True)
        out_2d = fc.forecast(gmt,       tas, n_samples=4, deterministic=True)
        assert out_1d.tas_mean.shape == out_2d.tas_mean.shape

    def test_quantiles_bracket_mean(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        out = fc.forecast(gmt, tas, n_samples=64)
        assert (out.tas_q10 <= out.tas_q90).all()
        assert (out.pr_q10 <= out.pr_q90).all()

    def test_shorter_horizon_truncates(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        out = fc.forecast(gmt, tas, horizon=3, n_samples=4)
        assert out.tas_mean.shape == (3, Dy) and out.horizon == 3

    def test_broadcasts_single_gmt_over_context_batch(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, _ = make_inputs()
        _, tas = make_inputs(B=3)
        out = fc.forecast(gmt[None], tas, n_samples=4)
        assert out.tas_mean.shape == (3, H, Dy)

    def test_outputs_are_in_physical_units(self, artefacts):
        """Inverting the tas scaler must undo the standardisation exactly."""
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()

        captured = {}
        real_forecast = fc.model.forecast

        def spy(y_ctx, u_ctx, u_fut, steps, n_samples):
            captured["y_ctx"] = y_ctx.clone()
            captured["u_ctx"] = u_ctx.clone()
            captured["u_fut"] = u_fut.clone()
            return real_forecast(y_ctx, u_ctx, u_fut, steps=steps, n_samples=n_samples)

        fc.model.forecast = spy
        out = fc.forecast(gmt, tas, n_samples=4)

        # Context reaches the model standardised ...
        np.testing.assert_allclose(
            captured["y_ctx"].numpy(), fc.tas_scaler.transform(tas[None]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            captured["u_ctx"].numpy(), fc.gmt_scaler.transform(gmt[None, :Tc]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            captured["u_fut"].numpy(), fc.gmt_scaler.transform(gmt[None, Tc:]), rtol=1e-5, atol=1e-5)

        # ... and the tas forecast comes back on the physical scale.
        tas_mean = float(np.mean(fc.tas_scaler.transform(out.tas_mean[None])))
        assert abs(tas_mean) < 20.0, "output looks like it was never inverse-transformed"


class TestForecastValidation:
    def test_gmt_too_short_for_context(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs(gmt_len=Tc)
        with pytest.raises(AssertionError, match="forecast step"):
            fc.forecast(gmt, tas, n_samples=2)

    def test_horizon_beyond_gmt(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        with pytest.raises(AssertionError, match="horizon must be between"):
            fc.forecast(gmt, tas, horizon=H + 1, n_samples=2)

    def test_wrong_feature_count(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        with pytest.raises(AssertionError, match="expects 2"):
            fc.forecast(gmt, tas[..., :1], n_samples=2)

    def test_incompatible_batch_sizes(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, _ = make_inputs(B=2)
        _, tas = make_inputs(B=3)
        with pytest.raises(AssertionError, match="incompatible"):
            fc.forecast(gmt, tas, n_samples=2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def test_forecast_from_checkpoint(artefacts):
    gmt, tas = make_inputs()
    out = forecast_from_checkpoint(
        gmt=gmt, tas_context=tas, n_samples=4, deterministic=True, **artefacts)
    assert out.tas_mean.shape == (H, Dy)


# ---------------------------------------------------------------------------
# Dimension asserts
# ---------------------------------------------------------------------------

class TestExpectedDims:
    def test_scales_defaults(self):
        assert (N_REGIONS, N_GMT_FEATURES) == EXPECTED_DIMS == (58, 1)

    def test_default_rejects_non_scales_checkpoint(self, artefacts):
        """The tiny test model must be refused under the 58-region default."""
        del artefacts["expected_dims"]
        with pytest.raises(AssertionError, match="tas regions"):
            SSMForecaster.from_checkpoint(**artefacts)

    def test_default_accepts_58_region_model(self, tmp_path):
        """A model with the SCALES dimensions loads without an override."""
        rng = np.random.default_rng(2)
        model = small_model(y_dim=N_REGIONS, u_dim=N_GMT_FEATURES)
        torch.save(model.state_dict(), tmp_path / "model.pt")

        paths = {"checkpoint_path": str(tmp_path / "model.pt")}
        for key, dim in (("tas", N_REGIONS), ("gmt", N_GMT_FEATURES), ("pr", N_REGIONS)):
            path = tmp_path / f"{key}_scaler.out"
            StandardScaler().fit(
                rng.standard_normal((2, 20, dim)).astype(np.float32)).save(str(path))
            paths[f"{key}_scaler_path"] = str(path)

        fc = SSMForecaster.from_checkpoint(**paths)
        assert fc.model.y_dim == N_REGIONS and fc.model.u_dim == N_GMT_FEATURES

        out = fc.forecast(
            rng.standard_normal((Tc + H, N_GMT_FEATURES)).astype(np.float32),
            rng.standard_normal((Tc, N_REGIONS)).astype(np.float32),
            n_samples=2, deterministic=True,
        )
        assert out.tas_mean.shape == (H, N_REGIONS)

    def test_gmt_dim_mismatch_is_reported(self, artefacts):
        with pytest.raises(AssertionError, match="gmt features"):
            SSMForecaster.from_checkpoint(**{**artefacts, "expected_dims": (Dy, Du + 1)})

    def test_scaler_feature_count_must_match_model(self, artefacts, tmp_path):
        """A scaler fitted on the wrong number of regions must be caught."""
        rng = np.random.default_rng(3)
        wrong = tmp_path / "wrong_scaler.out"
        StandardScaler().fit(
            rng.standard_normal((2, 20, Dy + 1)).astype(np.float32)).save(str(wrong))
        with pytest.raises(AssertionError, match="tas_scaler was fitted on 3 features"):
            SSMForecaster.from_checkpoint(**{**artefacts, "tas_scaler_path": str(wrong)})


class TestInputShapeAsserts:
    def test_tas_context_wrong_region_count(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        with pytest.raises(AssertionError, match=r"tas_context has 1 tas regions"):
            fc.forecast(gmt, tas[:, :1], n_samples=2)

    def test_gmt_wrong_feature_count(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        with pytest.raises(AssertionError, match=r"gmt has 2 gmt features"):
            fc.forecast(np.repeat(gmt, 2, axis=-1), tas, n_samples=2)

    def test_rank_too_high(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        with pytest.raises(AssertionError, match="must be 1-, 2- or 3-D"):
            fc.forecast(gmt, tas[None, None], n_samples=2)

    def test_1d_tas_context_rejected(self, artefacts):
        """tas is multi-region, so a bare [T] series is ambiguous."""
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        with pytest.raises(AssertionError, match=r"tas_context is 1-D"):
            fc.forecast(gmt, tas[:, 0], n_samples=2)

    def test_non_numeric_input(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, tas = make_inputs()
        with pytest.raises(AssertionError, match="must be a numeric array"):
            fc.forecast(gmt, np.full((Tc, Dy), "x"), n_samples=2)

    def test_empty_time_axis(self, artefacts):
        fc = SSMForecaster.from_checkpoint(**artefacts)
        gmt, _ = make_inputs()
        with pytest.raises(AssertionError, match="empty time axis"):
            fc.forecast(gmt, np.zeros((0, Dy), dtype=np.float32), n_samples=2)

    @pytest.mark.parametrize("context_len", [1, 5, 20])
    def test_context_length_is_unconstrained(self, artefacts, context_len):
        """Only the feature axis is fixed; any context length must work."""
        fc = SSMForecaster.from_checkpoint(**artefacts)
        rng = np.random.default_rng(4)
        gmt = rng.standard_normal((context_len + 4, Du)).astype(np.float32)
        tas = rng.standard_normal((context_len, Dy)).astype(np.float32)
        out = fc.forecast(gmt, tas, n_samples=2, deterministic=True)
        assert out.context_len == context_len and out.horizon == 4


def test_diagonal_checkpoint_forecasts_end_to_end(tmp_path):
    """A cov_rank=0 checkpoint must load and forecast with no extra configuration."""
    rng = np.random.default_rng(5)
    torch.manual_seed(3)

    torch.save(small_model(cov_rank=0).state_dict(), tmp_path / "model.pt")
    paths = {"checkpoint_path": str(tmp_path / "model.pt"), "expected_dims": (Dy, Du)}
    for key, dim in (("tas", Dy), ("gmt", Du), ("pr", Dy)):
        path = tmp_path / f"{key}_scaler.out"
        StandardScaler().fit(
            rng.standard_normal((3, 25, dim)).astype(np.float32)).save(str(path))
        paths[f"{key}_scaler_path"] = str(path)

    fc = SSMForecaster.from_checkpoint(**paths)
    assert fc.model.cov_rank == 0

    gmt, tas = make_inputs()
    out = fc.forecast(gmt, tas, n_samples=8)
    assert out.tas_mean.shape == (H, Dy)
    assert np.isfinite(out.tas_mean).all()
    assert (out.tas_q10 <= out.tas_q90).all()
