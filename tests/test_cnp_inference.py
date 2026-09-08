"""Tests for scales.inference.cnp_inference.

Focus:
  - Checkpoint discovery inside a run directory (final model vs epoch snapshots).
  - Architecture recovery from a CNP state dict — a wrong guess makes
    load_state_dict fail loudly, so this is the highest-value check.
  - The cov_rank=0 default, which pins the forecaster to the diagonal-emission
    SSM the CNP is built on.
  - pr_context being required and validated alongside tas_context.
"""

import os

import numpy as np
import pytest
import torch

from scales.cnp.cnp_ssm import DeepCnpSsmforESM
from scales.inference import CnpForecaster, forecast_from_run_dir
from scales.inference.cnp_inference import (
    DEFAULT_COV_RANK,
    find_checkpoint,
    infer_cnp_config,
)
from scales.model.ssm_model_utils import StandardScaler
from scales.model.ssm_tas_pr import DeepSSMPatternConditioned

Dy, Du = 4, 1
Tc, H = 12, 6
R_DIM, Z_CNP_DIM = 8, 5
ENCODER_HIDDEN = (32, 32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def small_cnp(cov_rank: int = 0, **ssm_overrides) -> DeepCnpSsmforESM:
    kwargs = {
        "y_dim": Dy, "u_dim": Du, "z_dim": 3,
        "rnn_hidden": 6, "u_rnn_hidden": 4, "mlp_hidden": 8,
        "emission_uses_u": True, "use_linear_model": True,
        "reservoir_dim": 2, "cov_rank": cov_rank,
    }
    kwargs.update(ssm_overrides)
    return DeepCnpSsmforESM(
        DeepSSMPatternConditioned(**kwargs),
        r_dim=R_DIM, z_cnp_dim=Z_CNP_DIM, encoder_hidden=ENCODER_HIDDEN,
    )


def write_run_dir(
    tmp_path,
    cov_rank: int = 0,
    epochs: tuple[int, ...] = (),
    final: bool = True,
    scaler_dims: tuple[int, int] | None = None,
    skip_scaler: str | None = None,
) -> str:
    """Lay out a run directory the way build_task_dict and train_cnp do."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)

    state = small_cnp(cov_rank=cov_rank).state_dict()
    if final:
        torch.save(state, run_dir / "cnp_model_out")
    if epochs:
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        for e in epochs:
            torch.save(state, run_dir / "checkpoints" / f"cnp_epoch{e:04d}.pt")

    n_tas, n_gmt = scaler_dims or (Dy, Du)
    for name, dim in (("y_scaler.out", n_tas), ("u_scaler.out", n_gmt),
                      ("pr_scaler.out", n_tas)):
        if name == skip_scaler:
            continue
        StandardScaler().fit(
            rng.standard_normal((2, 20, dim)).astype(np.float32)
        ).save(str(run_dir / name))

    return str(run_dir)


def make_inputs(B: int | None = None, gmt_len: int = Tc + H):
    """GMT series (context + horizon) plus tas and pr contexts, physical units."""
    rng = np.random.default_rng(1)
    shape_u = (gmt_len, Du) if B is None else (B, gmt_len, Du)
    shape_y = (Tc, Dy)      if B is None else (B, Tc, Dy)
    gmt = rng.standard_normal(shape_u).astype(np.float32)
    tas = rng.standard_normal(shape_y).astype(np.float32) * 5.0 + 280.0
    pr  = rng.standard_normal(shape_y).astype(np.float32) * 2.0 + 50.0
    return gmt, tas, pr


@pytest.fixture
def run_dir(tmp_path):
    return write_run_dir(tmp_path)


@pytest.fixture
def forecaster(run_dir):
    # The test model is deliberately tiny, not the 58-region SCALES setup.
    return CnpForecaster.from_run_dir(run_dir, expected_dims=(Dy, Du))


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

class TestFindCheckpoint:
    def test_prefers_final_model(self, tmp_path):
        d = write_run_dir(tmp_path, epochs=(10, 20))
        assert os.path.basename(find_checkpoint(d)) == "cnp_model_out"

    def test_falls_back_to_latest_epoch(self, tmp_path):
        d = write_run_dir(tmp_path, epochs=(10, 30, 20), final=False)
        assert os.path.basename(find_checkpoint(d)) == "cnp_epoch0030.pt"

    def test_explicit_epoch(self, tmp_path):
        d = write_run_dir(tmp_path, epochs=(10, 20))
        assert os.path.basename(find_checkpoint(d, epoch=10)) == "cnp_epoch0010.pt"

    def test_missing_run_dir(self):
        with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
            find_checkpoint("/definitely/not/here")

    def test_no_checkpoint_at_all(self, tmp_path):
        d = write_run_dir(tmp_path, final=False)
        with pytest.raises(FileNotFoundError, match="No CNP checkpoint"):
            find_checkpoint(d)

    def test_unknown_epoch_lists_alternatives(self, tmp_path):
        d = write_run_dir(tmp_path, epochs=(10,))
        with pytest.raises(FileNotFoundError, match=r"available: \['cnp_epoch0010.pt'\]"):
            find_checkpoint(d, epoch=99)


# ---------------------------------------------------------------------------
# Architecture recovery
# ---------------------------------------------------------------------------

class TestInferCnpConfig:
    def test_roundtrip(self):
        cfg = infer_cnp_config(small_cnp().state_dict())
        assert cfg == {
            "r_dim": R_DIM,
            "z_cnp_dim": Z_CNP_DIM,
            "encoder_hidden": ENCODER_HIDDEN,
        }

    @pytest.mark.parametrize("hidden", [(16,), (32, 32), (64, 32, 16)])
    def test_encoder_depth_recovered(self, hidden):
        model = DeepCnpSsmforESM(
            DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, z_dim=3, rnn_hidden=6,
                                      u_rnn_hidden=4, mlp_hidden=8,
                                      emission_uses_u=True, reservoir_dim=2, cov_rank=0),
            r_dim=R_DIM, z_cnp_dim=Z_CNP_DIM, encoder_hidden=hidden,
        )
        assert infer_cnp_config(model.state_dict())["encoder_hidden"] == hidden

    def test_rejects_plain_ssm_checkpoint(self):
        ssm = DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, cov_rank=0)
        with pytest.raises(ValueError, match="does not look like a DeepCnpSsmforESM"):
            infer_cnp_config(ssm.state_dict())


class TestLoading:
    def test_rebuilds_model_strictly(self, forecaster):
        assert isinstance(forecaster.model, DeepCnpSsmforESM)
        assert not forecaster.model.training
        assert forecaster.model.ssm.cov_rank == 0
        assert forecaster.model.z_cnp_dim == Z_CNP_DIM

    def test_missing_scaler_names_the_file(self, tmp_path):
        d = write_run_dir(tmp_path, skip_scaler="pr_scaler.out")
        with pytest.raises(FileNotFoundError, match="pr_scaler.out"):
            CnpForecaster.from_run_dir(d, expected_dims=(Dy, Du))

    def test_scaler_feature_count_must_match(self, tmp_path):
        d = write_run_dir(tmp_path, scaler_dims=(Dy + 1, Du))
        with pytest.raises(AssertionError, match="tas_scaler was fitted on 5 features"):
            CnpForecaster.from_run_dir(d, expected_dims=(Dy, Du))

    def test_region_default_rejects_non_scales_checkpoint(self, run_dir):
        with pytest.raises(AssertionError, match="tas regions"):
            CnpForecaster.from_run_dir(run_dir)

    def test_epoch_snapshot_loads(self, tmp_path):
        d = write_run_dir(tmp_path, epochs=(10,), final=False)
        fc = CnpForecaster.from_run_dir(d, epoch=10, expected_dims=(Dy, Du))
        assert fc.model.ssm.cov_rank == 0


# ---------------------------------------------------------------------------
# cov_rank default
# ---------------------------------------------------------------------------

class TestCovRankDefault:
    def test_default_is_zero(self):
        assert DEFAULT_COV_RANK == 0

    def test_default_rejects_low_rank_checkpoint(self, tmp_path):
        """The CNP is built on the diagonal SSM, so a rank-3 model must be refused."""
        d = write_run_dir(tmp_path, cov_rank=3)
        with pytest.raises(AssertionError, match=r"cov_rank=3, not the 0"):
            CnpForecaster.from_run_dir(d, expected_dims=(Dy, Du))

    def test_none_accepts_any_rank(self, tmp_path):
        d = write_run_dir(tmp_path, cov_rank=3)
        fc = CnpForecaster.from_run_dir(d, expected_dims=(Dy, Du), cov_rank=None)
        assert fc.model.ssm.cov_rank == 3

    def test_explicit_rank_accepts_matching_checkpoint(self, tmp_path):
        d = write_run_dir(tmp_path, cov_rank=3)
        fc = CnpForecaster.from_run_dir(d, expected_dims=(Dy, Du), cov_rank=3)
        assert fc.model.ssm.cov_rank == 3

    def test_low_rank_checkpoint_still_forecasts(self, tmp_path):
        d = write_run_dir(tmp_path, cov_rank=3)
        fc = CnpForecaster.from_run_dir(d, expected_dims=(Dy, Du), cov_rank=None)
        gmt, tas, pr = make_inputs()
        assert fc.forecast(gmt, tas, pr, n_samples=4).tas_mean.shape == (H, Dy)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class TestForecast:
    def test_unbatched_shapes(self, forecaster):
        gmt, tas, pr = make_inputs()
        out = forecaster.forecast(gmt, tas, pr, n_samples=4)
        assert out.context_len == Tc and out.horizon == H
        for arr in (out.tas_mean, out.tas_q10, out.tas_q90,
                    out.pr_mean, out.pr_q10, out.pr_q90):
            assert arr.shape == (H, Dy)
            assert np.isfinite(arr).all()

    def test_batched_shapes(self, forecaster):
        gmt, tas, pr = make_inputs(B=3)
        out = forecaster.forecast(gmt, tas, pr, n_samples=4)
        assert out.tas_mean.shape == (3, H, Dy)

    def test_broadcasts_single_gmt_over_context_batch(self, forecaster):
        gmt, _, _ = make_inputs()
        _, tas, pr = make_inputs(B=3)
        out = forecaster.forecast(gmt[None], tas, pr, n_samples=4)
        assert out.tas_mean.shape == (3, H, Dy)

    @pytest.mark.parametrize("kwargs", [
        {"deterministic": True},
        {"override_esm": True},
        {"deterministic": True, "override_esm": True},
    ])
    def test_rollout_modes(self, forecaster, kwargs):
        gmt, tas, pr = make_inputs()
        out = forecaster.forecast(gmt, tas, pr, n_samples=4, **kwargs)
        assert out.tas_mean.shape == (H, Dy)
        assert np.isfinite(out.tas_mean).all()

    def test_shorter_horizon_truncates(self, forecaster):
        gmt, tas, pr = make_inputs()
        out = forecaster.forecast(gmt, tas, pr, horizon=3, n_samples=4)
        assert out.tas_mean.shape == (3, Dy) and out.horizon == 3

    def test_quantiles_bracket(self, forecaster):
        gmt, tas, pr = make_inputs()
        out = forecaster.forecast(gmt, tas, pr, n_samples=48)
        assert (out.tas_q10 <= out.tas_q90).all()
        assert (out.pr_q10 <= out.pr_q90).all()

    @pytest.mark.parametrize("context_len", [1, 5, 20])
    def test_context_length_is_unconstrained(self, forecaster, context_len):
        rng = np.random.default_rng(4)
        gmt = rng.standard_normal((context_len + 4, Du)).astype(np.float32)
        tas = rng.standard_normal((context_len, Dy)).astype(np.float32)
        pr  = rng.standard_normal((context_len, Dy)).astype(np.float32)
        out = forecaster.forecast(gmt, tas, pr, n_samples=2, deterministic=True)
        assert out.context_len == context_len and out.horizon == 4

    def test_inputs_are_standardised_and_outputs_inverted(self, forecaster):
        """Each context must reach the model through its own scaler."""
        gmt, tas, pr = make_inputs()
        captured = {}
        real = forecaster.model.forecast

        def spy(y_ctx, u_ctx, u_fut, steps, n_samples, pr_ctx=None, override_esm=False):
            captured.update(y_ctx=y_ctx.clone(), u_ctx=u_ctx.clone(),
                            u_fut=u_fut.clone(), pr_ctx=pr_ctx.clone())
            return real(y_ctx, u_ctx, u_fut, steps, n_samples, pr_ctx, override_esm)

        forecaster.model.forecast = spy
        out = forecaster.forecast(gmt, tas, pr, n_samples=4)

        np.testing.assert_allclose(
            captured["y_ctx"].numpy(), forecaster.tas_scaler.transform(tas[None]),
            rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            captured["pr_ctx"].numpy(), forecaster.pr_scaler.transform(pr[None]),
            rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            captured["u_ctx"].numpy(), forecaster.gmt_scaler.transform(gmt[None, :Tc]),
            rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            captured["u_fut"].numpy(), forecaster.gmt_scaler.transform(gmt[None, Tc:]),
            rtol=1e-5, atol=1e-5)

        # tas comes back on the physical scale (inputs were centred near 280).
        assert abs(float(np.mean(
            forecaster.tas_scaler.transform(out.tas_mean[None])))) < 20.0


class TestPrContextValidation:
    def test_pr_context_is_required(self, forecaster):
        gmt, tas, _ = make_inputs()
        with pytest.raises(TypeError, match="pr_context"):
            forecaster.forecast(gmt, tas)

    def test_wrong_region_count(self, forecaster):
        gmt, tas, pr = make_inputs()
        with pytest.raises(AssertionError, match=r"pr_context has 2 pr regions"):
            forecaster.forecast(gmt, tas, pr[:, :2], n_samples=2)

    def test_length_must_match_tas_context(self, forecaster):
        gmt, tas, pr = make_inputs()
        with pytest.raises(AssertionError, match="must have the same shape"):
            forecaster.forecast(gmt, tas, pr[:5], n_samples=2)

    def test_batch_must_match_tas_context(self, forecaster):
        gmt, tas, _ = make_inputs(B=3)
        _, _, pr = make_inputs()
        with pytest.raises(AssertionError, match="must have the same shape"):
            forecaster.forecast(gmt, tas, pr, n_samples=2)

    def test_non_numeric(self, forecaster):
        gmt, tas, _ = make_inputs()
        with pytest.raises(AssertionError, match="must be a numeric array"):
            forecaster.forecast(gmt, tas, np.full((Tc, Dy), "x"), n_samples=2)


class TestForecastValidation:
    def test_gmt_too_short(self, forecaster):
        gmt, tas, pr = make_inputs(gmt_len=Tc)
        with pytest.raises(AssertionError, match="forecast step"):
            forecaster.forecast(gmt, tas, pr, n_samples=2)

    def test_horizon_beyond_gmt(self, forecaster):
        gmt, tas, pr = make_inputs()
        with pytest.raises(AssertionError, match="horizon must be between"):
            forecaster.forecast(gmt, tas, pr, horizon=H + 1, n_samples=2)

    def test_tas_wrong_region_count(self, forecaster):
        gmt, tas, pr = make_inputs()
        with pytest.raises(AssertionError, match=r"tas_context has 2 tas regions"):
            forecaster.forecast(gmt, tas[:, :2], pr[:, :2], n_samples=2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def test_forecast_from_run_dir(run_dir):
    gmt, tas, pr = make_inputs()
    out = forecast_from_run_dir(
        run_dir, gmt, tas, pr, n_samples=3, deterministic=True,
        expected_dims=(Dy, Du),
    )
    assert out.tas_mean.shape == (H, Dy)
