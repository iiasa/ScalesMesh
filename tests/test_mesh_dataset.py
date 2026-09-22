"""Tests for mesh.mesh_dataset: filename pairing, region stats, and the multi-var dataset.

Focus:
  - make_key/get_timerange pairing (silently mismatching tas/pr files means
    training on misaligned data — the most dangerous failure mode here).
  - NetCDFStatsPrepper's min/max and area-weighted region means, checked
    against ground-truth arrays rather than hand-computed constants.
  - NetCDFTimeDataset's index math (pair/time lookup) and that the returned
    HR/LR tensors are actually the right slice of the right file.

Uses a small synthetic two-region mask (monkeypatched in, so no network fetch
of the real IPCC AR6 shapefiles) and small NetCDF files written to tmp_path,
exercised through the real xarray/regionmask pipeline.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import regionmask
import torch
import xarray as xr
from shapely.geometry import box

from mesh.mesh_dataset import NetCDFStatsPrepper, NetCDFTimeDataset, get_timerange, make_key, normalize

LAT = np.array([-60.0, -20.0, 20.0, 60.0])
LON = np.array([45.0, 135.0, 225.0, 315.0])  # indices 0,1 < 180 (West); 2,3 >= 180 (East)
N_REGIONS = 2
T = 3  # time steps per file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def two_region_mask(monkeypatch):
    """Swap the real (network-fetched) AR6 regions for a tiny synthetic 2-region set."""
    polys = [box(0, -90, 180, 90), box(180, -90, 360, 90)]
    fake = regionmask.Regions(polys, names=["West", "East"], abbrevs=["W", "E"], name="test")
    monkeypatch.setattr(regionmask.defined_regions, "ar6", types.SimpleNamespace(all=fake))
    return fake


def _write_file(path: Path, var: str, data: np.ndarray) -> None:
    """data: (T, lat, lon)."""
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data.astype("float32"))},
        coords={"time": np.arange(data.shape[0]), "lat": LAT, "lon": LON},
    )
    ds.to_netcdf(path, engine="netcdf4")


@pytest.fixture
def tas_pr_files(tmp_path):
    """Two paired tas/pr files (T time steps each), with the raw arrays kept for verification."""
    rng = np.random.default_rng(0)
    tas_data = [rng.uniform(250, 300, size=(T, 4, 4)) for _ in range(2)]
    pr_data = [rng.uniform(0, 5, size=(T, 4, 4)) for _ in range(2)]

    tas_files, pr_files = [], []
    for i, data in enumerate(tas_data):
        p = tmp_path / f"tas_Amon_MODEL_hist_r1i1p1f1_gn_{195001 + i * 100:06d}-{195012 + i * 100:06d}.nc"
        _write_file(p, "tas", data)
        tas_files.append(str(p))
    for i, data in enumerate(pr_data):
        p = tmp_path / f"pr_Amon_MODEL_hist_r1i1p1f1_gn_{195001 + i * 100:06d}-{195012 + i * 100:06d}.nc"
        _write_file(p, "pr", data)
        pr_files.append(str(p))

    return {"tas_files": tas_files, "pr_files": pr_files, "tas_data": tas_data, "pr_data": pr_data}


def _build_dataset(files, hr_scale: int = 1, transform=None):
    tas_prepper = NetCDFStatsPrepper(files["tas_files"], var_name="tas")
    pr_prepper = NetCDFStatsPrepper(files["pr_files"], var_name="pr")
    ds = NetCDFTimeDataset(
        vars_files={"tas": files["tas_files"], "pr": files["pr_files"]},
        region_data={"tas": tas_prepper.get_region_means(), "pr": pr_prepper.get_region_means()},
        hr_scale=hr_scale,
        transform=transform,
    )
    return ds, tas_prepper, pr_prepper


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


class TestFilenameHelpers:
    def test_get_timerange_extracts_token(self):
        p = Path("tas_Amon_MODEL_r1i1p1f1_gn_195001-201412.nc")
        assert get_timerange(p) == "195001-201412"

    def test_get_timerange_missing_token(self):
        assert get_timerange(Path("no_timerange_here.nc")) == ""

    def test_make_key_matches_tas_and_pr(self):
        tas = Path("tas_Amon_MODEL_r1i1p1f1_gn_195001-201412.nc")
        pr = Path("pr_Amon_MODEL_r1i1p1f1_gn_195001-201412.nc")
        assert make_key(tas) == make_key(pr)

    def test_make_key_differs_for_different_members(self):
        a = Path("tas_Amon_MODEL_r1i1p1f1_gn_195001-201412.nc")
        b = Path("tas_Amon_MODEL_r2i1p1f1_gn_195001-201412.nc")
        assert make_key(a) != make_key(b)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_rescales_to_minus_one_one(self):
        x = torch.tensor([0.0, 50.0, 100.0])
        y = normalize(x, vmin=0.0, vmax=100.0)
        torch.testing.assert_close(y, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# NetCDFStatsPrepper
# ---------------------------------------------------------------------------


class TestNetCDFStatsPrepper:
    def test_min_max_match_raw_data(self, tas_pr_files):
        prepper = NetCDFStatsPrepper(tas_pr_files["tas_files"], var_name="tas")
        vmax, vmin = prepper.get_ds_max_min()
        all_vals = np.concatenate([d.ravel() for d in tas_pr_files["tas_data"]])
        assert vmax == pytest.approx(all_vals.max(), rel=1e-5)
        assert vmin == pytest.approx(all_vals.min(), rel=1e-5)

    def test_n_regions(self, tas_pr_files):
        prepper = NetCDFStatsPrepper(tas_pr_files["tas_files"], var_name="tas")
        assert prepper.get_n_regions() == N_REGIONS

    def test_region_means_shape_is_time_by_region(self, tas_pr_files):
        prepper = NetCDFStatsPrepper(tas_pr_files["tas_files"], var_name="tas")
        means = prepper.get_region_means()
        assert len(means) == 2  # one array per file
        for m in means:
            assert m.shape == (T, N_REGIONS)

    def test_region_means_are_area_weighted_averages(self, tas_pr_files):
        """Each region's mean must fall within the min/max of the raw cells in that region."""
        prepper = NetCDFStatsPrepper(tas_pr_files["tas_files"], var_name="tas")
        means = prepper.get_region_means()
        for file_idx, data in enumerate(tas_pr_files["tas_data"]):
            for t in range(T):
                west, east = data[t, :, :2], data[t, :, 2:]
                assert west.min() <= means[file_idx][t, 0] <= west.max()
                assert east.min() <= means[file_idx][t, 1] <= east.max()

    def test_plottable_mask_scatters_region_values(self, tas_pr_files):
        prepper = NetCDFStatsPrepper(tas_pr_files["tas_files"], var_name="tas")
        plottable = prepper.get_plottable_mask(np.array([1.0, 2.0]))
        vals = plottable.values
        assert np.all(vals[:, :2] == 1.0)  # West
        assert np.all(vals[:, 2:] == 2.0)  # East


# ---------------------------------------------------------------------------
# NetCDFTimeDataset
# ---------------------------------------------------------------------------


class TestNetCDFTimeDataset:
    def test_length_is_total_time_steps_across_files(self, tas_pr_files):
        ds, *_ = _build_dataset(tas_pr_files)
        assert len(ds) == 2 * T

    def test_item_shapes(self, tas_pr_files):
        ds, *_ = _build_dataset(tas_pr_files)
        lr, hr = ds[0]
        assert lr.shape == (2, N_REGIONS)  # 2 vars stacked as channels
        assert hr.shape == (2, 4, 4)  # 2 vars stacked, full resolution

    def test_hr_scale_downsamples(self, tas_pr_files):
        ds, *_ = _build_dataset(tas_pr_files, hr_scale=2)
        _, hr = ds[0]
        assert hr.shape == (2, 2, 2)

    def test_hr_matches_raw_file_data(self, tas_pr_files):
        """hr_scale=1 resizes to the same shape, which is an exact identity, so HR must equal the raw data."""
        ds, *_ = _build_dataset(tas_pr_files)
        for idx in (0, 2, 3, 5):
            file_idx, t_local = idx // T, idx % T
            _, hr = ds[idx]
            np.testing.assert_allclose(hr[0].numpy(), tas_pr_files["tas_data"][file_idx][t_local], atol=1e-4)
            np.testing.assert_allclose(hr[1].numpy(), tas_pr_files["pr_data"][file_idx][t_local], atol=1e-4)

    def test_lr_matches_region_means(self, tas_pr_files):
        ds, tas_prepper, pr_prepper = _build_dataset(tas_pr_files)
        idx = 4
        file_idx, t_local = idx // T, idx % T
        lr, _ = ds[idx]
        np.testing.assert_allclose(lr[0].numpy(), tas_prepper.get_region_means()[file_idx][t_local], atol=1e-4)
        np.testing.assert_allclose(lr[1].numpy(), pr_prepper.get_region_means()[file_idx][t_local], atol=1e-4)

    def test_negative_index_matches_positive_equivalent(self, tas_pr_files):
        ds, *_ = _build_dataset(tas_pr_files)
        lr_last, hr_last = ds[len(ds) - 1]
        lr_neg, hr_neg = ds[-1]
        torch.testing.assert_close(hr_last, hr_neg)
        torch.testing.assert_close(lr_last, lr_neg)

    def test_transform_applied_per_variable(self, tas_pr_files):
        tas_prepper = NetCDFStatsPrepper(tas_pr_files["tas_files"], var_name="tas")
        pr_prepper = NetCDFStatsPrepper(tas_pr_files["pr_files"], var_name="pr")
        vmax_tas, vmin_tas = tas_prepper.get_ds_max_min()
        vmax_pr, vmin_pr = pr_prepper.get_ds_max_min()
        ds, *_ = _build_dataset(
            tas_pr_files,
            transform={
                "tas": lambda x: normalize(x, vmin_tas, vmax_tas),
                "pr": lambda x: normalize(x, vmin_pr, vmax_pr),
            },
        )
        _, hr = ds[0]
        assert hr.min() >= -1.0 - 1e-4
        assert hr.max() <= 1.0 + 1e-4

    def test_empty_vars_files_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            NetCDFTimeDataset(vars_files={}, region_data={})

    def test_mismatched_file_counts_raises(self, tas_pr_files):
        with pytest.raises(ValueError, match="same number of files"):
            NetCDFTimeDataset(
                vars_files={"tas": tas_pr_files["tas_files"], "pr": tas_pr_files["pr_files"][:1]},
                region_data={"tas": [], "pr": []},
            )
