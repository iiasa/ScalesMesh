"""NetCDF tas/pr data loading utilities for MESH.

Pairs monthly ``tas``/``pr`` CMIP-style files by ensemble member and time
range (via :func:`make_key`/:func:`get_timerange`), computes IPCC AR6
region-weighted statistics for normalization (:class:`NetCDFStatsPrepper`),
and exposes a multi-variable, multi-file, time-indexed dataset
(:class:`NetCDFTimeDataset`) that yields (region-mean, HR field) pairs
stacked across variables into channels.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import regionmask
import torch
import torchvision.transforms.functional as TF
import xarray as xr
from torch.utils.data import Dataset
from torchvision import transforms

TIME_RE = re.compile(r"_(\d{6}-\d{6})\.nc$")


def normalize(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """Rescale ``x`` from ``[vmin, vmax]`` to ``[-1, 1]``."""
    return (x - vmin) / (vmax - vmin + 1e-8) * 2.0 - 1.0


def denormalize(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """Invert :func:`normalize`: rescale ``x`` from ``[-1, 1]`` back to ``[vmin, vmax]``."""
    return (x + 1.0) / 2.0 * (vmax - vmin + 1e-8) + vmin


def make_key(path: Path) -> str:
    """Key shared by matching tas/pr files: strip the variable prefix and time range."""
    name = TIME_RE.sub(".nc", path.name)  # drop trailing _YYYYMM-YYYYMM
    return re.sub(r"^(tas|pr)_", "", name)  # drop leading var_


def get_timerange(path: Path) -> str:
    """Extract the trailing ``YYYYMM-YYYYMM`` time-range token from a filename."""
    m = TIME_RE.search(path.name)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Region-weighted stats
# ---------------------------------------------------------------------------


class NetCDFStatsPrepper:
    """AR6 region-weighted means and the global min/max for one variable across files."""

    def __init__(self, file_list: list[str], var_name: str = "tas"):
        self.file_list = file_list
        regions = regionmask.defined_regions.ar6.all
        self.regions_dim = len(regions.names)

        all_mins, all_maxs, region_means = [], [], []
        for path in file_list:
            ds = xr.open_dataset(path, engine="netcdf4", decode_cf=True, chunks={"time": 128})
            da = ds[var_name].astype("float64")

            all_mins.append(da.min(skipna=True).compute().item())
            all_maxs.append(da.max(skipna=True).compute().item())

            mask_3d = regions.mask_3D(ds["lon"], ds["lat"])
            self.mask = regions.mask(ds["lon"], ds["lat"])
            lat_weight = np.cos(np.deg2rad(ds["lat"])).astype("float64")
            weight = mask_3d * lat_weight  # area-weighted per region
            num = (da * weight).sum(("lat", "lon"), skipna=True)
            den = weight.sum(("lat", "lon"))
            region_means.append((num / den).values)  # (time, region) — da's leading dim wins the broadcast

        self.file_mins = all_mins  # per-file extrema, aligned with file_list
        self.file_maxs = all_maxs
        self._min = min(all_mins)
        self._max = max(all_maxs)
        self.region_means = region_means

    def get_region_means(self) -> list[np.ndarray]:
        return self.region_means

    def get_n_regions(self) -> int:
        return self.regions_dim

    def get_ds_max_min(self) -> tuple[float, float]:
        return self._max, self._min

    def get_plottable_mask(self, data: np.ndarray) -> xr.DataArray:
        """Scatter per-region values in ``data`` back onto the lat/lon region mask."""
        mask_values = self.mask.values
        data_mask = np.empty(mask_values.shape)
        for region in range(len(data)):
            data_mask[mask_values == region] = data[region]

        plottable = self.mask.copy(data=data_mask)
        plottable.attrs["standard_name"] = "tas"
        return plottable


# ---------------------------------------------------------------------------
# Multi-variable, multi-file, time-indexed dataset
# ---------------------------------------------------------------------------


class NetCDFTimeDataset(Dataset):
    """Yields (LR region means, HR field) pairs, stacked across variables into channels.

    ``vars_files`` maps variable name -> list of filepaths, e.g.::

        {"tas": [tas_file_0, tas_file_1, ...], "pr": [pr_file_0, pr_file_1, ...]}

    All lists must be the same length and aligned by "pair index": pair ``i`` is the
    same ensemble member / experiment / time-range chunk across variables.
    """

    def __init__(
        self,
        vars_files: dict[str, list[str]],
        region_data: dict[str, list[np.ndarray]],
        hr_scale: int = 1,
        time_dim: str = "time",
        sel_spatial: dict[str, slice | int] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        dtype: torch.dtype = torch.float32,
        engine: str = "netcdf4",
        decode_cf: bool = True,
        chunks: dict[str, int] | None = None,  # e.g. {"time": 32, "lat": 256, "lon": 256}
        drop_variables: str | list[str] | None = None,
        cache_size: int = 8,  # per-worker, per-variable open-file cache
        preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
        esm_id: int | None = None,  # if set, __getitem__ also returns it as a long tensor
    ):
        super().__init__()
        if not vars_files:
            raise ValueError("vars_files must be a non-empty dict {var: [files...]}")

        self.vars = list(vars_files.keys())
        self.vars_files = {v: list(paths) for v, paths in vars_files.items()}

        nfiles = {v: len(paths) for v, paths in self.vars_files.items()}
        if len(set(nfiles.values())) != 1:
            raise ValueError(f"All variables must have the same number of files. Got: {nfiles}")
        self.npairs = next(iter(nfiles.values()))

        self.region_data = region_data
        self.hr_scale = int(hr_scale)
        self.time_dim = time_dim
        self.sel_spatial = sel_spatial or {}
        self.transform = transform
        self.dtype = dtype
        self.engine = engine
        self.decode_cf = decode_cf
        self.chunks = chunks
        self.drop_variables = drop_variables
        self.preprocess = preprocess
        self.esm_id = esm_id

        # Global time index, built from the first variable (assumes all variables
        # share the same time length within each paired file).
        ref_var = self.vars[0]
        self._cum = [0]
        spatial_dims = []
        for pair_idx in range(self.npairs):
            fp = self.vars_files[ref_var][pair_idx]
            with xr.open_dataset(
                fp,
                engine=self.engine,
                decode_cf=self.decode_cf,
                chunks={self.time_dim: 1} if self.chunks is None else self.chunks,
                drop_variables=self.drop_variables,
            ) as ds:
                if ref_var not in ds:
                    raise KeyError(f"{ref_var} not in {fp}")
                self._cum.append(self._cum[-1] + int(ds.sizes[self.time_dim]))
                spatial_dims.append((int(ds.sizes["lat"]), int(ds.sizes["lon"])))

        self.N = self._cum[-1]
        lat_size, lon_size = spatial_dims[-1]
        self.hr_size = (lat_size // self.hr_scale, lon_size // self.hr_scale)

        # Per-variable cached opener, so tas/pr open caches don't collide.
        self._open = {v: self._make_file_cache(cache_size) for v in self.vars}

    def __len__(self) -> int:
        return self.N

    def _make_file_cache(self, cache_size: int) -> Callable[[str], xr.Dataset]:
        @lru_cache(maxsize=cache_size)
        def _lazy_open(path: str) -> xr.Dataset:
            ds = xr.open_dataset(
                path,
                engine=self.engine,
                decode_cf=self.decode_cf,
                chunks=self.chunks,
                drop_variables=self.drop_variables,
            )
            return self.preprocess(ds) if self.preprocess else ds

        return _lazy_open

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map a global time index to (pair_index, local_time_index)."""
        if idx < 0:
            idx += self.N
        pair_idx = bisect.bisect_right(self._cum, idx) - 1
        return pair_idx, idx - self._cum[pair_idx]

    def _select_da(self, ds: xr.Dataset, var: str, t_local: int) -> xr.DataArray:
        isel_spatial, sel_spatial = {}, {}
        for k, v in self.sel_spatial.items():
            if isinstance(v, int | np.integer):
                isel_spatial[k] = int(v)
            else:
                sel_spatial[k] = v

        da = ds[var]
        if sel_spatial:
            da = da.sel(**sel_spatial)
        if isel_spatial:
            da = da.isel(**isel_spatial)
        return da.isel({self.time_dim: t_local})

    def _hr_tensor(self, var: str, pair_idx: int, t_local: int) -> torch.Tensor:
        ds = self._open[var](self.vars_files[var][pair_idx])
        da = self._select_da(ds, var, t_local).astype(np.float32)
        arr = da.data
        np_arr = np.asarray(arr.compute() if hasattr(arr, "compute") else arr)  # [H, W]
        t = torch.from_numpy(np_arr).to(self.dtype).unsqueeze(0)  # [1, H, W]
        if self.transform is not None and self.transform.get(var) is not None:
            t = self.transform[var](t)
        return t

    def _lr_tensor(self, var: str, pair_idx: int, t_local: int) -> torch.Tensor:
        lr = self.region_data[var][pair_idx][t_local]
        t = torch.from_numpy(np.expand_dims(lr, 0)).to(self.dtype)
        if self.transform is not None and self.transform.get(var) is not None:
            t = self.transform[var](t)
        return t

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        pair_idx, t_local = self._locate(idx)

        hr = torch.cat([self._hr_tensor(v, pair_idx, t_local) for v in self.vars], dim=0)
        lr = torch.cat([self._lr_tensor(v, pair_idx, t_local) for v in self.vars], dim=0)

        hr_scaled = TF.resize(hr, self.hr_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if self.esm_id is not None:
            return lr, hr_scaled, torch.tensor(self.esm_id, dtype=torch.long)
        return lr, hr_scaled


# ---------------------------------------------------------------------------
# Convenience: discover, pair, and load a tas/pr dataset from a data directory
# ---------------------------------------------------------------------------

# Defaults this project was developed against (ACCESS-ESM1-5); pass others per call.
DEFAULT_ESM = "ACCESS-ESM1-5"
DEFAULT_EXPERIMENTS = ("ssp585", "historical", "ssp3", "ssp534")
DEFAULT_EXCLUDES = ("ssp534-over_r3",)


def discover_experiment_files(
    data_path: str | Path,
    var: str,
    esm_name: str = DEFAULT_ESM,
    experiments: Sequence[str] = DEFAULT_EXPERIMENTS,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> list[str]:
    """List ``var`` NetCDF files of ``esm_name`` for the given experiments.

    A file ``{var}_Amon_{esm_name}_...`` is kept if its name contains
    ``{var}_Amon_{esm_name}_{experiment}`` for any entry of ``experiments`` (a prefix
    match, so ``"ssp3"`` also matches ``ssp370``) and contains none of the ``excludes``
    substrings. Experiments with no matching file are reported on stdout and skipped,
    so an ESM lacking some experiment does not abort the run.
    """
    prefix = f"{var}_Amon_{esm_name}_"
    candidates = sorted(f for f in Path(data_path).iterdir() if f.is_file() and not any(x in f.name for x in excludes))

    files: list[str] = []
    for exp in experiments:
        matched = [f for f in candidates if f"{prefix}{exp}" in f.name]
        if not matched:
            print(f"[mesh_dataset] {esm_name}: no '{var}' files for experiment '{exp}' in {data_path}; skipping")
        files.extend(str(f.absolute()) for f in matched if str(f.absolute()) not in files)
    return files


def pair_tas_pr_files(tas_files: list[str], pr_files: list[str]) -> tuple[list[str], list[str]]:
    """Pair tas/pr files by ensemble member and time range (see :func:`make_key`)."""
    tas_map = {(make_key(Path(f)), get_timerange(Path(f))): f for f in tas_files}
    pr_map = {(make_key(Path(f)), get_timerange(Path(f))): f for f in pr_files}
    keys = sorted(set(tas_map) & set(pr_map))
    return [tas_map[k] for k in keys], [pr_map[k] for k in keys]


def build_tas_pr_dataset(
    tas_data_path: str | Path,
    pr_data_path: str | Path,
    hr_scale: int = 1,
    esm_name: str = DEFAULT_ESM,
    experiments: Sequence[str] = DEFAULT_EXPERIMENTS,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> tuple[NetCDFTimeDataset, dict[str, NetCDFStatsPrepper]]:
    """Discover, pair, and load matching tas/pr NetCDF files into a :class:`NetCDFTimeDataset`.

    ``esm_name``, ``experiments`` and ``excludes`` select the files; see
    :func:`discover_experiment_files`. For several ESMs with shared normalisation
    see :func:`build_multi_esm_dataset`.

    Both variables are normalised to ``[-1, 1]`` using their own min/max.

    Returns the dataset and the fitted :class:`NetCDFStatsPrepper` for each
    variable, so callers can invert the normalisation on model output with
    :func:`denormalize`.
    """
    tas_files = discover_experiment_files(tas_data_path, "tas", esm_name, experiments, excludes)
    pr_files = discover_experiment_files(pr_data_path, "pr", esm_name, experiments, excludes)
    tas_files, pr_files = pair_tas_pr_files(tas_files, pr_files)
    if not tas_files:
        raise ValueError(f"No matching tas/pr file pairs for {esm_name} found under {tas_data_path} / {pr_data_path}")

    preppers = {
        "tas": NetCDFStatsPrepper(tas_files, var_name="tas"),
        "pr": NetCDFStatsPrepper(pr_files, var_name="pr"),
    }
    bounds = {var: prepper.get_ds_max_min() for var, prepper in preppers.items()}  # (vmax, vmin)

    dataset = NetCDFTimeDataset(
        vars_files={"tas": tas_files, "pr": pr_files},
        region_data={var: prepper.get_region_means() for var, prepper in preppers.items()},
        hr_scale=hr_scale,
        transform={
            var: (lambda x, vmax=vmax, vmin=vmin: normalize(x, vmin, vmax))
            for var, (vmax, vmin) in bounds.items()
        },
    )
    return dataset, preppers


# ---------------------------------------------------------------------------
# Multi-ESM task dictionary (MAML/CNP-style), shared normalisation
# ---------------------------------------------------------------------------


def _split_runs(tas_files: list[str], support_frac: float, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Split paired-file indices into support/query, keeping all time chunks of one run together."""
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(tas_files):
        groups.setdefault(make_key(Path(f)), []).append(i)
    keys = sorted(groups)
    order = rng.permutation(len(keys))
    n_support = min(max(1, int(np.floor(support_frac * len(keys)))), len(keys) - 1)
    support = [i for k in order[:n_support] for i in groups[keys[k]]]
    query = [i for k in order[n_support:] for i in groups[keys[k]]]
    return sorted(support), sorted(query)


def build_multi_esm_dataset(
    tas_data_path: str | Path,
    pr_data_path: str | Path,
    esm_names: Sequence[str],
    experiments: Sequence[str] = DEFAULT_EXPERIMENTS,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
    hr_scale: int = 1,
    support_frac: float = 0.7,
    weights: dict[str, float] | None = None,
    fit_stats_on: str = "support",
    seed: int = 0,
) -> tuple[dict[str, dict], dict[str, tuple[float, float]]]:
    """Build a task dictionary with one task per ESM, all sharing one tas/pr normalisation.

    All ESMs must be on the same lat/lon grid. For each ESM the paired tas/pr files
    are grouped by run (:func:`make_key`, i.e. experiment + member, so time chunks of
    one run stay together) and the runs are split by a seeded shuffle into a support
    set (``support_frac``) and a query set, each a :class:`NetCDFTimeDataset`.

    ``vmax``/``vmin`` are computed once over the files of all ESMs and used for every
    task; ``fit_stats_on`` is ``"support"`` (default, no query data enters the
    statistics, like ``build_task_dict`` in SCALES) or ``"all"``.

    ESMs without any matching tas/pr pair, or with fewer than two runs, are reported on
    stdout and skipped. Missing experiments are reported by
    :func:`discover_experiment_files`.

    Returns:
        tasks:  ``{esm_name: {"support": ds, "query": ds, "esm_id": int, "weight": float}}``.
                ``esm_id`` is the index into ``esm_names`` (use ``F.one_hot`` or an
                ``nn.Embedding``); it is stable even if an ESM in the middle is skipped.
                Both datasets also yield the ID as a third item per sample.
        bounds: ``{"tas": (vmax, vmin), "pr": (vmax, vmin)}`` for :func:`denormalize`.
    """
    if fit_stats_on not in ("support", "all"):
        raise ValueError(f"fit_stats_on must be 'support' or 'all', got {fit_stats_on!r}")
    weights = weights or {}
    rng = np.random.default_rng(seed)

    # ---- Pass 1: discover, pair, split, and compute region means / per-file extrema ----
    raw: dict[str, dict] = {}
    for esm_id, esm_name in enumerate(esm_names):
        tas_files = discover_experiment_files(tas_data_path, "tas", esm_name, experiments, excludes)
        pr_files = discover_experiment_files(pr_data_path, "pr", esm_name, experiments, excludes)
        tas_files, pr_files = pair_tas_pr_files(tas_files, pr_files)
        if not tas_files:
            print(f"[mesh_dataset] {esm_name}: no matching tas/pr file pairs; skipping ESM")
            continue
        if len({make_key(Path(f)) for f in tas_files}) < 2:
            print(f"[mesh_dataset] {esm_name}: need at least 2 runs for a support/query split; skipping ESM")
            continue

        sup_idx, qry_idx = _split_runs(tas_files, support_frac, rng)
        preppers = {
            "tas": NetCDFStatsPrepper(tas_files, var_name="tas"),
            "pr": NetCDFStatsPrepper(pr_files, var_name="pr"),
        }
        raw[esm_name] = {
            "files": {"tas": tas_files, "pr": pr_files},
            "preppers": preppers,
            "support": sup_idx,
            "query": qry_idx,
            "esm_id": esm_id,
        }

    if not raw:
        raise ValueError(f"No usable ESM found among {list(esm_names)} under {tas_data_path} / {pr_data_path}")

    # ---- Shared normalisation bounds across all ESMs ----
    bounds: dict[str, tuple[float, float]] = {}
    for var in ("tas", "pr"):
        mins, maxs = [], []
        for d in raw.values():
            idx = d["support"] if fit_stats_on == "support" else range(len(d["files"][var]))
            mins += [d["preppers"][var].file_mins[i] for i in idx]
            maxs += [d["preppers"][var].file_maxs[i] for i in idx]
        bounds[var] = (max(maxs), min(mins))  # (vmax, vmin)

    transform = {
        var: (lambda x, vmax=vmax, vmin=vmin: normalize(x, vmin, vmax)) for var, (vmax, vmin) in bounds.items()
    }

    # ---- Pass 2: build support/query datasets ----
    tasks: dict[str, dict] = {}
    ref_size: tuple[int, int] | None = None
    for esm_name, d in raw.items():

        def make(idx: list[int], d: dict = d) -> NetCDFTimeDataset:
            return NetCDFTimeDataset(
                vars_files={v: [d["files"][v][i] for i in idx] for v in ("tas", "pr")},
                region_data={v: [d["preppers"][v].get_region_means()[i] for i in idx] for v in ("tas", "pr")},
                hr_scale=hr_scale,
                transform=transform,
                esm_id=d["esm_id"],
            )

        support, query = make(d["support"]), make(d["query"])
        if ref_size is None:
            ref_size = support.hr_size
        if support.hr_size != ref_size or query.hr_size != ref_size:
            raise ValueError(
                f"ESM '{esm_name}' has grid {support.hr_size}/{query.hr_size}, expected {ref_size}; "
                "all ESMs must be on the same grid"
            )
        tasks[esm_name] = {
            "support": support,
            "query": query,
            "esm_id": d["esm_id"],
            "weight": float(weights.get(esm_name, 1.0)),
        }

    return tasks, bounds
