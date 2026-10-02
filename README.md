# SCALES-MESH

Generative emulators for regional climate fields, conditioned on a global
mean temperature (GMT) trajectory. The repository has two independent
components:

- **SCALES** (`scales/`) — emulates *regional, monthly* `tas`/`pr` time
  series (e.g. IPCC AR6 region means) from a GMT scenario. Two model
  families are provided:
  - `scales.dit` — a 2-D diffusion transformer (DiT) over (time, region)
    tokens that denoises a window of monthly tas/pr values, conditioned on
    the GMT history via a causal encoder.
  - `scales.ssm` — a deep state-space model (`DeepSSMPatternConditioned`)
    with a low-rank/diagonal Gaussian emission for `tas` and a Sinh-Arcsinh
    flow emission for `pr`, plus a Conditional Neural Process (CNP) wrapper
    (`scales.ssm.cnp_ssm`) that adapts the emulator to different Earth
    System Models (ESMs) from a small context set.
- **MESH** (`mesh/`) — spatial downscaling: a ViT/DiT-style flow-matching
  model that generates *gridded, high-resolution* `tas`/`pr` fields
  conditioned on the corresponding region-mean values.

Both operate on the same underlying idea (GMT/region-mean forcing in,
regional climate response out) but at different spatial resolutions — SCALES
per-region, MESH per-gridpoint.

> **SCALES and MESH are not directly chainable in this version.** SCALES
> (`scales.dit` / `scales.ssm`) is trained and normalised on tas/pr
> **anomalies**; MESH (`mesh`) is trained and normalised on **absolute**
> tas/pr values. Feeding SCALES's output straight into MESH as conditioning
> would silently mix the two scales. MESH's own conditioning is built
> directly from real tas/pr NetCDF data via `mesh.mesh_dataset`, not from a
> SCALES emulator.

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
# test/dev tools (pytest, ruff, pre-commit):
pip install -e ".[dev]"
```

Core dependencies: `torch`, `torchvision`, `numpy`, `xarray`, `netCDF4`,
`dask`, `regionmask` and `requests` (`xarray`/`netCDF4`/`dask`/`regionmask`
are only needed for `mesh`, which reads CMIP-style NetCDF files and computes
IPCC AR6 region statistics; `requests` is only needed to download a
pretrained checkpoint from Zenodo, see below).

## Usage

### SCALES — DiT emulator (`scales.dit`)

Input format: a list of numpy arrays, one per simulation, each shaped
`(1 + n_tas + n_pr, T)` with `T` a multiple of 12 — row 0 is the (annual)
GMT, the rest are monthly regional `tas` then `pr` values. See
`scales/dit/data.py` for the full layout and the preprocessing rationale.

```python
from scales.dit import Config, train_from_sims, ScenarioSampler

cfg = Config()
cfg.train.out_dir = "runs/v1"
cfg.train.max_steps = 200_000
out = train_from_sims(sims, cfg, groups=scenario_labels)

s = ScenarioSampler.from_checkpoint("runs/v1/last.pt", device="cuda")
ens = s.sample(new_gmt_monthly, n_members=20)   # (20, n_tas + n_pr, N)
```

Or from the command line, once you have a checkpoint:

```bash
python -m scales.dit.inference \
    --checkpoint runs/v1/best.pt \
    --gmt gmt_scenario.npy \
    --members 20 --out emulated.npy
```

#### Pretrained checkpoints (Zenodo)

Pretrained SCALES DiT checkpoints are published on Zenodo. `ScenarioSampler.from_zenodo`
(and the `--zenodo-record` CLI flag) fetch and cache a checkpoint by its
Zenodo record — no manual download needed:

```python
from scales.dit import ScenarioSampler

s = ScenarioSampler.from_zenodo("10.5281/zenodo.22899657", device="cuda")
ens = s.sample(new_gmt_monthly, n_members=20)
```

```bash
python -m scales.dit.inference \
    --zenodo-record 10.5281/zenodo.22899657 \
    --gmt gmt_scenario.npy \
    --members 20 --out emulated.npy
```

The checkpoint is downloaded once, verified against Zenodo's published
checksum, and cached under `~/.cache/scalesmesh/zenodo/` (override with the
`SCALESMESH_CACHE` environment variable); later calls reuse the cached copy.
`record` accepts a Zenodo record ID, DOI, or URL; pass `filename=` /
`--zenodo-file` if a record ever holds more than one file.

`scales/dit/evaluate.py` has diagnostics (`report`, `spread_ratio`,
`trend_drift`, ...) for checking a trained emulator against a reference ESM
ensemble before trusting it.

### SCALES — state-space emulator (`scales.ssm`)

```python
from scales.ssm import run_train, SSMForecaster

model, y_scaler, u_scaler, pr_scaler = run_train(
    y_np, pr_np, u_np,           # (N, T, Dy), (N, T, Dy), (N, T, Du)
    run_dir="runs/ssm_v1",
)

fc = SSMForecaster.from_checkpoint(
    "runs/ssm_v1/checkpoints/model_epoch0050.pt",
    tas_scaler_path="runs/ssm_v1/y_scaler.out",
    gmt_scaler_path="runs/ssm_v1/u_scaler.out",
    pr_scaler_path="runs/ssm_v1/pr_scaler.out",
)
out = fc.forecast(gmt, tas_context)   # gmt must cover context + horizon
out.tas_mean                          # (H, Dy), physical units
```

#### Pretrained checkpoints (Zenodo)

A pretrained SCALES SSM checkpoint, together with its three fitted
`StandardScaler`s, is published at
[10.5281/zenodo.22998100](https://doi.org/10.5281/zenodo.22998100).
`SSMForecaster.from_zenodo` downloads and caches all four files from that
one record (model file auto-detected by its `.pt` extension; scalers by
their `tas_scaler.out`/`gmt_scaler.out`/`pr_scaler.out` names) and builds a
ready-to-use forecaster in one call:

```python
from scales.ssm import SSMForecaster

fc = SSMForecaster.from_zenodo("10.5281/zenodo.22998100", device="cuda")
out = fc.forecast(gmt, tas_context)   # gmt must cover context + horizon
out.tas_mean                          # (H, Dy), physical units
```

`scales.ssm.forecast_from_zenodo` is the equivalent one-shot convenience
wrapper (download + normalise + forecast in one call), analogous to
`forecast_from_checkpoint`. If a Zenodo record ever uses different
filenames, or publishes the model without scalers, pass
`checkpoint_filename`/`tas_scaler_filename`/`gmt_scaler_filename`/`pr_scaler_filename`
to override, or fall back to `SSMForecaster.from_checkpoint` with local
scaler paths.

To generalize across multiple ESMs, wrap a trained SSM with a CNP that
infers an ESM embedding from a small context set:

```python
from scales.ssm import build_task_dict, train_cnp, DeepCnpSsmforESM, CnpForecaster

tasks, scalers = build_task_dict(esm_data, run_dir="runs/cnp_v1")
model = train_cnp(DeepCnpSsmforESM(ssm_model, r_dim=128, z_cnp_dim=16),
                   tasks, num_epochs=100, horizon=1200, run_dir="runs/cnp_v1")

fc = CnpForecaster.from_run_dir("runs/cnp_v1")
out = fc.forecast(gmt, tas_context, pr_context)
```

#### Pretrained checkpoints (Zenodo)

A pretrained SCALES CNP checkpoint, together with its three fitted
`StandardScaler`s, is published at
[10.5281/zenodo.22998276](https://doi.org/10.5281/zenodo.22998276).
`CnpForecaster.from_zenodo` downloads and caches all four files from that
one record, the same way `SSMForecaster.from_zenodo` does for the plain SSM
(see above):

```python
from scales.ssm import CnpForecaster

fc = CnpForecaster.from_zenodo("10.5281/zenodo.22998276", device="cuda")
out = fc.forecast(gmt, tas_context, pr_context)
```

`scales.ssm.cnp_forecast_from_zenodo` (imported under that name to avoid
clashing with the plain-SSM `forecast_from_zenodo`; it's `forecast_from_zenodo`
inside `scales.ssm.cnp_inference` itself) is the equivalent one-shot
convenience wrapper. As with the SSM record, pass
`checkpoint_filename`/`tas_scaler_filename`/`gmt_scaler_filename`/`pr_scaler_filename`
to override the expected filenames.

### MESH — spatial downscaling (`mesh`)

`mesh.mesh_dataset` discovers, pairs, and loads matching `tas`/`pr` NetCDF
files (paired by filename via `make_key`/`get_timerange`) into a
`NetCDFTimeDataset`, and computes the region-mean conditioning and the
per-variable min/max used to normalise both to `[-1, 1]`:

```python
from mesh.mesh_dataset import build_tas_pr_dataset

dataset, preppers = build_tas_pr_dataset("/path/to/tas/", "/path/to/pr/", hr_scale=1)
lr, hr = dataset[0]   # lr: (2, n_regions) region means; hr: (2, H, W) gridded field, both normalised

vmax_tas, vmin_tas = preppers["tas"].get_ds_max_min()   # for mapping model output back to physical units
```

(`NetCDFStatsPrepper` and `NetCDFTimeDataset` are also usable directly for a
custom file layout; `build_tas_pr_dataset` is the convenience wrapper both
training and inference use.)

`mesh.mesh_vit_flow` trains the flow-matching super-resolution model on top
of this dataset and is launched with `torchrun` (it always initializes a
process group, even for a single process):

```bash
torchrun --nproc_per_node=1 -m mesh.mesh_vit_flow \
    --tas-data-path /path/to/tas/ \
    --pr-data-path /path/to/pr/
```

`--tas-data-path`/`--pr-data-path` are directories of CMIP-style monthly
NetCDF files (e.g. `tas_Amon_ACCESS-ESM1-5_ssp585_r1i1p1f1_gn_YYYYMM-YYYYMM.nc`);
`main()` in `mesh/mesh_vit_flow.py` currently filters to the ACCESS-ESM1-5
experiments it was developed against, so adjust that filter for other models
or experiments. Training writes `model_flow_ema.pt` (the checkpoint to use
for inference) and `model_flow_out.pt` (raw, non-EMA weights) to a
timestamped `outputs_DiffusionTransformer/` run directory.

`mesh.mesh_inference` downscales `tas`/`pr` fields from a trained checkpoint,
building its conditioning the same way as training — from real NetCDF data
via `mesh.mesh_dataset`, **not** from a SCALES emulator's output (see the
anomalies-vs-absolute-values note above):

```bash
python -m mesh.mesh_inference \
    --checkpoint outputs_DiffusionTransformer/<run>/model_flow_ema.pt \
    --tas-data-path /path/to/tas/ \
    --pr-data-path /path/to/pr/ \
    --n-samples 8 --steps 15 --sampler heun --out downscaled.npz
```

This writes an `.npz` with `tas_gen`/`pr_gen` (generated) and
`tas_truth`/`pr_truth` (the dataset's own fields, for comparison), all in
physical units, and prints a quick MAE-vs-ground-truth sanity check. The
`--dim`/`--depth`/`--heads`/`--patch`/`--lr-patch` flags must match the
architecture the checkpoint was trained with (they default to
`mesh_vit_flow.main()`'s training defaults). By default the EMA weights are
used when the checkpoint has them (pass `--no-ema` for the raw weights).

#### Pretrained checkpoints (Zenodo)

A pretrained MESH checkpoint is published at
[10.5281/zenodo.23100103](https://doi.org/10.5281/zenodo.23100103) --
`--zenodo-record` fetches and caches it the same way as for SCALES:

```bash
python -m mesh.mesh_inference \
    --zenodo-record 10.5281/zenodo.23100103 \
    --tas-data-path /path/to/tas/ \
    --pr-data-path /path/to/pr/ \
    --n-samples 8 --steps 15 --sampler heun --out downscaled.npz
```

Programmatically, `mesh.mesh_inference.load_model_from_zenodo(...)` downloads
the checkpoint and returns a ready-to-use `ViTCondDiffusionSR` in one call,
in place of `load_model(checkpoint_path, ...)`. This particular record is a
mid-training checkpoint (holding both the raw and EMA weights, as
`mesh_vit_flow.train()` writes for resuming) rather than a bare
`model_flow_ema.pt`; both checkpoint shapes are handled transparently, and
the EMA weights are preferred by default either way.

## Testing

```bash
pytest
```

## License

SCALES-MESH is licensed under the Apache License, Version 2.0 (see
`LICENSE`). Use of the name "SCALES-MESH" to describe results,
publications, products or services is additionally subject to the Naming
and Calibration Reporting Condition in `NOTICE`, which requires stating
the software version and the version/DOI of the Official Calibration
(model weights) used — see "Official calibrations" below for the current
register that condition refers to.

## Official calibrations

These are the Official Calibrations referenced by the Naming and
Calibration Reporting Condition in `NOTICE`. Each row is the current
released set of model weights for that component; superseded calibrations
will be listed underneath the component they replace, together with the
date they were superseded.

| Component | Software version | Calibration version | DOI |
| --- | --- | --- | --- |
| SCALES — DiT (`scales.dit`) | v1.0.1 | v1.0.0 | [10.5281/zenodo.22899657](https://doi.org/10.5281/zenodo.22899657) |
| SCALES — SSM (`scales.ssm`) | v1.0.1 | v1.1.0 | [10.5281/zenodo.22998100](https://doi.org/10.5281/zenodo.22998100) |
| SCALES — CNP (`scales.ssm.cnp_ssm`) | v1.0.1 | v1.1.0 | [10.5281/zenodo.22998276](https://doi.org/10.5281/zenodo.22998276) |
| MESH (`mesh`) | v1.0.1 | v1.0.1 | [10.5281/zenodo.23100103](https://doi.org/10.5281/zenodo.23100103) |

## Layout

```
common/
  zenodo.py   download_from_zenodo() -- model-agnostic; used by scales.dit
              today, and by scales.ssm/mesh once their checkpoints are
              published too
scales/
  dit/    diffusion-transformer emulator (config, data, model, diffusion,
          train, sample, inference, evaluate)
  ssm/    state-space emulator (ssm_tas_pr, cnp_ssm) and their inference
          wrappers (ssm_inference, cnp_inference)
mesh/
  mesh_dataset.py    NetCDF loading + region statistics
  mesh_vit_flow.py   ViT/DiT flow-matching downscaling model + training
  mesh_inference.py  downscaling from a trained checkpoint
tests/    pytest suite for both packages
```
