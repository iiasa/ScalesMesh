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
`dask` and `regionmask` (the last three are only needed for `mesh`, which
reads CMIP-style NetCDF files and computes IPCC AR6 region statistics).

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
`mesh_vit_flow.main()`'s training defaults).

## Testing

```bash
pytest
```

## Layout

```
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
