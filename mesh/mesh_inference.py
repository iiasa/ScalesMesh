"""Inference for the MESH flow-matching downscaling model.

Loads a trained :class:`~mesh.mesh_vit_flow.ViTCondDiffusionSR` checkpoint
(the EMA weights written by ``mesh.mesh_vit_flow``, or downloaded from
Zenodo via :mod:`common.zenodo`) and generates high-resolution tas/pr fields
conditioned on low-resolution region means built directly from real NetCDF
data via :mod:`mesh.mesh_dataset` -- the same input the model was trained on.

This does *not* take a SCALES emulator's output as conditioning: SCALES
(``scales.dit`` / ``scales.ssm``) is trained on tas/pr *anomalies*, while
MESH in this version is trained and normalised on *absolute* tas/pr values,
so the two are not interchangeable inputs (see the project README).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import torch

from common.zenodo import download_from_zenodo

from .mesh_dataset import NetCDFStatsPrepper, NetCDFTimeDataset, build_tas_pr_dataset, denormalize
from .mesh_vit_flow import FlowMatchingLinear, ViTCondDiffusionSR


def _load_state_dict(checkpoint_path: str, use_ema: bool = True) -> dict[str, torch.Tensor]:
    """Load a MESH state dict from a bare checkpoint (``model_flow_ema.pt`` /
    ``model_flow_out.pt``) or a mid-training checkpoint dict (as written by
    ``mesh_vit_flow.train()`` for resuming).

    A mid-training checkpoint holds both the raw training weights
    (``"model_state_dict"``) and, once training has reached its first EMA
    update, an EMA shadow (``"ema_state_dict"``, itself
    ``{"ema_model": <state dict>, "num_updates": ..., "decay": ...}`` from
    ``ModelEMA.state_dict()``). ``use_ema=True`` (the default, matching
    ``mesh_vit_flow.main()``'s own preference) picks the EMA weights when
    present. Any torch.compile/DDP key prefixes are stripped either way.
    """
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        ema = obj.get("ema_state_dict")
        state = ema["ema_model"] if (use_ema and ema is not None) else obj["model_state_dict"]
    else:
        state = obj
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}


def load_model(
    checkpoint_path: str,
    lr_regions: int,
    varmax: tuple[float, float],
    varmin: tuple[float, float],
    hr_hw: tuple[int, int],
    dim: int = 192,
    depth: int = 16,
    heads: int = 6,
    patch: int = 3,
    lr_patch: int = 3,
    use_ema: bool = True,
    device: str | torch.device = "cpu",
) -> ViTCondDiffusionSR:
    """Rebuild a ViTCondDiffusionSR with the given hyperparameters and load a checkpoint.

    The architecture hyperparameters (dim/depth/heads/patch/lr_patch) are not
    stored in the checkpoint and must match the ones used for training, or
    ``load_state_dict`` fails with a shape mismatch.
    """
    model = ViTCondDiffusionSR(
        varmax=varmax, varmin=varmin, in_ch=2, dim=dim, depth=depth, heads=heads,
        patch=patch, lr_patch=lr_patch, hr_hw=hr_hw, lr_regions=lr_regions,
    )
    model.load_state_dict(_load_state_dict(checkpoint_path, use_ema=use_ema), strict=True)
    return model.to(device).eval()


def load_model_from_zenodo(
    record: str | int,
    lr_regions: int,
    varmax: tuple[float, float],
    varmin: tuple[float, float],
    hr_hw: tuple[int, int],
    filename: str | None = None,
    dim: int = 192,
    depth: int = 16,
    heads: int = 6,
    patch: int = 3,
    lr_patch: int = 3,
    use_ema: bool = True,
    device: str | torch.device = "cpu",
    dest_dir: str | None = None,
    force: bool = False,
) -> ViTCondDiffusionSR:
    """Download a checkpoint from a Zenodo record and load it with `load_model`.

    ``filename`` selects which file to fetch when the record holds more than
    one; if omitted and the record has exactly one file, that file is used.
    """
    checkpoint_path = download_from_zenodo(record, filename=filename, dest_dir=dest_dir, force=force)
    return load_model(
        str(checkpoint_path), lr_regions=lr_regions, varmax=varmax, varmin=varmin, hr_hw=hr_hw,
        dim=dim, depth=depth, heads=heads, patch=patch, lr_patch=lr_patch, use_ema=use_ema, device=device,
    )


@torch.no_grad()
def downscale(
    model: ViTCondDiffusionSR,
    dataset: NetCDFTimeDataset,
    preppers: dict[str, NetCDFStatsPrepper],
    indices: Sequence[int],
    n_steps: int = 15,
    sampler: str = "heun",
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Generate HR tas/pr fields for the given dataset indices, in physical units.

    Returns a dict of ``tas_gen``, ``pr_gen`` (generated), ``tas_truth``,
    ``pr_truth`` (the dataset's own HR fields, for comparison) and
    ``indices``, all as numpy arrays.
    """
    lr = torch.stack([dataset[i][0] for i in indices]).to(device)
    hr_truth = torch.stack([dataset[i][1] for i in indices]).to(device)

    fm = FlowMatchingLinear(hr_hw=dataset.hr_size, channels=2, device=device)
    sample_fn = fm.sample_heun if sampler == "heun" else fm.sample
    hr_gen = sample_fn(model, lr, n_steps=n_steps)

    vmax_tas, vmin_tas = preppers["tas"].get_ds_max_min()
    vmax_pr, vmin_pr = preppers["pr"].get_ds_max_min()

    return {
        "tas_gen": denormalize(hr_gen[:, 0], vmin_tas, vmax_tas).cpu().numpy(),
        "pr_gen": denormalize(hr_gen[:, 1], vmin_pr, vmax_pr).cpu().numpy(),
        "tas_truth": denormalize(hr_truth[:, 0], vmin_tas, vmax_tas).cpu().numpy(),
        "pr_truth": denormalize(hr_truth[:, 1], vmin_pr, vmax_pr).cpu().numpy(),
        "indices": np.asarray(list(indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MESH downscaling inference")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to a local MESH .pt checkpoint")
    src.add_argument("--zenodo-record", help="Zenodo record ID, DOI or URL to fetch the checkpoint from")
    parser.add_argument("--zenodo-file", default=None,
                        help="filename to fetch from --zenodo-record, if it holds more than one file")
    parser.add_argument("--tas-data-path", required=True, help="directory of tas NetCDF files")
    parser.add_argument("--pr-data-path", required=True, help="directory of pr NetCDF files")
    parser.add_argument("--indices", type=str, default=None,
                        help="comma-separated dataset indices to downscale (default: first --n-samples)")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="how many items to downscale when --indices is not given")
    parser.add_argument("--steps", type=int, default=15, help="ODE integration steps")
    parser.add_argument("--sampler", choices=["euler", "heun"], default="heun")
    parser.add_argument("--hr-scale", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="mesh_downscaled.npz", help="output .npz path")
    parser.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA")
    # Model hyperparameters -- must match the checkpoint being loaded.
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--patch", type=int, default=3)
    parser.add_argument("--lr-patch", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset, preppers = build_tas_pr_dataset(args.tas_data_path, args.pr_data_path, hr_scale=args.hr_scale)
    print(f"[inference] loaded dataset: {len(dataset)} samples, hr_size={dataset.hr_size}")

    indices = (
        [int(i) for i in args.indices.split(",")] if args.indices
        else list(range(min(args.n_samples, len(dataset))))
    )

    if args.zenodo_record:
        checkpoint_path = str(download_from_zenodo(args.zenodo_record, filename=args.zenodo_file))
    else:
        checkpoint_path = args.checkpoint

    vmax_tas, vmin_tas = preppers["tas"].get_ds_max_min()
    vmax_pr, vmin_pr = preppers["pr"].get_ds_max_min()
    model = load_model(
        checkpoint_path, lr_regions=preppers["tas"].get_n_regions(),
        varmax=(vmax_tas, vmax_pr), varmin=(vmin_tas, vmin_pr), hr_hw=dataset.hr_size,
        dim=args.dim, depth=args.depth, heads=args.heads, patch=args.patch, lr_patch=args.lr_patch,
        use_ema=not args.no_ema, device=device,
    )
    print("[inference] model loaded")

    out = downscale(model, dataset, preppers, indices, n_steps=args.steps, sampler=args.sampler, device=device)
    np.savez(args.out, **out)

    for var in ("tas", "pr"):
        gen, truth = out[f"{var}_gen"], out[f"{var}_truth"]
        mae = np.abs(gen - truth).mean()
        print(f"[inference] {var}: generated range [{gen.min():.3g}, {gen.max():.3g}]  "
              f"MAE vs. ground truth = {mae:.4g}")
    print(f"[done] saved {len(indices)} downscaled sample(s) to {args.out}")


if __name__ == "__main__":
    main()
