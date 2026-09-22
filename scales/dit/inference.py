"""Simple inference: sample from a trained checkpoint given a monthly GMT timeseries.

The checkpoint can be a local file, or a record published on Zenodo (fetched
and cached via :mod:`common.zenodo`).

Example
-------
    from scales.dit import ScenarioSampler, run_inference
    import numpy as np

    gmt = np.load("gmt.npy")                       # 1-D, length = multiple of 12
    results = run_inference("best.pt", gmt, n_members=10, seed=42)
    # results: list of (1 + n_tas + n_pr, T) arrays
    #   row 0    : GMT (echoed)
    #   rows 1.. : emulated fields in physical units

    # or, from a Zenodo-hosted checkpoint:
    sampler = ScenarioSampler.from_zenodo("10.5281/zenodo.1234567", device="cpu")
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common.zenodo import download_from_zenodo

from .sample import ScenarioSampler


def load_gmt(path: str) -> np.ndarray:
    """Load a monthly GMT array from a .npy file or a plain-text file."""
    arr = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size % 12 != 0:
        raise ValueError(
            f"GMT length {arr.size} is not a multiple of 12 months. "
            "Trim or pad to a whole number of years."
        )
    return arr


def save_object_list(path: str, arrays) -> None:
    obj = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        obj[i] = np.asarray(a)
    np.save(path, obj, allow_pickle=True)


def run_inference(
    checkpoint: str,
    gmt_monthly: np.ndarray,
    n_members: int = 5,
    stride: int | None = None,
    sample_steps: int | None = None,
    device: str = "cuda",
    use_ema: bool = True,
    seed: int | None = None,
    area_weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Sample from a trained checkpoint conditioned on a monthly GMT trajectory.

    Returns a list of ``n_members`` arrays, each shaped ``(1 + n_tas + n_pr, T)``:
      row 0    : GMT (echoed from ``gmt_monthly``)
      rows 1.. : emulated climate fields in physical units
    """
    sampler = ScenarioSampler.from_checkpoint(checkpoint, device=device, use_ema=use_ema)
    cfg = sampler.cfg
    stride = stride if stride is not None else cfg.data.window // 2
    sample_steps = sample_steps if sample_steps is not None else cfg.diffusion.sample_steps

    T = gmt_monthly.size
    n_tas = cfg.data.n_tas
    n_pr = cfg.data.n_pr
    print(
        f"[inference] checkpoint : {checkpoint}\n"
        f"[inference] GMT length : {T} months ({T // 12} years)\n"
        f"[inference] members    : {n_members}\n"
        f"[inference] stride     : {stride}  sample_steps={sample_steps}\n"
        f"[inference] layout     : 1 GMT + {n_tas} tas + {n_pr} pr",
        flush=True,
    )

    ens = sampler.sample(
        gmt_monthly,
        n_members=n_members,
        stride=stride,
        steps=sample_steps,
        seed=seed,
        area_weights=area_weights,
        progress=True,
    )

    results = []
    for m in range(ens.shape[0]):
        arr = np.empty((1 + n_tas + n_pr, T), dtype=np.float32)
        arr[0] = gmt_monthly
        arr[1:] = ens[m]
        results.append(arr)
    return results


def main() -> None:
    """CLI entry point: ``python -m scales.dit.inference``."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to a local .pt checkpoint (best.pt or last.pt)")
    src.add_argument("--zenodo-record", help="Zenodo record ID, DOI or URL to fetch the checkpoint from")
    p.add_argument("--zenodo-file", default=None,
                   help="filename to fetch from --zenodo-record, if it holds more than one file")
    p.add_argument("--gmt", required=True,
                   help="monthly GMT timeseries: .npy or plain-text, length % 12 == 0")
    p.add_argument("--out", default="emulated.npy",
                   help="output path (default: emulated.npy)")
    p.add_argument("--members", type=int, default=5,
                   help="ensemble members to generate (default: 5)")
    p.add_argument("--stride", type=int, default=None,
                   help="window stride in months (default: window // 2)")
    p.add_argument("--sample-steps", type=int, default=None,
                   help="DDIM steps (default: from checkpoint)")
    p.add_argument("--device", default="cuda",
                   help="torch device (default: cuda)")
    p.add_argument("--no-ema", action="store_true",
                   help="use raw weights instead of EMA")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducibility")
    p.add_argument("--area-weights", default=None,
                   help=".npy file with per-region area weights (n_tas,) to "
                        "hard-constrain the area-weighted mean of tas to the GMT")
    args = p.parse_args()

    if args.zenodo_record:
        checkpoint = str(download_from_zenodo(args.zenodo_record, filename=args.zenodo_file))
    else:
        checkpoint = args.checkpoint
        if not os.path.exists(checkpoint):
            raise SystemExit(f"[fatal] checkpoint not found: {checkpoint}")

    gmt = load_gmt(args.gmt)
    area_weights = np.load(args.area_weights).astype(np.float32) if args.area_weights else None

    results = run_inference(
        checkpoint=checkpoint,
        gmt_monthly=gmt,
        n_members=args.members,
        stride=args.stride,
        sample_steps=args.sample_steps,
        device=args.device,
        use_ema=not args.no_ema,
        seed=args.seed,
        area_weights=area_weights,
    )

    out_path = args.out if args.out.endswith(".npy") else args.out + ".npy"
    save_object_list(out_path, results)
    print(f"[done] {len(results)} members -> {out_path}")
    print(f"[done] load with:  sims = list(np.load('{out_path}', allow_pickle=True))")
