"""Multi-ESM flow-matching super-resolution for MESH, with a learned ESM embedding.

Builds on :mod:`mesh.mesh_vit_flow`: the same ViT/DiT backbone and linear flow
matching, but a learned per-ESM embedding is added to the time embedding that
drives the AdaLN conditioning of every block (DiT-style class conditioning).
Training pools the support sets of all ESMs from
:func:`mesh.mesh_dataset.build_multi_esm_dataset` and weights each sample's
loss by its ESM, so ESMs with few files are not drowned out by the larger ones
(see :func:`compute_esm_weights`). Validation is reported per ESM on the query
sets.

Launch with ``torchrun`` like ``mesh_vit_flow`` (it always initialises a
process group, even for one process).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .mesh_dataset import DEFAULT_EXCLUDES, DEFAULT_EXPERIMENTS, build_multi_esm_dataset
from .mesh_vit_flow import (
    FlowMatchingLinear,
    ModelEMA,
    ViTCondDiffusionSR,
    log_cuda_devices,
    patchify,
    set_seed,
    unpatchify,
)

# ----------------------------
# Model
# ----------------------------


class ViTCondDiffusionSRMultiESM(ViTCondDiffusionSR):
    """:class:`ViTCondDiffusionSR` with a learned ESM embedding added to the time embedding."""

    def __init__(self, n_esms: int, **kwargs):
        super().__init__(**kwargs)
        self.n_esms = n_esms
        dim = self.hr_in.out_features
        self.esm_emb = nn.Embedding(n_esms, dim)
        nn.init.trunc_normal_(self.esm_emb.weight, std=0.02)

    def forward(self, x_hr_noisy: torch.Tensor, t: torch.Tensor, lr_img: torch.Tensor, esm_id: torch.Tensor):
        """As :meth:`ViTCondDiffusionSR.forward`, plus ``esm_id``: (B,) long ESM indices."""
        hr_tok = self.hr_in(patchify(x_hr_noisy, self.patch)) + self.pos_hr
        lr_tok = self.encode_lr_tokens(lr_img)
        lr_summary = self.lr_summary(lr_tok.mean(dim=1))

        te = self.time_mlp(self.time_emb(t * 1000)) + self.esm_emb(esm_id)
        cond_vec = torch.cat([te, lr_summary], dim=-1)

        x = hr_tok
        for blk in self.blocks:
            x = blk(x, cond_vec, lr_tok)
        v = self.out(self.norm(x))
        return unpatchify(v, self.patch, self.hr_hw, out_ch=self.in_ch)

    def for_esm(self, esm_id: int) -> nn.Module:
        """A ``model(x, t, lr)`` view fixed to one ESM, as expected by ``FlowMatching*.sample``."""
        return _ESMView(self, esm_id)


class _ESMView(nn.Module):
    def __init__(self, model: ViTCondDiffusionSRMultiESM, esm_id: int):
        super().__init__()
        self.model = model
        self.esm_id = esm_id

    def forward(self, x: torch.Tensor, t: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        ids = torch.full((x.size(0),), self.esm_id, dtype=torch.long, device=x.device)
        return self.model(x, t, lr_img, ids)


# ----------------------------
# ESM loss weights
# ----------------------------


def compute_esm_weights(tasks: dict[str, dict], alpha: float = 0.5) -> dict[str, float]:
    """Per-ESM loss weights from the support-set sizes and each task's own ``weight``.

    ``w_e ∝ task_weight_e * n_e**(-alpha)`` where ``n_e`` is the number of support samples:
    ``alpha=0`` leaves the natural sample mix, ``alpha=1`` is full inverse frequency (every
    ESM contributes equally). The weights are rescaled so the sample-averaged weight is 1,
    which keeps the loss on the same scale as the unweighted one.
    """
    n = {e: len(t["support"]) for e, t in tasks.items()}
    raw = {e: float(tasks[e].get("weight", 1.0)) * n[e] ** (-alpha) for e in tasks}
    scale = sum(n.values()) / sum(n[e] * raw[e] for e in tasks)
    return {e: raw[e] * scale for e in tasks}


# ----------------------------
# Training
# ----------------------------


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda"))


def _is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _flow_batch(fm: FlowMatchingLinear, hr_img: torch.Tensor):
    x0 = torch.randn_like(hr_img)
    t = torch.rand(hr_img.size(0), device=hr_img.device)
    return t, fm.sample_xt(x0, hr_img, t), fm.target_velocity(x0, hr_img, t)


def _per_sample_mse(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    # fp32 outside autocast, as in mesh_vit_flow: bf16 reductions lose precision.
    return ((v_pred.float() - v_target.float()) ** 2).mean(dim=(1, 2, 3))


@torch.no_grad()
def evaluate_per_esm(
    model: nn.Module,
    fm: FlowMatchingLinear,
    query_loaders: dict[str, DataLoader],
    device: torch.device,
) -> dict[str, float]:
    """Unweighted v-MSE per ESM on its query set (summed across ranks when distributed)."""
    model.eval()
    out = {}
    for esm, loader in query_loaders.items():
        tot = torch.zeros(2, device=device, dtype=torch.float64)  # (sum of per-sample mse, count)
        for lr_img, hr_img, esm_id in loader:
            lr_img, hr_img, esm_id = lr_img.to(device), hr_img.to(device), esm_id.to(device)
            t, x_t, v_target = _flow_batch(fm, hr_img)
            with _autocast(device):
                v_pred = model(x_t, t, lr_img, esm_id)
            tot[0] += _per_sample_mse(v_pred, v_target).sum().double()
            tot[1] += hr_img.size(0)
        if dist.is_initialized():
            dist.all_reduce(tot)
        out[esm] = (tot[0] / tot[1].clamp(min=1)).item()
    return out


def train_multi_esm(
    model: nn.Module,
    fm: FlowMatchingLinear,
    train_loader: DataLoader,
    query_loaders: dict[str, DataLoader],
    esm_weights: torch.Tensor,
    checkpoint_path: str,
    epochs: int = 10,
    lr: float = 2e-4,
    device: torch.device = "cuda",
    train_sampler: DistributedSampler | None = None,
    ema: ModelEMA | None = None,
) -> None:
    """Flow-matching training on pooled ESM data with per-ESM loss weights.

    ``esm_weights`` is indexed by ESM id (0 for ids that have no data).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    esm_weights = esm_weights.to(device)

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(cleaned)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for ep in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        loss_sum, n = 0.0, 0
        for lr_img, hr_img, esm_id in train_loader:
            lr_img, hr_img, esm_id = lr_img.to(device), hr_img.to(device), esm_id.to(device)
            t, x_t, v_target = _flow_batch(fm, hr_img)

            with _autocast(device):
                v_pred = model(x_t, t, lr_img, esm_id)
            loss = (esm_weights[esm_id] * _per_sample_mse(v_pred, v_target)).mean()

            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Skip non-finite steps so one bad batch cannot poison the weights.
            if torch.isfinite(loss) and torch.isfinite(gnorm):
                opt.step()
                if ema is not None:
                    ema.update(model)
            elif _is_main():
                print(f"[ep {ep}] skipping step: loss={loss.item()} grad_norm={gnorm.item()}")

            loss_sum += loss.item()
            n += 1

        if ep % 10 == 0 and _is_main():
            ckpt = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss.item(),
            }
            if ema is not None:
                ckpt["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt, checkpoint_path)

        val = evaluate_per_esm(model, fm, query_loaders, device)
        per_esm = " ".join(f"{e}={v:.5f}" for e, v in val.items())
        msg = f"Epoch {ep:02d} | Train weighted v-MSE: {loss_sum / n:.5f} | Val v-MSE: {per_esm}"
        if ema is not None:
            ema_val = evaluate_per_esm(ema.ema_model, fm, query_loaders, device)
            msg += " | EMA val: " + " ".join(f"{e}={v:.5f}" for e, v in ema_val.items())
        if _is_main():
            print(msg)


# ----------------------------
# Main
# ----------------------------


def _parse_esm_weights(items: Sequence[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, _, val = item.partition("=")
        out[name] = float(val)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-ESM MESH with a learned ESM embedding")
    parser.add_argument("--tas-data-path", type=str, required=True)
    parser.add_argument("--pr-data-path", type=str, required=True)
    parser.add_argument("--esm-names", type=str, nargs="+", required=True, help="ESM names; list order defines ESM ids")
    parser.add_argument("--experiments", type=str, nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--excludes", type=str, nargs="*", default=list(DEFAULT_EXCLUDES))
    parser.add_argument("--esm-weights", type=str, nargs="*", default=[], help="Per-ESM task weights, NAME=VALUE")
    parser.add_argument("--weight-alpha", type=float, default=0.5, help="0: natural mix, 1: full inverse frequency")
    parser.add_argument("--support-frac", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="outputs_MultiESM")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log_cuda_devices()

    set_seed(0)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    lr, patch, lr_patch, dim, depth, heads = 2e-4, 3, 3, 192, 16, 6

    # Data
    tasks, bounds = build_multi_esm_dataset(
        args.tas_data_path,
        args.pr_data_path,
        args.esm_names,
        experiments=args.experiments,
        excludes=args.excludes,
        support_frac=args.support_frac,
        weights=_parse_esm_weights(args.esm_weights),
    )
    first = next(iter(tasks.values()))["support"]
    hr_hw = first.hr_size
    regions_dim = first.region_data["tas"][0].shape[-1]
    (vmax_tas, vmin_tas), (vmax_pr, vmin_pr) = bounds["tas"], bounds["pr"]
    esm_w = compute_esm_weights(tasks, alpha=args.weight_alpha)
    for e, t in tasks.items():
        print(f"{e}: id={t['esm_id']} support={len(t['support'])} query={len(t['query'])} loss weight={esm_w[e]:.3f}")

    weight_vec = torch.zeros(len(args.esm_names))
    for e, t in tasks.items():
        weight_vec[t["esm_id"]] = esm_w[e]

    use_distributed = dist.get_world_size() > 1
    train_ds = ConcatDataset([t["support"] for t in tasks.values()])
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    query_loaders = {
        e: DataLoader(
            t["query"],
            batch_size=args.batch_size,
            sampler=DistributedSampler(t["query"], shuffle=False) if use_distributed else None,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        for e, t in tasks.items()
    }

    run_dir = f"{args.out_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    checkpoint_path = f"{run_dir}/checkpoints/checkpoint_flow.pt"
    if _is_main():
        with open(f"{run_dir}/config.json", "w") as f:
            json.dump(
                {
                    "esm_ids": {e: t["esm_id"] for e, t in tasks.items()},
                    "n_esms": len(args.esm_names),
                    "esm_loss_weights": esm_w,
                    "bounds": bounds,
                    "hr_hw": list(hr_hw),
                    "lr_regions": regions_dim,
                    "model": {"dim": dim, "depth": depth, "heads": heads, "patch": patch, "lr_patch": lr_patch},
                    "args": vars(args),
                },
                f,
                indent=2,
            )

    # Model + schedule
    model = ViTCondDiffusionSRMultiESM(
        n_esms=len(args.esm_names),
        dim=dim,
        depth=depth,
        heads=heads,
        patch=patch,
        lr_patch=lr_patch,
        hr_hw=hr_hw,
        lr_regions=regions_dim,
        varmax=(vmax_tas, vmax_pr),
        varmin=(vmin_tas, vmin_pr),
        in_ch=2,
    ).to(device)
    ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if use_cuda else {}
    model = DDP(model, **ddp_kwargs)
    if use_cuda:
        model = torch.compile(model)

    fm = FlowMatchingLinear(hr_hw=hr_hw, channels=2, device=device)
    ema = ModelEMA(model, decay=0.9999)

    train_multi_esm(
        model, fm, train_loader, query_loaders, weight_vec, checkpoint_path,
        epochs=args.epochs, lr=lr, device=device, train_sampler=train_sampler, ema=ema,
    )  # fmt: skip

    if _is_main():
        torch.save(ema.ema_model.state_dict(), f"{run_dir}/model_flow_ema.pt")
        torch.save(ModelEMA._unwrap(model).state_dict(), f"{run_dir}/model_flow_out.pt")


if __name__ == "__main__":
    main()
