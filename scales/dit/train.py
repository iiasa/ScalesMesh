"""Training loop for MISCH-MASCH.

Checkpoint selection matters more than it looks: validation loss on a run
like this typically bottoms out well before max_steps and then rises again,
so keeping only a final ``last.pt`` can silently discard the best model. This
loop tracks the best validation loss, writes ``best.pt`` whenever it
improves, stops on patience, and aborts on a collapse instead of grinding on
in a worse basin.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Hashable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import Config
from .data import CropDataset, Normalizer, check_data, collate, group_split, n_gmt_features
from .diffusion import Diffusion
from .model import build_model


class EMA:
    """Exponential moving average of parameters -- essential for diffusion."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        return self.shadow


def lr_at(step: int, cfg: Config) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / max(t.warmup_steps, 1)
    prog = (step - t.warmup_steps) / max(t.max_steps - t.warmup_steps, 1)
    return t.lr * (0.5 * (1 + math.cos(math.pi * min(prog, 1.0))) * 0.95 + 0.05)


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def save_checkpoint(path, model, ema, normalizer, cfg, step, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "normalizer": normalizer.state_dict(),
            "config": cfg.to_dict(),
            "extra": extra or {},
        },
        tmp,
    )
    #os.replace(tmp, path)   # atomic: a killed job never leaves a half-written file


def make_val_loader(ds_va: CropDataset, cfg: Config) -> DataLoader:
    """A FIXED random subset spread over every validation simulation.

    Iterating the first N batches in index order instead would only ever see
    the earliest months of the first validation run -- a consistent signal,
    but a very narrow basis for a stopping decision.
    """
    n = len(ds_va)
    k = min(n, cfg.train.val_batches * cfg.train.batch_size)
    rng = np.random.default_rng(cfg.data.seed + 777)
    idx = rng.permutation(n)[:k].tolist()
    return DataLoader(
        Subset(ds_va, idx), batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate, drop_last=False,
    )


@torch.no_grad()
def validate(model, diffusion, loader, device) -> float:
    """Deterministic: same crops, same diffusion timesteps, every call."""
    model.eval()
    tot, n = 0.0, 0
    g = torch.Generator(device=device).manual_seed(0)
    for batch in loader:
        batch = _to_device(batch, device)
        tot += diffusion.loss(model, batch, generator=g).item()
        n += 1
    model.train()
    return tot / max(n, 1)


def train_from_sims(
    sims: Sequence[np.ndarray],
    cfg: Config | None = None,
    groups: Sequence[Hashable] | None = None,
    esm_ids: Sequence[int] | None = None,
    verbose: bool = True,
):
    """Train the diffusion model from a list of ``(1 + n_tas + n_pr, T)`` arrays.

    Parameters
    ----------
    sims    : list of simulations, see :mod:`scales.dit.data`.
    cfg     : :class:`Config`; ``scales/dit/config.py`` holds the defaults.
    groups  : one label per simulation used for the train/val split.  Use the
              scenario (or the parent run for branched scenarios) so that
              ensemble members of the same scenario never straddle the split.
    esm_ids : one integer per simulation if you train on several ESMs; also set
              ``cfg.model.n_esm``.

    Returns a dict with ``best_path`` (use this for inference), ``path``
    (``last.pt``), ``best_val`` and ``best_step``.
    """
    cfg = (cfg or Config()).finalize()
    device = torch.device(cfg.train.device if torch.cuda.is_available()
                          or cfg.train.device == "cpu" else "cpu")
    if verbose:
        print(f"[device] {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
              + (" -- REQUESTED CUDA BUT RUNNING ON CPU"
                 if cfg.train.device.startswith("cuda") and device.type != "cuda" else ""))

    check_data(sims, cfg.data, verbose=verbose)
    normalizer = Normalizer.fit(sims, cfg.data)

    tr_idx, va_idx = group_split(len(sims), groups, cfg.data.val_fraction, cfg.data.seed)
    if verbose:
        print(f"[split] {len(tr_idx)} train sims / {len(va_idx)} val sims")

    ds_tr = CropDataset(sims, normalizer, cfg, tr_idx, esm_ids, train=True)
    ds_va = (CropDataset(sims, normalizer, cfg, va_idx, esm_ids, train=False)
             if va_idx else None)
    if verbose:
        n_eff = sum(np.asarray(sims[i]).shape[1] for i in tr_idx) // cfg.data.window
        print(f"[data] {len(ds_tr)} train crops"
              + (f" / {len(ds_va)} val crops" if ds_va else ""))
        print(f"[data] crops overlap heavily; effective independent samples "
              f"~= sum(T)/window = {n_eff}. Keep the model small if this is "
              f"below ~1e4.")

    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.train.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.train.num_workers > 0,
    )
    dl_va = make_val_loader(ds_va, cfg) if ds_va else None

    model = build_model(cfg, n_gmt_features(cfg)).to(device)
    if verbose:
        print(f"[model] {model.n_params()/1e6:.2f} M parameters")
    diffusion = Diffusion(cfg.diffusion.n_train_steps, cfg.diffusion.schedule).to(device)
    ema = EMA(model, cfg.train.ema_decay)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr,
        betas=tuple(cfg.train.betas), weight_decay=cfg.train.weight_decay,
    )
    use_amp = cfg.train.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.train.out_dir, "config.json"))
    last_path = os.path.join(cfg.train.out_dir, "last.pt")
    best_path = os.path.join(cfg.train.out_dir, "best.pt")

    best_val, best_step, n_since_best = float("inf"), -1, 0
    n_skipped = 0
    step, t0, running, gnorms = 0, time.time(), [], []
    stop_reason = None
    model.train()

    while step < cfg.train.max_steps and stop_reason is None:
        for batch in dl_tr:
            if step >= cfg.train.max_steps or stop_reason is not None:
                break
            batch = _to_device(batch, device)
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step, cfg)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = diffusion.loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            if cfg.train.skip_nonfinite_grads and not torch.isfinite(gn):
                # one bad batch should not be allowed to move the weights
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()
                ema.update(model)
                gnorms.append(gn.item())

            running.append(loss.detach().item())
            step += 1

            if verbose and step % cfg.train.log_every == 0:
                dt = time.time() - t0
                print(f"step {step:>7d}  loss {np.mean(running):.4f}  "
                      f"lr {lr_at(step, cfg):.2e}  "
                      f"|g| mean {np.mean(gnorms or [0]):.2f} max "
                      f"{np.max(gnorms or [0]):.2f}  "
                      f"{cfg.train.log_every/dt:.1f} it/s"
                      + (f"  [{n_skipped} skipped]" if n_skipped else ""), flush=True)
                running, gnorms, t0 = [], [], time.time()

            if dl_va is not None and step % cfg.train.val_every == 0:
                vl = validate(model, diffusion, dl_va, device)
                improved = vl < best_val - 1e-5
                if improved:
                    best_val, best_step, n_since_best = vl, step, 0
                    if cfg.train.save_best:
                        save_checkpoint(best_path, model, ema, normalizer, cfg, step,
                                        extra={"val_loss": vl})
                else:
                    n_since_best += 1
                if verbose:
                    print(f"step {step:>7d}  VAL loss {vl:.4f}   "
                          f"best {best_val:.4f} @ {best_step}"
                          + ("  *" if improved else f"  ({n_since_best} since best)"),
                          flush=True)

                if cfg.train.spike_abort_ratio and vl > best_val * cfg.train.spike_abort_ratio:
                    stop_reason = (
                        f"validation loss {vl:.4f} exceeded {cfg.train.spike_abort_ratio:g}x "
                        f"the best ({best_val:.4f} @ step {best_step}) -- training has "
                        f"collapsed, aborting"
                    )
                elif (cfg.train.early_stop_patience
                      and n_since_best >= cfg.train.early_stop_patience):
                    stop_reason = (
                        f"no improvement for {n_since_best} validations "
                        f"({n_since_best * cfg.train.val_every} steps); best "
                        f"{best_val:.4f} @ step {best_step}"
                    )
                t0 = time.time()

            if step % cfg.train.ckpt_every == 0:
                save_checkpoint(last_path, model, ema, normalizer, cfg, step)

    save_checkpoint(last_path, model, ema, normalizer, cfg, step)

    have_best = cfg.train.save_best and os.path.exists(best_path)
    if verbose:
        if stop_reason:
            print(f"[stop] {stop_reason}")
        if n_skipped:
            print(f"[warn] skipped {n_skipped} step(s) with non-finite gradients")
        print(f"[done] last -> {last_path}  (step {step})")
        if have_best:
            print(f"[done] best -> {best_path}  (step {best_step}, val {best_val:.4f})")
            if best_step < 0.5 * step:
                print(f"[note] the best model is from step {best_step} of {step} -- "
                      f"you are training well past the optimum. Consider "
                      f"max_steps ~= {int(best_step * 1.5)}.")
        else:
            print("[note] no validation set, so no best.pt -- last.pt is all you have.")

    return {
        "path": last_path,
        "best_path": best_path if have_best else last_path,
        "best_val": best_val if have_best else None,
        "best_step": best_step if have_best else step,
        "stop_reason": stop_reason,
        "model": model, "ema": ema, "normalizer": normalizer, "config": cfg,
    }
