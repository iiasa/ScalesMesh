"""Gaussian diffusion with v-prediction, cosine schedule, masked loss.

Notes on choices
----------------
* **v-prediction** (Salimans & Ho) + cosine schedule is markedly more stable
  than eps-prediction for unbounded, non-image data and behaves well with few
  sampling steps.
* **No classifier-free guidance.**  CFG with w > 1 systematically contracts
  the sample distribution.  For an image generator that reads as "higher
  quality"; for an ensemble emulator it silently destroys the ensemble spread,
  which is the whole product.  Diversity here comes from the initial noise and
  from ancestral sampling (``eta = 1``).
* **Masked loss.**  The clean prefix is given to the network as conditioning,
  so predicting it back is free; the loss is averaged only over the positions
  that actually have to be generated.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def cosine_alpha_bar(n_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:].clamp(1e-8, 1.0 - 1e-8).float()   # alpha_bar[0..T-1]


def linear_alpha_bar(n_steps: int) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, n_steps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0).clamp(1e-8, 1 - 1e-8).float()


class Diffusion:
    def __init__(self, n_steps: int = 1000, schedule: str = "cosine"):
        self.n_steps = n_steps
        if schedule == "cosine":
            ab = cosine_alpha_bar(n_steps)
        elif schedule == "linear":
            ab = linear_alpha_bar(n_steps)
        else:
            raise ValueError(schedule)
        self.alpha_bar = ab
        self.sqrt_ab = ab.sqrt()
        self.sqrt_1mab = (1.0 - ab).sqrt()

    def to(self, device) -> Diffusion:
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_ab = self.sqrt_ab.to(device)
        self.sqrt_1mab = self.sqrt_1mab.to(device)
        return self

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _bcast(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return v.view(-1, *([1] * (ref.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self._bcast(self.sqrt_ab[t], x0)
        b = self._bcast(self.sqrt_1mab[t], x0)
        return a * x0 + b * noise

    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self._bcast(self.sqrt_ab[t], x0)
        b = self._bcast(self.sqrt_1mab[t], x0)
        return a * noise - b * x0

    def x0_from_v(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self._bcast(self.sqrt_ab[t], x_t)
        b = self._bcast(self.sqrt_1mab[t], x_t)
        return a * x_t - b * v

    def eps_from_v(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self._bcast(self.sqrt_ab[t], x_t)
        b = self._bcast(self.sqrt_1mab[t], x_t)
        return a * v + b * x_t

    # -- training ----------------------------------------------------------
    def loss(self, model, batch: dict, generator: torch.Generator | None = None) -> torch.Tensor:
        x0 = batch["x0"]
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.n_steps, (B,), device=device, generator=generator)
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=x0.dtype)
        x_t = self.q_sample(x0, t, noise)

        # inside the clean prefix the network sees the true values anyway
        keep = batch["ctx_mask"]                                  # (B, W), 1 = context
        x_t = torch.where(keep.unsqueeze(1).bool(), x0, x_t)

        v_pred = model(
            x_t, t, batch["ctx"], batch["ctx_mask"], batch["gmt"],
            batch["end_year"], batch["gmt_feats"], batch["month_idx"],
            batch.get("esm_id"),
        )
        v_true = self.v_target(x0, noise, t)

        w = (1.0 - keep).unsqueeze(1)                              # loss on generated part
        se = (v_pred - v_true) ** 2 * w
        return se.sum() / w.expand_as(se).sum().clamp(min=1.0)

    # -- sampling ----------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        cond: dict,
        n_steps: int = 100,
        eta: float = 1.0,
        x0_clip: float | None = None,
        generator: torch.Generator | None = None,
        device=None,
        x0_projector: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """DDIM / ancestral sampler.

        ``cond`` must contain ctx, ctx_mask, gmt, end_year, gmt_feats,
        month_idx and optionally esm_id, all already on ``device``.
        ``x0_projector(x0, month_idx)`` is an optional hard-constraint hook
        (see ``sample.py`` for the GMT-consistency projector).
        """
        device = device or cond["ctx"].device
        x = torch.randn(shape, device=device, generator=generator)

        ts = torch.linspace(self.n_steps - 1, 0, n_steps, device=device).round().long()
        keep = cond["ctx_mask"].unsqueeze(1).bool()

        for i in range(n_steps):
            t = ts[i]
            t_b = t.expand(shape[0])

            # keep the known prefix pinned to its true (noise-free) value
            x = torch.where(keep, cond["ctx"], x)

            v = model(
                x, t_b, cond["ctx"], cond["ctx_mask"], cond["gmt"],
                cond["end_year"], cond["gmt_feats"], cond["month_idx"],
                cond.get("esm_id"),
            )
            x0 = self.x0_from_v(x, v, t_b)
            if x0_clip is not None:
                x0 = x0.clamp(-x0_clip, x0_clip)
            if x0_projector is not None:
                x0 = x0_projector(x0, cond["month_idx"])
            eps = self.eps_from_v(x, v, t_b)

            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < n_steps else torch.tensor(
                1.0, device=device
            )

            sigma = eta * torch.sqrt(
                ((1 - ab_prev) / (1 - ab_t)).clamp(min=0) * (1 - ab_t / ab_prev).clamp(min=0)
            )
            dir_coef = torch.sqrt((1 - ab_prev - sigma ** 2).clamp(min=0))
            x = ab_prev.sqrt() * x0 + dir_coef * eps
            if i + 1 < n_steps and sigma > 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)

        return torch.where(keep, cond["ctx"], x)
