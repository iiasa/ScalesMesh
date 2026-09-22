"""Long-scenario inference by context-conditioned outpainting.

Given a monthly GMT trajectory for a new scenario (any length, multiple of 12),
walk 96-month windows with a 60-month stride.  Window 0 is generated
unconditionally (apart from the GMT embedding); every later window receives the
last 36 already-generated months as a *clean prefix*, exactly the setting the
model was trained on.  All ensemble members are generated as one batch, so
member count costs almost nothing in wall time.

Two things worth knowing:

* The GMT embedding re-anchors the model at every window, which is the main
  defence against autoregressive drift over hundreds of windows.  Still, check
  for drift explicitly (``evaluate.trend_drift``) before trusting a 250-year
  emulation.
* Because the window is 8 years long, variability on timescales much longer
  than ~8 years is only represented insofar as it is carried by the 36-month
  overlap and the GMT conditioning.  If low-frequency (multidecadal) spread
  comes out too weak in evaluation, lengthen ``cfg.data.window`` (240 months is
  still cheap) or go to a two-stage annual->monthly cascade.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from common.zenodo import download_from_zenodo

from .config import Config
from .data import Normalizer, gmt_path_features, n_gmt_features
from .diffusion import Diffusion
from .model import build_model


class ScenarioSampler:
    def __init__(self, model, diffusion: Diffusion, normalizer: Normalizer,
                 cfg: Config, device="cpu"):
        self.model = model.eval()
        self.diffusion = diffusion
        self.norm = normalizer
        self.cfg = cfg
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda", use_ema: bool = True):
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        ck = torch.load(path, map_location=dev, weights_only=False)
        cfg = Config.from_dict(ck["config"])
        model = build_model(cfg, n_gmt_features(cfg)).to(dev)
        sd = ck["ema"] if (use_ema and ck.get("ema") is not None) else ck["model"]
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        diffusion = Diffusion(cfg.diffusion.n_train_steps, cfg.diffusion.schedule).to(dev)
        normalizer = Normalizer.from_state_dict(ck["normalizer"])
        return cls(model, diffusion, normalizer, cfg, dev)

    # ------------------------------------------------------------------
    @classmethod
    def from_zenodo(
        cls,
        record: str | int,
        filename: str | None = None,
        device: str = "cuda",
        use_ema: bool = True,
        dest_dir: str | None = None,
        force: bool = False,
    ):
        """Download a checkpoint from a Zenodo record and build a sampler from it.

        ``record`` is a Zenodo record ID, DOI or URL; ``filename`` selects
        which file to fetch if the record holds more than one. The download
        is cached locally (under ``~/.cache/scalesmesh/zenodo`` by default)
        and re-verified against its published checksum on later calls, so
        repeated calls with the same record are cheap.
        """
        path = download_from_zenodo(record, filename=filename, dest_dir=dest_dir, force=force)
        return cls.from_checkpoint(str(path), device=device, use_ema=use_ema)

    # ------------------------------------------------------------------
    def _gmt_projector(self, area_weights, gmt_target_window, n_tas):
        """Hard-constrain the area-weighted mean of generated tas to the
        prescribed GMT, by orthogonal projection of x0 at every denoising step.

        ``area_weights``: (n_tas,) weights over the tas regions -- they should
        sum to 1 and cover the globe for this to be physically meaningful.
        ``gmt_target_window``: (W,) prescribed GMT for this window, in the same
        units/baseline as the tas rows.
        """
        w = torch.as_tensor(area_weights, dtype=torch.float32, device=self.device)
        tgt = torch.as_tensor(gmt_target_window, dtype=torch.float32, device=self.device)
        wsq = float((w ** 2).sum())
        mu = torch.as_tensor(self.norm.mu[:n_tas], dtype=torch.float32, device=self.device)
        sd = torch.as_tensor(self.norm.sd[:n_tas], dtype=torch.float32, device=self.device)

        def proj(x0: torch.Tensor, month_idx: torch.Tensor) -> torch.Tensor:
            m = month_idx[0]                              # (W,), identical across batch
            mu_w, sd_w = mu[:, m], sd[:, m]               # (n_tas, W)
            tas = x0[:, :n_tas] * sd_w + mu_w             # physical units
            cur = (w[None, :, None] * tas).sum(dim=1)     # (B, W)
            corr = (tgt[None] - cur) / wsq                # (B, W)
            tas = tas + w[None, :, None] * corr[:, None, :]
            out = x0.clone()
            out[:, :n_tas] = (tas - mu_w) / sd_w
            return out

        return proj

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        gmt_monthly: Sequence[float],
        n_members: int = 1,
        stride: int | None = None,
        steps: int | None = None,
        eta: float | None = None,
        esm_id: int = 0,
        seed: int | None = None,
        area_weights: np.ndarray | None = None,
        progress: bool = True,
    ) -> np.ndarray:
        """Emulate a scenario.

        Returns ``(n_members, n_channels, N)`` in the original physical units,
        rows ordered exactly like rows 1..116 of your input arrays.
        """
        cfg, d = self.cfg, self.cfg.data
        W = d.window
        stride = stride or (W - 36)
        steps = steps or cfg.diffusion.sample_steps
        eta = cfg.diffusion.eta if eta is None else eta

        g_monthly = np.asarray(gmt_monthly, dtype=np.float64).reshape(-1)
        N = g_monthly.size
        if N % 12 != 0:
            raise ValueError(f"scenario length {N} must be a multiple of 12")
        if N < W:
            raise ValueError(f"scenario length {N} shorter than window {W}")
        if stride % 12 != 0 and d.january_start:
            raise ValueError("stride must be a multiple of 12 for January-aligned windows")

        gy = g_monthly.reshape(-1, 12)
        if not np.allclose(gy, gy[:, :1], atol=1e-8):
            print("[warn] GMT is not constant within calendar years; using the "
                  "January value of each year.")
        gmt_annual = self.norm.transform_gmt(g_monthly[::12])          # (Y,)
        Y = gmt_annual.size

        C = d.n_channels
        M = n_members
        out = np.zeros((M, C, N), dtype=np.float32)      # normalised space

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()

        gmt_t = torch.from_numpy(gmt_annual).to(self.device).unsqueeze(0).expand(M, Y)
        esm_t = torch.full((M,), int(esm_id), dtype=torch.long, device=self.device)

        starts = list(range(0, N - W + 1, stride))
        if starts[-1] != N - W:
            starts.append(N - W)

        filled = 0
        for wi, start in enumerate(starts):
            L = int(np.clip(filled - start, 0, W - 12))
            if start + W <= filled:
                continue
            if L > 0 and L not in tuple(d.context_lengths):
                print(f"[warn] context length {L} was not seen in training "
                      f"(trained on {tuple(d.context_lengths)}); expect seams.")

            ctx = torch.zeros(M, C, W, device=self.device)
            ctx_mask = torch.zeros(M, W, device=self.device)
            if L > 0:
                ctx[:, :, :L] = torch.from_numpy(out[:, :, start : start + L]).to(self.device)
                ctx_mask[:, :L] = 1.0

            end_year = (start + W) // 12 - 1
            feats = gmt_path_features(
                gmt_annual, end_year, cfg.model.use_elapsed_time_feature
            )
            month_idx = ((np.arange(start, start + W) + self.norm.start_month) % 12)

            cond = {
                "ctx": ctx,
                "ctx_mask": ctx_mask,
                "gmt": gmt_t,
                "end_year": torch.full((M,), end_year, dtype=torch.long, device=self.device),
                "gmt_feats": torch.from_numpy(feats).to(self.device).unsqueeze(0).expand(M, -1),
                "month_idx": torch.from_numpy(month_idx).to(self.device).unsqueeze(0).expand(M, W),
                "esm_id": esm_t,
            }

            projector = None
            if area_weights is not None:
                projector = self._gmt_projector(
                    area_weights, g_monthly[start : start + W], d.n_tas
                )

            xw = self.diffusion.sample(
                self.model, (M, C, W), cond, n_steps=steps, eta=eta,
                x0_clip=cfg.diffusion.x0_clip, generator=gen,
                device=self.device, x0_projector=projector,
            )
            out[:, :, start + L : start + W] = xw[:, :, L:].float().cpu().numpy()
            filled = start + W

            if progress:
                print(f"\r[sample] window {wi+1}/{len(starts)} "
                      f"(months {start}-{start+W}, ctx={L})      ", end="", flush=True)
        if progress:
            print()

        # back to physical units, per member
        phys = np.stack(
            [self.norm.inverse_transform_targets(out[m], t0=0) for m in range(M)]
        )
        return phys.astype(np.float32)
