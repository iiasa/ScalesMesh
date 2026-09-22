"""Conditional diffusion / flow-matching super-resolution model for MESH.

A ViT/DiT-style transformer backbone learns a flow-matching model on HR
tas/pr fields conditioned on LR (region-mean) tokens: it self-attends over
HR patch tokens and cross-attends to LR region tokens produced by a graph
encoder.

Training: regress the flow-matching velocity field.
Sampling: integrate the learned ODE (Euler or Heun).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .mesh_dataset import build_tas_pr_dataset


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_cuda_devices() -> None:
    print("CUDA device count (visible):", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0, T-1], shape (B,)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -(math.log(10000.0) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLNScaleShift(nn.Module):
    """AdaLN: learn per-token scale/shift from a conditioning vector."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D), cond_vec: (B,cond_dim)
        scale, shift = self.proj(cond_vec).chunk(2, dim=-1)  # each (B,D)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


# ----------------------------
# Tokenizers (patchify/unpatchify)
# ----------------------------
def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B,C,H,W) -> (B, N, C*P*P): square (P,P,C) patches, flattened."""
    B, C, H, W = x.shape
    P = patch_size
    return x.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 3, 5, 1).reshape(B, (H // P) * (W // P), P * P * C)


def unpatchify(x: torch.Tensor, patch_size: int, img_hw: tuple[int, int], out_ch: int = 1) -> torch.Tensor:
    """(B, N, D) with D = C*P*P -> (B,C,H,W)."""
    B = x.shape[0]
    H, W = img_hw
    P = patch_size
    return x.reshape(B, H // P, W // P, P, P, out_ch).permute(0, 5, 1, 3, 2, 4).reshape(B, out_ch, H, W)


# ----------------------------
# Attention blocks
# ----------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    """SDPA-based self-attention with QK-RMSNorm for bf16 stability."""

    def __init__(self, dim: int, heads: int, drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, Dh)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """SDPA-based cross-attention with QK-RMSNorm. q from HR tokens, kv from LR/region tokens."""

    def __init__(self, dim_q: int, dim_kv: int, heads: int, drop: float = 0.0):
        super().__init__()
        assert dim_q % heads == 0
        self.heads = heads
        self.head_dim = dim_q // heads
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.kv_proj = nn.Linear(dim_kv, 2 * dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.drop = drop

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, Dq = q.shape
        Nkv = kv.shape[1]
        q = self.q_proj(q).reshape(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv).reshape(B, Nkv, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # each (B, H, Nkv, Dh)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, Nq, Dq)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Self-attn on HR tokens + cross-attn over LR cond tokens, with AdaLN conditioning."""

    def __init__(self, dim: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.adaln1 = AdaLNScaleShift(dim, cond_dim)
        self.self_attn = SelfAttention(dim, heads, drop)
        self.adaln2 = AdaLNScaleShift(dim, cond_dim)
        self.cross_attn = CrossAttention(dim, dim, heads, drop)
        self.adaln3 = AdaLNScaleShift(dim, cond_dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x_hr: torch.Tensor, cond_vec: torch.Tensor, tokens_lr: torch.Tensor) -> torch.Tensor:
        # x_hr: (B,N_hr,C), tokens_lr: (B,N_lr,C_lr), cond_vec: (B,cond_dim)
        x = x_hr + self.self_attn(self.adaln1(x_hr, cond_vec))
        x = x + self.cross_attn(self.adaln2(x, cond_vec), tokens_lr)
        x = x + self.mlp(self.adaln3(x, cond_vec))
        return x


class GraphEncoder(nn.Module):
    """Encodes per-region features into tokens via a plain Transformer encoder."""

    def __init__(self, in_dim: int, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, node_feats: torch.Tensor, adj_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        node_feats: (B, N, in_dim) input features for N regions
        adj_mask: (B, N, N) attention mask where True = block attention (optional)
        Returns tokens: (B, N, d_model)
        """
        x = self.in_proj(node_feats)  # (B, N, d_model)
        return self.encoder(x, mask=adj_mask)  # (B, N, d_model)


# ----------------------------
# DiT-like SR model
# ----------------------------
class ViTCondDiffusionSR(nn.Module):
    """Predicts the flow-matching velocity on HR patches given timestep t and LR conditioning tokens."""

    def __init__(
        self,
        varmax: float | tuple[float, ...] = 350,
        varmin: float | tuple[float, ...] = 100,
        in_ch: int = 1,
        dim: int = 192,
        depth: int = 8,
        heads: int = 6,
        patch: int = 4,
        lr_patch: int = 3,
        hr_hw: tuple[int, int] = (28, 28),
        lr_regions: int = 58,
    ):
        super().__init__()
        self.register_buffer("varmax", torch.tensor(varmax))
        self.register_buffer("varmin", torch.tensor(varmin))
        self.hr_hw = hr_hw
        self.in_ch = in_ch
        self.patch = patch
        self.hr_tokens = (hr_hw[0] // patch) * (hr_hw[1] // patch)
        self.lr_patch = lr_patch
        self.lr_tokens = lr_regions

        # Embeddings
        self.hr_in = nn.Linear(in_features=in_ch * patch * patch, out_features=dim)
        self.pos_hr = nn.Parameter(torch.zeros(1, self.hr_tokens, dim))
        self.pos_lr = nn.Parameter(torch.zeros(1, self.lr_tokens, dim))
        self.graph_encoder = GraphEncoder(in_dim=in_ch, d_model=dim)

        # Time + global cond vector
        self.time_emb = SinusoidalTimeEmbedding(dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        # Summarize LR tokens into a single vector
        self.lr_summary = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        # Transformer blocks. cond_dim = dim*2: the time embedding and the LR
        # summary are each of size dim and are concatenated into one cond vector.
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, heads=heads, cond_dim=dim * 2, mlp_ratio=2.0) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # Output head: predict velocity on patchified HR (same shape as input patches)
        self.out = nn.Linear(dim, in_ch * patch * patch)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.pos_hr, std=0.02)
        nn.init.trunc_normal_(self.pos_lr, std=0.02)

    def encode_lr_tokens(self, lr_img: torch.Tensor) -> torch.Tensor:
        # lr_img: (B, C, N_regions) -> tokens from graph encoding, N_lr = N_regions
        return self.graph_encoder(lr_img.transpose(1, 2)) + self.pos_lr

    def forward(self, x_hr_noisy: torch.Tensor, t: torch.Tensor, lr_img: torch.Tensor) -> torch.Tensor:
        """
        x_hr_noisy: (B, C, H, W) noised/interpolated HR field
        t: (B,) timesteps in [0, 1]
        lr_img: (B, C, N_regions) LR conditioning
        Returns predicted velocity, same shape as x_hr_noisy.
        """
        hr_tok = patchify(x_hr_noisy, self.patch)  # (B,N_hr,P*P*C)
        hr_tok = self.hr_in(hr_tok) + self.pos_hr  # (B,N_hr,dim)

        lr_tok = self.encode_lr_tokens(lr_img)  # (B,N_lr,dim)
        lr_summary = self.lr_summary(lr_tok.mean(dim=1))  # (B,dim)

        # Scale t from [0,1] to [0,1000] for the sinusoidal frequencies.
        te = self.time_mlp(self.time_emb(t * 1000))  # (B,dim)
        cond_vec = torch.cat([te, lr_summary], dim=-1)  # (B, 2*dim)

        x = hr_tok
        for blk in self.blocks:
            x = blk(x, cond_vec, lr_tok)

        x = self.norm(x)
        v = self.out(x)  # (B,N_hr,P*P*C)
        return unpatchify(v, self.patch, self.hr_hw, out_ch=self.in_ch)


# ----------------------------
# EMA
# ----------------------------
class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Keeps a deep-copied *unwrapped* (no DDP, no torch.compile) shadow module
    whose parameters are an EMA of the training model's. Use `self.ema_model`
    directly for evaluation / sampling.

    Update rule: p_ema <- d * p_ema + (1 - d) * p_train.
    Decay ramps in early via `d = min(decay, (1 + n) / (10 + n))` so the EMA
    isn't dominated by the (random) init in the first few hundred steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        base = self._unwrap(model)
        self.ema_model = copy.deepcopy(base).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        base = model
        # Strip torch.compile then DDP (or either order, robustly).
        for _ in range(4):
            nxt = getattr(base, "_orig_mod", None)
            if nxt is None:
                nxt = getattr(base, "module", None) if isinstance(base, DDP) else None
            if nxt is None:
                break
            base = nxt
        return base

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        src = self._unwrap(model)
        for p_ema, p in zip(self.ema_model.parameters(), src.parameters(), strict=True):
            p_ema.mul_(d).add_(p.data, alpha=1.0 - d)
        # Buffers (varmax/varmin) are constants — copy verbatim, don't average.
        for b_ema, b in zip(self.ema_model.buffers(), src.buffers(), strict=True):
            b_ema.copy_(b)

    def state_dict(self) -> dict:
        return {"ema_model": self.ema_model.state_dict(), "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, sd: dict) -> None:
        self.ema_model.load_state_dict(sd["ema_model"])
        self.num_updates = sd.get("num_updates", 0)
        self.decay = sd.get("decay", self.decay)


# ----------------------------
# Training loop (flow matching)
# ----------------------------
def train(
    model: nn.Module,
    fm,
    train_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_path: str,
    epochs: int = 10,
    lr: float = 2e-4,
    device: torch.device = "cuda",
    train_sampler: DistributedSampler | None = None,
    ema: ModelEMA | None = None,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # Normalise keys so checkpoints from torch.compile-wrapped runs
        # (keys prefixed with "_orig_mod.") load into non-compiled models and vice versa.
        raw_state = checkpoint["model_state_dict"]
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        target.load_state_dict(cleaned)
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema is not None:
            if "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
            else:
                # Old checkpoint without EMA state: seed the EMA shadow from the
                # just-loaded training weights (better than leaving it at random init).
                # Reset num_updates so the decay-warmup starts from scratch.
                src = ema._unwrap(model)
                for p_ema, p in zip(ema.ema_model.parameters(), src.parameters(), strict=True):
                    p_ema.data.copy_(p.data)
                for b_ema, b in zip(ema.ema_model.buffers(), src.buffers(), strict=True):
                    b_ema.copy_(b)
                ema.num_updates = 0
                print("No EMA state in checkpoint — seeded EMA shadow from loaded training weights.")
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for ep in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        loss_sum, n = 0.0, 0
        for lr_img, hr_img in train_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            x0 = torch.randn_like(hr_img)
            t = torch.rand(hr_img.size(0), device=device)
            x_t = fm.sample_xt(x0, hr_img, t)
            v_target = fm.target_velocity(x0, hr_img, t)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                v_pred = model(x_t, t, lr_img)
            # Compute the loss in fp32 outside autocast — bf16 MSE reduction quickly
            # loses precision once the loss becomes small and can produce NaN/Inf grads.
            loss = F.mse_loss(v_pred.float(), v_target.float())

            opt.zero_grad()
            loss.backward()
            unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if unused and dist.get_rank() == 0:
                print("Unused params:", unused[:50])

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Skip the step if anything went non-finite; keeps a single bad batch
            # from poisoning the weights (which is what turns loss→NaN forever).
            if torch.isfinite(loss) and torch.isfinite(gnorm):
                opt.step()
                if ema is not None:
                    ema.update(model)
            elif dist.get_rank() == 0:
                print(f"[ep {ep}] skipping step: loss={loss.item()} grad_norm={gnorm.item()}")

            loss_sum += loss.item()
            n += 1

        if ep % 10 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            ckpt = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss.item(),
            }
            if ema is not None:
                ckpt["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt, checkpoint_path)

        model.eval()
        with torch.no_grad():
            vloss_sum, eloss_sum, m = 0.0, 0.0, 0
            for lr_img, hr_img in test_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                x0 = torch.randn_like(hr_img)
                t = torch.rand(hr_img.size(0), device=device)
                x_t = fm.sample_xt(x0, hr_img, t)
                v_target = fm.target_velocity(x0, hr_img, t)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    v_pred = model(x_t, t, lr_img)
                vloss = F.mse_loss(v_pred.float(), v_target.float())
                vloss_sum += vloss.item()
                # Score the EMA shadow on the *same* (x0, t, batch) so the two
                # numbers are directly comparable — a random redraw would add
                # enough variance to hide a real gap either way.
                if ema is not None:
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
                    ):
                        v_ema = ema.ema_model(x_t, t, lr_img)
                    eloss_sum += F.mse_loss(v_ema.float(), v_target.float()).item()
                m += 1
        msg = f"Epoch {ep:02d} | Train v-MSE: {loss_sum / n:.5f} | Val v-MSE: {vloss_sum / m:.5f}"
        if ema is not None:
            decay = min(ema.decay, (1 + ema.num_updates) / (10 + ema.num_updates))
            msg += f" | EMA val v-MSE: {eloss_sum / m:.5f} (d={decay:.6f}, n={ema.num_updates})"
        print(msg)


# ----------------------------
# Flow matching
# ----------------------------
class FlowMatching:
    """
    Trigonometric conditional flow matching.

    x0 = noise,  x1 = daily fields

    x_t  = cos(t·π/2)·x0 + sin(t·π/2)·x1
    v_t  = dx_t/dt = (π/2)·(-sin(t·π/2)·x0 + cos(t·π/2)·x1)
    """

    def __init__(self, hr_hw: tuple[int, int], channels: int = 2, device: str = "cuda"):
        self.device = device
        self.H, self.W = hr_hw
        self.C = channels

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x0: noise (B,C,H,W); x1: data (B,C,H,W); t: (B,) in [0, 1]."""
        t_view = t.view(-1, 1, 1, 1)
        alpha = torch.cos(t_view * math.pi / 2)
        beta = torch.sin(t_view * math.pi / 2)
        return alpha * x0 + beta * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Analytic derivative of the interpolation path at time t."""
        t_view = t.view(-1, 1, 1, 1)
        return (math.pi / 2) * (-torch.sin(t_view * math.pi / 2) * x0 + torch.cos(t_view * math.pi / 2) * x1)

    @torch.no_grad()
    def sample(self, model: nn.Module, lr: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """Generate HR fields from an LR region-mean conditioning tensor."""
        B = lr.shape[0]
        device = lr.device
        x = torch.randn(B, self.C, self.H, self.W, device=device)
        dt = 1.0 / n_steps

        for k in range(n_steps):
            t = torch.full((B,), k / n_steps, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                v = model(x, t, lr)
            x = x + dt * v.float()  # Euler ODE step

        return x

    @torch.no_grad()
    def sample_heun(self, model: nn.Module, lr: torch.Tensor, n_steps: int = 15) -> torch.Tensor:
        """
        Heun (2nd-order) ODE solver. 2 NFE per step, ~O(dt^3) local error vs Euler's O(dt^2).
        Cost parity: n_steps=25 here ≈ 50 Euler NFE. Try 10-15 for a speed win.

          k1 = v(x, t)
          x* = x + dt·k1
          k2 = v(x*, t+dt)
          x  = x + (dt/2)·(k1 + k2)
        """
        B = lr.shape[0]
        device = lr.device
        x = torch.randn(B, self.C, self.H, self.W, device=device)
        dt = 1.0 / n_steps

        for k in range(n_steps):
            t1 = torch.full((B,), k / n_steps, device=device)
            t2 = torch.full((B,), (k + 1) / n_steps, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                k1 = model(x, t1, lr).float()
            x_pred = x + dt * k1

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                k2 = model(x_pred, t2, lr).float()
            x = x + 0.5 * dt * (k1 + k2)

        return x


class FlowMatchingLinear:
    """
    Straight-line conditional flow matching.

    x0 = noise,  x1 = daily fields

    x_t = (1 - t)·x0 + t·x1
    v   = dx_t/dt = x1 - x0   (constant — no t dependency)

    Faster early convergence than trigonometric because the regression
    target is the same at every timestep.
    """

    def __init__(self, hr_hw: tuple[int, int], channels: int = 2, device: str = "cuda"):
        self.device = device
        self.H, self.W = hr_hw
        self.C = channels

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_view = t.view(-1, 1, 1, 1)
        return (1.0 - t_view) * x0 + t_view * x1

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor, _t: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    @torch.no_grad()
    def sample(self, model: nn.Module, lr: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        B = lr.shape[0]
        device = lr.device
        x = torch.randn(B, self.C, self.H, self.W, device=device)
        dt = 1.0 / n_steps

        for k in range(n_steps):
            t = torch.full((B,), k / n_steps, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                v = model(x, t, lr)
            x = x + dt * v.float()

        return x

    @torch.no_grad()
    def sample_heun(self, model: nn.Module, lr: torch.Tensor, n_steps: int = 15) -> torch.Tensor:
        """
        Heun (2nd-order) ODE solver. 2 NFE per step. See FlowMatching.sample_heun.

        Note: for the straight-line schedule the true velocity is constant along the path,
        so Heun and Euler should agree in the limit of a well-trained model — but with
        a finite-error network Heun still tends to give cleaner trajectories.
        """
        B = lr.shape[0]
        device = lr.device
        x = torch.randn(B, self.C, self.H, self.W, device=device)
        dt = 1.0 / n_steps

        for k in range(n_steps):
            t1 = torch.full((B,), k / n_steps, device=device)
            t2 = torch.full((B,), (k + 1) / n_steps, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                k1 = model(x, t1, lr).float()
            x_pred = x + dt * k1

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                k2 = model(x_pred, t2, lr).float()
            x = x + 0.5 * dt * (k1 + k2)

        return x


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MESH for downscaling")
    parser.add_argument("--tas-data-path", type=str, required=True, help="Directory containing the tas NetCDF files")
    parser.add_argument("--pr-data-path", type=str, required=True, help="Directory containing the pr NetCDF files")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend)
    if use_cuda:
        torch.cuda.set_device(local_rank)
        torch.set_float32_matmul_precision("high")  # TF32 for non-autocast matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log_cuda_devices()

    set_seed(0)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    batch_size = 128
    epochs = 200
    lr = 2e-4
    patch = 3
    lr_patch = 3
    dim = 192
    depth = 16
    heads = 6
    hr_hw = (72, 144)

    # Data
    ds, preppers = build_tas_pr_dataset(args.tas_data_path, args.pr_data_path, hr_scale=1)
    regions_dim = preppers["tas"].get_n_regions()
    vmax_tas, vmin_tas = preppers["tas"].get_ds_max_min()
    vmax_pr, vmin_pr = preppers["pr"].get_ds_max_min()
    print("Loaded dataset")

    n_ds = len(ds)
    train_ds = Subset(ds, list(range(0, (n_ds * 75) // 100)))
    test_ds = Subset(ds, list(range((n_ds * 75) // 100, n_ds)))
    print("Split off train dataset")

    use_distributed = dist.is_initialized() and dist.get_world_size() > 1
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if use_distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, sampler=test_sampler, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    print("built data loaders")

    run_dir = f"outputs_DiffusionTransformer/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    checkpoint_dir = f"{run_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_flow.pt"

    # Model + schedule
    model = ViTCondDiffusionSR(
        dim=dim, depth=depth, heads=heads, patch=patch, lr_patch=lr_patch, hr_hw=hr_hw,
        lr_regions=regions_dim, varmax=(vmax_tas, vmax_pr), varmin=(vmin_tas, vmin_pr), in_ch=2,
    ).to(device)
    ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if use_cuda else {}
    model = DDP(model, **ddp_kwargs)

    if use_cuda:
        # Compile after DDP wrap; inductor handles the allreduce graph break.
        model = torch.compile(model)
    print("Built the model")

    fm = FlowMatchingLinear(hr_hw=hr_hw, channels=2, device=device)
    ema = ModelEMA(model, decay=0.9999)

    train(
        model, fm, train_loader, test_loader, checkpoint_path=checkpoint_path,
        epochs=epochs, lr=lr, device=device, train_sampler=train_sampler, ema=ema,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        # Unwrapped keys (no "_orig_mod."/"module." prefixes) so both files
        # load into the same eval code. EMA is the model to use for
        # inference; the raw weights are kept for comparison.
        torch.save(ema.ema_model.state_dict(), f"{run_dir}/model_flow_ema.pt")
        torch.save(ModelEMA._unwrap(model).state_dict(), f"{run_dir}/model_flow_out.pt")


if __name__ == "__main__":
    main()
