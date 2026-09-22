"""Network: causal GMT-history encoder + 1-D DiT denoiser.

Why a *1-D* DiT and not an image-style 2-D ViT
----------------------------------------------
The (region x time) matrix is not an image.  The time axis is
translation-invariant; the region axis is an arbitrary permutation of IPCC
regions.  A 2-D patch of size 4x4 would therefore impose a completely
spurious locality prior (mixing, say, Central Africa with Northern Europe
because they happen to be adjacent in the row index).

So: **one token per month**, with the full 116-vector of regions/variables as
the token's channel dimension.  Cross-region structure is then modelled by the
patch-embedding matrix and the residual stream, which have no false locality,
and cross-time structure by full self-attention over the 96 tokens.

If you later want explicit region reasoning, the natural extension is axial
attention: alternate attention over time tokens and attention over region
tokens with learned region embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def sinusoidal_positions(n: int, dim: int, device=None) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32, device=device)
    return timestep_embedding(pos, dim)


class RMSNorm(nn.Module):
    """RMS normalisation, computed in fp32 so it is safe under bf16 autocast."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dt)


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional QK normalisation.

    Unbounded QK logits are the mechanism behind attention entropy collapse:
    the softmax saturates toward one-hot, gradients through attention vanish,
    and training stalls with no recovery. Normalising q and k to unit RMS
    before the dot product bounds the logits at roughly +/- sqrt(head_dim)
    regardless of how large the projections grow, which makes that runaway
    self-limiting. Costs two vectors of parameters per attention layer and no
    measurable time.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0,
                 causal: bool = False, qk_norm: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, L, hd)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-LN transformer block with adaLN-Zero conditioning (Peebles & Xie)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float = 0.0,
                 qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, n_heads, dropout, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio, dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class PlainBlock(nn.Module):
    """Ordinary pre-LN block, used inside the (unconditioned) GMT encoder."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 causal: bool = True, qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, causal=causal, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------
# GMT history encoder
# --------------------------------------------------------------------------


class GMTEncoder(nn.Module):
    """Encodes the *full* GMT history up to a given year into a vector.

    The transformer is causal, so a single forward pass over an entire
    scenario yields a valid embedding at *every* year end.  That is what makes
    long-scenario inference cheap: one pass, then gather.

    Learned attention is combined with explicit path features
    (cumulative GMT, 10/50-yr trends, peak, overshoot depth), because the
    physical mechanism behind path dependence -- ocean heat uptake -- is a
    near-integral of the forcing, and an unaided learned encoder extrapolates
    poorly to scenario shapes (e.g. deep overshoots) it never saw in training.
    """

    def __init__(self, cfg: Config, n_feats: int):
        super().__init__()
        m = cfg.model
        d = m.gmt_d_model
        self.in_proj = nn.Linear(1, d)
        self.register_buffer(
            "pos", sinusoidal_positions(m.gmt_max_years, d), persistent=False
        )
        self.pos_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList([
            PlainBlock(d, m.gmt_heads, causal=True, qk_norm=m.qk_norm)
            for _ in range(m.gmt_depth)
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d + n_feats, m.cond_dim), nn.SiLU(),
            nn.Linear(m.cond_dim, m.cond_dim),
        )
        self.max_years = m.gmt_max_years

    def forward(
        self, gmt: torch.Tensor, end_year: torch.Tensor, feats: torch.Tensor
    ) -> torch.Tensor:
        """gmt: (B, Y) normalised annual GMT.  end_year: (B,).  feats: (B, F)."""
        B, Y = gmt.shape
        if Y > self.max_years:
            raise ValueError(
                f"GMT history of {Y} years exceeds gmt_max_years={self.max_years}"
            )
        h = self.in_proj(gmt.unsqueeze(-1)) + self.pos_proj(self.pos[:Y]).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        idx = end_year.clamp(0, Y - 1).view(B, 1, 1).expand(B, 1, h.shape[-1])
        h_end = h.gather(1, idx).squeeze(1)               # (B, d)
        return self.head(torch.cat([h_end, feats], dim=-1))


# --------------------------------------------------------------------------
# denoiser
# --------------------------------------------------------------------------


class ScalesDiT(nn.Module):
    """v-prediction denoiser for a (C, W) window of regional tas/pr."""

    def __init__(self, cfg: Config, n_gmt_feats: int):
        super().__init__()
        self.cfg = cfg
        C = cfg.data.n_channels
        d = cfg.model.d_model
        self.C, self.d = C, d

        # x_t, clean context, and the binary context mask
        self.in_proj = nn.Linear(2 * C + 1, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.model.max_window, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.month_emb = nn.Embedding(12, d)
        nn.init.normal_(self.month_emb.weight, std=0.02)

        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.gmt_encoder = GMTEncoder(cfg, n_gmt_feats)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.model.cond_dim, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.esm_emb = nn.Embedding(max(cfg.model.n_esm, 1), d)
        nn.init.zeros_(self.esm_emb.weight)

        self.blocks = nn.ModuleList([
            DiTBlock(d, cfg.model.n_heads, cfg.model.mlp_ratio, cfg.model.dropout,
                     qk_norm=cfg.model.qk_norm)
            for _ in range(cfg.model.depth)
        ])
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out_proj = nn.Linear(d, C)
        # zero-init the output path: the model starts as the identity residual
        for mod in (self.ada_out[1], self.out_proj):
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x_t: torch.Tensor,        # (B, C, W)
        t: torch.Tensor,          # (B,)
        ctx: torch.Tensor,        # (B, C, W) clean context, zero elsewhere
        ctx_mask: torch.Tensor,   # (B, W)
        gmt: torch.Tensor,        # (B, Y)
        end_year: torch.Tensor,   # (B,)
        gmt_feats: torch.Tensor,  # (B, F)
        month_idx: torch.Tensor,  # (B, W)
        esm_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, W = x_t.shape
        tok = torch.cat([x_t, ctx, ctx_mask.unsqueeze(1)], dim=1)   # (B, 2C+1, W)
        h = self.in_proj(tok.transpose(1, 2))                       # (B, W, d)
        h = h + self.pos_emb[:, :W] + self.month_emb(month_idx)

        c = self.t_mlp(timestep_embedding(t, self.d))
        c = c + self.cond_proj(self.gmt_encoder(gmt, end_year, gmt_feats))
        if esm_id is not None:
            c = c + self.esm_emb(esm_id)

        for blk in self.blocks:
            h = blk(h, c)

        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), shift, scale)
        return self.out_proj(h).transpose(1, 2)                     # (B, C, W)


def build_model(cfg: Config, n_gmt_feats: int) -> ScalesDiT:
    return ScalesDiT(cfg, n_gmt_feats)
