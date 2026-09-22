"""Data handling for the SCALES DiT emulator.

Input format
------------
A ``list`` of numpy arrays, one per simulation, each of shape ``(117, T)``
with ``T % 12 == 0`` and column 0 = January:

    row 0            global mean temperature (annual value, repeated 12x)
    rows 1 .. 57     monthly regional ``tas``   (57 IPCC regions)
    rows 58 .. 116   monthly regional ``pr``    (59 IPCC regions)

Simulations may have different lengths.

Key design points
-----------------
* **Per-(channel, calendar-month) standardisation.**  Without it the ``pr``
  channels (O(1e-5)) contribute ~1e-12 of the gradient relative to ``tas``
  (O(10)) and never train.  Standardising per calendar month additionally
  removes the climatological seasonal cycle, so the network spends its
  capacity on anomalies rather than on re-learning summer/winter.
* **Signed cube root for pr** before standardisation: monotone, exactly
  invertible, symmetric, and it tames the heavy right tail of precipitation.
* **GMT is annual.**  Row 0 is constant within each calendar year, so we
  encode it at annual resolution (12x cheaper, and honest about the
  information content).
* **Causal path-dependence features.**  Regional response depends on the
  *path* of GMT mostly through ocean heat uptake, so we hand the encoder
  explicit cumulative / rate / overshoot summaries instead of hoping a
  learned encoder extrapolates correctly to novel scenario shapes.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config, DataConfig

# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------


def signed_cbrt(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** (1.0 / 3.0)


def signed_cube(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** 3.0


N_GMT_FEATURES_BASE = 8


def gmt_path_features(
    gmt_annual_norm: np.ndarray, end_idx: int, use_elapsed: bool = True
) -> np.ndarray:
    """Causal summaries of the GMT path up to and including year ``end_idx``.

    All inputs/outputs are in *normalised* GMT units except the two features
    that are explicitly rescaled by 1/100.
    """
    g = gmt_annual_norm[: end_idx + 1]
    cur = g[-1]
    feats = [
        cur,                                  # current level
        g[-10:].mean(),                       # 10-yr mean
        cur - g[max(0, end_idx - 9)],         # 10-yr trend
        cur - g[max(0, end_idx - 49)],        # 50-yr trend
        g.mean(),                             # running mean since start
        g.sum() / 100.0,                      # cumulative GMT (heat-uptake proxy)
        g.max(),                              # peak so far
        cur - g.max(),                        # overshoot depth (<= 0)
    ]
    if use_elapsed:
        feats.append((end_idx + 1) / 100.0)   # elapsed years
    return np.asarray(feats, dtype=np.float32)


def n_gmt_features(cfg: Config) -> int:
    return N_GMT_FEATURES_BASE + int(cfg.model.use_elapsed_time_feature)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def check_data(sims: Sequence[np.ndarray], cfg: DataConfig, verbose: bool = True) -> None:
    """Sanity-check the raw simulation list and report magnitudes."""
    assert len(sims) > 0, "empty simulation list"
    tas_absmax, pr_absmax, gmt_span = [], [], []
    for i, s in enumerate(sims):
        s = np.asarray(s)
        if s.ndim != 2:
            raise ValueError(f"sim {i}: expected 2-D array, got shape {s.shape}")
        if s.shape[0] != cfg.n_rows:
            raise ValueError(
                f"sim {i}: expected {cfg.n_rows} rows "
                f"(1 GMT + {cfg.n_tas} tas + {cfg.n_pr} pr), got {s.shape[0]}"
            )
        if s.shape[1] % 12 != 0:
            raise ValueError(f"sim {i}: length {s.shape[1]} is not a multiple of 12")
        if s.shape[1] < cfg.window:
            raise ValueError(f"sim {i}: shorter ({s.shape[1]}) than window ({cfg.window})")
        g = s[cfg.gmt_row]
        gy = g.reshape(-1, 12)
        if not np.allclose(gy, gy[:, :1], atol=1e-8, rtol=0):
            warnings.warn(
                f"sim {i}: GMT row is not constant within calendar years. "
                "The encoder uses the January value of each year; if your GMT "
                "really is monthly, set gmt_row handling accordingly.",
                stacklevel=2,
            )
        tas_absmax.append(np.abs(s[1 : 1 + cfg.n_tas]).max())
        pr_absmax.append(np.abs(s[1 + cfg.n_tas :]).max())
        gmt_span.append((g.min(), g.max()))

    if verbose:
        lengths = [np.asarray(s).shape[1] for s in sims]
        print(f"[check_data] {len(sims)} simulations")
        print(f"[check_data] months: min={min(lengths)} max={max(lengths)} "
              f"(= {min(lengths)//12}-{max(lengths)//12} years)")
        print(f"[check_data] |tas| max  = {max(tas_absmax):.4g}")
        print(f"[check_data] |pr|  max  = {max(pr_absmax):.4g}")
        print(f"[check_data] GMT range  = [{min(a for a, _ in gmt_span):.3g}, "
              f"{max(b for _, b in gmt_span):.3g}]")
        ratio = max(tas_absmax) / max(max(pr_absmax), 1e-30)
        print(f"[check_data] tas/pr magnitude ratio = {ratio:.3g}  "
              "(this is why per-channel standardisation is mandatory)")


# --------------------------------------------------------------------------
# normaliser
# --------------------------------------------------------------------------


class Normalizer:
    """Per-(channel, calendar-month) mean/std, plus GMT mean/std."""

    def __init__(
        self,
        mu: np.ndarray,          # (C, 12)
        sd: np.ndarray,          # (C, 12)
        gmt_mu: float,
        gmt_sd: float,
        n_tas: int,
        n_pr: int,
        pr_transform: str,
        start_month: int,
    ):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self.gmt_mu = float(gmt_mu)
        self.gmt_sd = float(gmt_sd)
        self.n_tas = int(n_tas)
        self.n_pr = int(n_pr)
        self.pr_transform = pr_transform
        self.start_month = int(start_month)

    # -- pr transform ------------------------------------------------------
    def _fwd_pr(self, y: np.ndarray) -> np.ndarray:
        if self.pr_transform == "none":
            return y
        if self.pr_transform == "signed_cbrt":
            y = y.copy()
            y[self.n_tas :] = signed_cbrt(y[self.n_tas :])
            return y
        raise ValueError(self.pr_transform)

    def _inv_pr(self, y: np.ndarray) -> np.ndarray:
        if self.pr_transform == "none":
            return y
        if self.pr_transform == "signed_cbrt":
            y = y.copy()
            y[..., self.n_tas :, :] = signed_cube(y[..., self.n_tas :, :])
            return y
        raise ValueError(self.pr_transform)

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(cls, sims: Sequence[np.ndarray], cfg: DataConfig) -> Normalizer:
        C = cfg.n_channels
        s1 = np.zeros((C, 12), dtype=np.float64)
        s2 = np.zeros((C, 12), dtype=np.float64)
        cnt = np.zeros((C, 12), dtype=np.float64)
        gsum = gsq = gn = 0.0

        for sim in sims:
            sim = np.asarray(sim)
            y = sim[1 : 1 + C].astype(np.float64)
            if cfg.pr_transform == "signed_cbrt":
                y = y.copy()
                y[cfg.n_tas :] = signed_cbrt(y[cfg.n_tas :])
            T = y.shape[1]
            months = (np.arange(T) + cfg.start_month) % 12
            for m in range(12):
                sel = y[:, months == m]
                s1[:, m] += sel.sum(axis=1)
                s2[:, m] += (sel ** 2).sum(axis=1)
                cnt[:, m] += sel.shape[1]
            g = sim[cfg.gmt_row, :: 12].astype(np.float64)
            gsum += g.sum()
            gsq += (g ** 2).sum()
            gn += g.size

        mu = s1 / np.maximum(cnt, 1.0)
        var = np.maximum(s2 / np.maximum(cnt, 1.0) - mu ** 2, 0.0)
        sd = np.sqrt(var)
        # guard against constant channels
        sd = np.where(sd < 1e-12, 1.0, sd)

        gmt_mu = gsum / gn
        gmt_sd = float(np.sqrt(max(gsq / gn - gmt_mu ** 2, 1e-24)))

        return cls(mu, sd, gmt_mu, gmt_sd, cfg.n_tas, cfg.n_pr,
                   cfg.pr_transform, cfg.start_month)

    # -- target transforms -------------------------------------------------
    def month_index(self, t0: int, n: int) -> np.ndarray:
        """Calendar-month indices for ``n`` months starting at column ``t0``."""
        return (np.arange(t0, t0 + n) + self.start_month) % 12

    def transform_targets(self, y: np.ndarray, t0: int = 0) -> np.ndarray:
        """``y``: (C, T) raw -> (C, T) normalised."""
        y = self._fwd_pr(np.asarray(y, dtype=np.float64))
        m = self.month_index(t0, y.shape[1])
        return ((y - self.mu[:, m]) / self.sd[:, m]).astype(np.float32)

    def inverse_transform_targets(self, y: np.ndarray, t0: int = 0) -> np.ndarray:
        """``y``: (..., C, T) normalised -> raw physical units."""
        y = np.asarray(y, dtype=np.float64)
        m = self.month_index(t0, y.shape[-1])
        y = y * self.sd[:, m] + self.mu[:, m]
        return self._inv_pr(y)

    # -- GMT ---------------------------------------------------------------
    def annual_gmt(self, sim: np.ndarray, gmt_row: int = 0) -> np.ndarray:
        return np.asarray(sim)[gmt_row, ::12].astype(np.float64)

    def transform_gmt(self, g: np.ndarray) -> np.ndarray:
        return ((np.asarray(g, dtype=np.float64) - self.gmt_mu) / self.gmt_sd).astype(np.float32)

    # -- (de)serialisation -------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "mu": self.mu, "sd": self.sd, "gmt_mu": self.gmt_mu, "gmt_sd": self.gmt_sd,
            "n_tas": self.n_tas, "n_pr": self.n_pr,
            "pr_transform": self.pr_transform, "start_month": self.start_month,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> Normalizer:
        return cls(**d)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


class CropDataset(Dataset):
    """January-aligned crops with a randomly-sized clean prefix (context).

    Each item yields
        x0        (C, W)  normalised target window
        ctx       (C, W)  clean context, zeroed outside the prefix
        ctx_mask  (W,)    1.0 where the prefix is known, else 0.0
        month_idx (W,)    calendar month of each token
        gmt       (Y,)    normalised annual GMT of the *whole* simulation
        end_year  ()      index of the last year covered by the window
        gmt_feats (F,)    causal path features evaluated at ``end_year``
        esm_id    ()      ESM index (0 unless you train on several models)
    """

    def __init__(
        self,
        sims: Sequence[np.ndarray],
        normalizer: Normalizer,
        cfg: Config,
        sim_indices: Sequence[int] | None = None,
        esm_ids: Sequence[int] | None = None,
        train: bool = True,
    ):
        self.cfg = cfg
        d = cfg.data
        self.norm = normalizer
        self.train = train
        self.window = d.window

        if sim_indices is None:
            sim_indices = range(len(sims))
        sim_indices = list(sim_indices)

        self.targets: list[np.ndarray] = []
        self.gmts: list[np.ndarray] = []
        self.esm_ids: list[int] = []
        self.index: list[tuple[int, int]] = []

        step = 12 if d.january_start else 1
        for local, gi in enumerate(sim_indices):
            sim = np.asarray(sims[gi])
            y = normalizer.transform_targets(sim[1 : 1 + d.n_channels], t0=0)
            self.targets.append(np.ascontiguousarray(y))
            self.gmts.append(
                normalizer.transform_gmt(normalizer.annual_gmt(sim, d.gmt_row))
            )
            self.esm_ids.append(0 if esm_ids is None else int(esm_ids[gi]))
            T = y.shape[1]
            for x in range(0, T - d.window + 1, step):
                self.index.append((local, x))

        self.n_feats = n_gmt_features(cfg)
        self._ctx_choices = np.asarray(list(d.context_lengths), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def _sample_ctx_len(self, rng: np.random.Generator) -> int:
        d = self.cfg.data
        if rng.random() < d.p_no_context:
            return 0
        return int(rng.choice(self._ctx_choices))

    def __getitem__(self, i: int):
        local, x = self.index[i]
        W = self.window
        d = self.cfg.data

        y = self.targets[local][:, x : x + W]            # (C, W) float32
        gmt = self.gmts[local]                            # (Y,)
        end_year = (x + W) // 12 - 1

        if self.train:
            rng = np.random.default_rng()
            L = self._sample_ctx_len(rng)
        else:
            # deterministic for reproducible validation loss
            rng = np.random.default_rng(i + 12345)
            L = self._sample_ctx_len(rng)
        L = min(L, W - 12)

        ctx = np.zeros_like(y)
        ctx_mask = np.zeros(W, dtype=np.float32)
        if L > 0:
            ctx[:, :L] = y[:, :L]
            ctx_mask[:L] = 1.0

        month_idx = ((np.arange(x, x + W) + d.start_month) % 12).astype(np.int64)
        feats = gmt_path_features(
            gmt, end_year, self.cfg.model.use_elapsed_time_feature
        )

        return {
            "x0": torch.from_numpy(np.ascontiguousarray(y)),
            "ctx": torch.from_numpy(ctx),
            "ctx_mask": torch.from_numpy(ctx_mask),
            "month_idx": torch.from_numpy(month_idx),
            "gmt": torch.from_numpy(gmt),
            "end_year": torch.tensor(end_year, dtype=torch.long),
            "gmt_feats": torch.from_numpy(feats),
            "esm_id": torch.tensor(self.esm_ids[local], dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    """Right-pad the (variable-length) GMT histories.

    Because the GMT encoder is *causal*, right padding cannot influence the
    readout at ``end_year``, so no key-padding mask is needed.
    """
    ymax = max(b["gmt"].shape[0] for b in batch)
    gmt = torch.zeros(len(batch), ymax)
    for i, b in enumerate(batch):
        gmt[i, : b["gmt"].shape[0]] = b["gmt"]
    out = {
        k: torch.stack([b[k] for b in batch])
        for k in batch[0]
        if k != "gmt"
    }
    out["gmt"] = gmt
    return out


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def group_split(
    n_sims: int,
    groups: Sequence[Hashable] | None = None,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split *simulations* (never crops) into train/val, respecting groups.

    Pass ``groups`` = one label per simulation (e.g. the scenario name, or the
    parent run for branched scenarios).  All simulations sharing a label land
    on the same side of the split, which is what prevents leakage between
    ensemble members and between scenarios that share a historical prefix.
    """
    if groups is None:
        groups = list(range(n_sims))
    groups = list(groups)
    assert len(groups) == n_sims

    uniq = sorted(set(groups), key=lambda g: str(g))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(val_fraction * len(uniq)))) if val_fraction > 0 else 0
    val_groups = {uniq[j] for j in perm[:n_val]}

    train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
    val_idx = [i for i, g in enumerate(groups) if g in val_groups]
    return train_idx, val_idx
