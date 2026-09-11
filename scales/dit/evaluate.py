"""Diagnostics: is the emulator actually good?

This is the part that decides whether the project works, so it is worth more
attention than the architecture.  The failure modes to watch, roughly in order
of how often they bite:

1. **Spread miscalibration** -- generated ensembles too narrow (the classic
   symptom of guidance, too few sampling steps, or an over-fit model).
   -> ``spread_ratio``, ``rank_histogram``, ``crps``
2. **Missing low-frequency variability** -- the 8-year window cannot represent
   multidecadal internal variability on its own.
   -> ``power_spectrum``, ``variance_by_timescale``
3. **Autoregressive drift** across hundreds of stitched windows.
   -> ``trend_drift``
4. **Broken cross-region / cross-variable structure** -- each region looks fine
   marginally but the joint distribution is wrong.
   -> ``corr_error``, ``tas_pr_coupling``
5. **Memorisation** -- with a few thousand effective samples this is real.
   -> ``nearest_neighbour_distance``

All functions take numpy arrays.  ``gen`` is ``(M, C, T)``, a reference ESM
ensemble ``ref`` is ``(M_ref, C, T)`` (rows 1..116 of your simulations, i.e.
without the GMT row).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# basic reshaping helpers
# --------------------------------------------------------------------------


def annual_means(x: np.ndarray) -> np.ndarray:
    """(..., C, T) -> (..., C, T//12) calendar-year means."""
    T = x.shape[-1]
    return x[..., : T // 12 * 12].reshape(*x.shape[:-1], T // 12, 12).mean(-1)


def seasonal_cycle(x: np.ndarray, start_month: int = 0) -> np.ndarray:
    """(..., C, T) -> (..., C, 12) climatological mean by calendar month."""
    T = x.shape[-1]
    m = (np.arange(T) + start_month) % 12
    return np.stack([x[..., m == k].mean(-1) for k in range(12)], axis=-1)


def deseasonalize(x: np.ndarray, start_month: int = 0) -> np.ndarray:
    T = x.shape[-1]
    m = (np.arange(T) + start_month) % 12
    clim = seasonal_cycle(x, start_month)
    return x - clim[..., m]


# --------------------------------------------------------------------------
# 1. spread calibration
# --------------------------------------------------------------------------


def spread_ratio(gen: np.ndarray, ref: np.ndarray, start_month: int = 0) -> np.ndarray:
    """Per-channel ratio of across-member std (deseasonalised).

    1.0 = perfectly calibrated internal variability.  < 1 means the emulator
    is under-dispersive, which is the usual failure.
    """
    g = deseasonalize(gen, start_month).std(axis=0).mean(axis=-1)
    r = deseasonalize(ref, start_month).std(axis=0).mean(axis=-1)
    return g / np.maximum(r, 1e-30)


def rank_histogram(gen: np.ndarray, obs: np.ndarray, bins: int | None = None) -> np.ndarray:
    """Rank of each observation within the generated ensemble.

    A flat histogram means calibrated; U-shaped = under-dispersive;
    dome-shaped = over-dispersive.  ``obs`` is ``(C, T)``.
    """
    M = gen.shape[0]
    ranks = (gen < obs[None]).sum(axis=0).reshape(-1)
    hist = np.bincount(ranks, minlength=M + 1).astype(float)
    if bins is not None and bins < M + 1:
        edges = np.linspace(0, M + 1, bins + 1).astype(int)
        hist = np.array([hist[edges[i]:edges[i + 1]].sum() for i in range(bins)])
    return hist / hist.sum()


def crps(gen: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Ensemble CRPS per channel (lower is better).  ``obs``: (C, T)."""
    M = gen.shape[0]
    term1 = np.abs(gen - obs[None]).mean(axis=0)                       # (C, T)
    g = np.sort(gen, axis=0)
    # sum_{i<j}(g_j - g_i) = sum_i (2i - M + 1) g_i
    coef = (2 * np.arange(M) - M + 1).reshape(M, 1, 1)
    term2 = (coef * g).sum(axis=0) / (M * M)
    return (term1 - term2).mean(axis=-1)


# --------------------------------------------------------------------------
# 2. temporal structure
# --------------------------------------------------------------------------


def acf(x: np.ndarray, max_lag: int = 36, start_month: int = 0) -> np.ndarray:
    """Autocorrelation of deseasonalised series, lags 1..max_lag. -> (C, max_lag)."""
    a = deseasonalize(np.asarray(x, dtype=np.float64), start_month)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-2], a.shape[-1])
        a = a - a.mean(-1, keepdims=True)
        var = (a ** 2).mean(-1)
        out = np.empty((a.shape[1], max_lag))
        for L in range(1, max_lag + 1):
            out[:, L - 1] = ((a[..., L:] * a[..., :-L]).mean(-1) /
                             np.maximum(var, 1e-30)).mean(0)
        return out
    a = a - a.mean(-1, keepdims=True)
    var = (a ** 2).mean(-1)
    return np.stack([
        (a[..., L:] * a[..., :-L]).mean(-1) / np.maximum(var, 1e-30)
        for L in range(1, max_lag + 1)
    ], axis=-1)


def power_spectrum(x: np.ndarray, annual: bool = True, start_month: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Mean power spectrum of the (deseasonalised) series.

    Returns ``(freqs_in_cycles_per_year, power)`` with power shape ``(C, F)``.
    Use this to check that multidecadal variance is not missing -- the low-
    frequency end is where an 8-year generation window is most likely to fail.
    """
    a = deseasonalize(np.asarray(x, dtype=np.float64), start_month)
    if annual:
        a = annual_means(a)
        dt = 1.0
    else:
        dt = 1.0 / 12.0
    a = a - a.mean(-1, keepdims=True)
    F = np.fft.rfft(a, axis=-1)
    p = (np.abs(F) ** 2) / a.shape[-1]
    freqs = np.fft.rfftfreq(a.shape[-1], d=dt)
    while p.ndim > 2:
        p = p.mean(0)
    return freqs, p


def variance_by_timescale(x: np.ndarray, windows=(1, 5, 10, 20, 50),
                          start_month: int = 0) -> dict:
    """Variance of running means at several timescales (in years)."""
    ann = annual_means(deseasonalize(np.asarray(x, dtype=np.float64), start_month))
    out = {}
    for w in windows:
        if ann.shape[-1] < 2 * w:
            continue
        k = np.ones(w) / w
        sm = np.apply_along_axis(lambda v, k=k: np.convolve(v, k, mode="valid"), -1, ann)
        out[w] = sm.var(axis=-1).mean(axis=tuple(range(sm.ndim - 2)))
    return out


# --------------------------------------------------------------------------
# 3. drift
# --------------------------------------------------------------------------


def trend_drift(gen: np.ndarray, gmt_monthly: np.ndarray, n_tas: int,
                area_weights: np.ndarray | None = None) -> np.ndarray:
    """Residual between the (weighted) mean of generated tas and prescribed GMT.

    Returned as an annual series, averaged over members.  A slow ramp here is
    the signature of autoregressive drift; a constant offset is only a baseline
    mismatch and is harmless.
    """
    tas = gen[:, :n_tas]
    w = (np.ones(n_tas) / n_tas if area_weights is None
         else np.asarray(area_weights, dtype=np.float64))
    w = w / w.sum()
    mean_tas = np.tensordot(w, tas, axes=([0], [1]))          # (M, T)
    resid = mean_tas - np.asarray(gmt_monthly)[None, : mean_tas.shape[-1]]
    return annual_means(resid[:, None])[:, 0].mean(0)


# --------------------------------------------------------------------------
# 4. joint structure
# --------------------------------------------------------------------------


def corr_matrix(x: np.ndarray, start_month: int = 0) -> np.ndarray:
    """Cross-channel correlation of deseasonalised anomalies. -> (C, C)."""
    a = deseasonalize(np.asarray(x, dtype=np.float64), start_month)
    if a.ndim == 3:
        a = np.concatenate(list(a), axis=-1)
    return np.corrcoef(a)


def corr_error(gen: np.ndarray, ref: np.ndarray, start_month: int = 0) -> dict:
    cg, cr = corr_matrix(gen, start_month), corr_matrix(ref, start_month)
    d = cg - cr
    return {
        "frobenius": float(np.linalg.norm(d) / np.linalg.norm(cr)),
        "max_abs": float(np.abs(d).max()),
        "mean_abs": float(np.abs(d).mean()),
    }


def tas_pr_coupling(x: np.ndarray, n_tas: int, start_month: int = 0) -> np.ndarray:
    """Correlation between each region's tas and pr anomalies.

    Only meaningful if tas row i and pr row i refer to the same region; if your
    two blocks use different region sets, compare the full cross-block block of
    ``corr_matrix`` instead.
    """
    a = deseasonalize(np.asarray(x, dtype=np.float64), start_month)
    if a.ndim == 3:
        a = np.concatenate(list(a), axis=-1)
    n = min(n_tas, a.shape[0] - n_tas)
    t, p = a[:n], a[n_tas : n_tas + n]
    t = t - t.mean(-1, keepdims=True)
    p = p - p.mean(-1, keepdims=True)
    return (t * p).mean(-1) / np.maximum(t.std(-1) * p.std(-1), 1e-30)


# --------------------------------------------------------------------------
# 5. memorisation
# --------------------------------------------------------------------------


def nearest_neighbour_distance(
    gen_windows: np.ndarray, train_windows: np.ndarray, normalise: bool = True
) -> np.ndarray:
    """Min RMS distance from each generated window to any training window.

    Compare against the typical *within-training* nearest-neighbour distance:
    if generated windows sit systematically closer to training data than
    training windows sit to each other, the model is copying.
    ``gen_windows``: (G, C, W); ``train_windows``: (K, C, W).
    """
    g = gen_windows.reshape(gen_windows.shape[0], -1).astype(np.float64)
    tr = train_windows.reshape(train_windows.shape[0], -1).astype(np.float64)
    if normalise:
        s = tr.std(axis=0, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        g, tr = g / s, tr / s
    d = np.empty(g.shape[0])
    for i in range(g.shape[0]):
        d[i] = np.sqrt(((tr - g[i]) ** 2).mean(axis=1)).min()
    return d


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def report(gen: np.ndarray, ref: np.ndarray, n_tas: int,
           gmt_monthly: np.ndarray | None = None, start_month: int = 0) -> dict:
    """Print a compact comparison of a generated ensemble against an ESM one."""
    T = min(gen.shape[-1], ref.shape[-1])
    gen, ref = gen[..., :T], ref[..., :T]
    res = {}

    sr = spread_ratio(gen, ref, start_month)
    res["spread_ratio_tas"] = float(np.median(sr[:n_tas]))
    res["spread_ratio_pr"] = float(np.median(sr[n_tas:]))

    gm = gen.mean(0)
    rm = ref.mean(0)
    res["mean_bias_tas"] = float(np.abs(gm[:n_tas] - rm[:n_tas]).mean())
    res["mean_bias_pr"] = float(np.abs(gm[n_tas:] - rm[n_tas:]).mean())

    ag, ar = acf(gen, 24, start_month), acf(ref, 24, start_month)
    res["acf_mae_lag1_12"] = float(np.abs(ag[:, :12] - ar[:, :12]).mean())
    res["acf_mae_lag13_24"] = float(np.abs(ag[:, 12:] - ar[:, 12:]).mean())

    res.update({f"corr_{k}": v for k, v in corr_error(gen, ref, start_month).items()})

    vg = variance_by_timescale(gen, start_month=start_month)
    vr = variance_by_timescale(ref, start_month=start_month)
    for w in sorted(set(vg) & set(vr)):
        res[f"var_ratio_{w}yr"] = float(
            np.median(vg[w] / np.maximum(vr[w], 1e-30))
        )

    if gmt_monthly is not None:
        dr = trend_drift(gen, gmt_monthly, n_tas)
        if dr.size > 4:
            yrs = np.arange(dr.size)
            res["drift_per_century"] = float(np.polyfit(yrs, dr, 1)[0] * 100)

    print("=" * 62)
    print("MISCH-MASCH diagnostics  (generated vs. reference ensemble)")
    print("=" * 62)
    for k, v in res.items():
        print(f"  {k:<26s} {v: .5g}")
    print("-" * 62)
    print("  spread_ratio_*  : want ~1.00; <0.9 = under-dispersive")
    print("  var_ratio_*yr   : want ~1.00 at ALL timescales; a fall-off at")
    print("                    20-50 yr means the window is too short")
    print("  drift_per_century: want ~0; a ramp = autoregressive drift")
    print("=" * 62)
    return res
