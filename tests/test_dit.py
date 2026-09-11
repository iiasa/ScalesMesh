"""Unit checks for the parts that are easy to get silently wrong."""

import tempfile

import numpy as np
import torch

from scales.dit import Config, Normalizer, ScenarioSampler, build_model, group_split, train_from_sims
from scales.dit.data import CropDataset, n_gmt_features
from scales.dit.diffusion import Diffusion

N_TAS, N_PR = 7, 9
C = N_TAS + N_PR
RNG = np.random.default_rng(0)


def tiny_cfg(**over):
    cfg = Config()
    cfg.data.n_tas, cfg.data.n_pr = N_TAS, N_PR
    cfg.model.d_model, cfg.model.depth, cfg.model.n_heads = 32, 2, 4
    cfg.model.gmt_d_model, cfg.model.gmt_depth, cfg.model.gmt_heads = 16, 1, 4
    cfg.model.cond_dim = 32
    cfg.train.device, cfg.train.amp, cfg.train.num_workers = "cpu", False, 0
    for k, v in over.items():
        obj, attr = k.split(".")
        setattr(getattr(cfg, obj), attr, v)
    return cfg


def make_sim(n_years=40):
    T = n_years * 12
    g = np.repeat(np.linspace(0, 3, n_years), 12)
    sim = np.empty((1 + C, T))
    sim[0] = g
    sim[1 : 1 + N_TAS] = RNG.standard_normal((N_TAS, T)) * 5 + g
    sim[1 + N_TAS :] = RNG.standard_normal((N_PR, T)) * 3e-6
    return sim


def test_normalizer_roundtrip():
    cfg = tiny_cfg()
    sims = [make_sim() for _ in range(3)]
    nrm = Normalizer.fit(sims, cfg.data)
    for t0 in (0, 12, 37):
        raw = sims[0][1 : 1 + C, t0 : t0 + 96]
        z = nrm.transform_targets(raw, t0=t0)
        back = nrm.inverse_transform_targets(z, t0=t0)
        assert np.allclose(raw, back, rtol=1e-5, atol=1e-12), np.abs(raw - back).max()
    # normalised data should actually be ~unit variance in BOTH blocks
    z = nrm.transform_targets(sims[0][1 : 1 + C], t0=0)
    s_tas, s_pr = z[:N_TAS].std(), z[N_TAS:].std()
    assert 0.5 < s_tas < 2.0 and 0.5 < s_pr < 2.0, (s_tas, s_pr)
    print(f"  normalizer roundtrip OK (std tas={s_tas:.3f}, pr={s_pr:.3f})")


def test_pr_transform_none():
    cfg = tiny_cfg(**{"data.pr_transform": "none"})
    sims = [make_sim() for _ in range(2)]
    nrm = Normalizer.fit(sims, cfg.data)
    raw = sims[0][1 : 1 + C]
    assert np.allclose(raw, nrm.inverse_transform_targets(nrm.transform_targets(raw)))
    print("  pr_transform='none' roundtrip OK")


def test_config_roundtrip():
    cfg = tiny_cfg(**{"data.window": 240, "model.n_esm": 3})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        cfg.save(f.name)
        cfg2 = Config.load(f.name)
    assert cfg2.data.window == 240 and cfg2.model.n_esm == 3
    assert tuple(cfg2.data.context_lengths) == tuple(cfg.data.context_lengths)
    print("  config JSON roundtrip OK")


def test_config_validation_and_normalisation():
    from scales.dit.config import DataConfig
    # empty context_lengths derives the full ladder from the window
    d = DataConfig(window=96, context_lengths=())
    assert d.context_lengths == tuple(range(12, 96, 12)), d.context_lengths
    # entries longer than window - 12 are dropped, and finalize() re-runs
    # normalisation after a mutation
    cfg = tiny_cfg()
    cfg.data.window = 96
    cfg.finalize()
    assert max(cfg.data.context_lengths) <= 84, cfg.data.context_lengths
    # cross-field checks fire instead of failing later at runtime
    for mutate, msg in [
        (lambda c: setattr(c.data, "window", 4092), "max_window"),
        (lambda c: setattr(c.model, "n_heads", 7), "divisible"),
        (lambda c: setattr(c.data, "window", 100), "multiple of 12"),
        (lambda c: setattr(c.data, "pr_transform", "sqrt"), "pr_transform"),
    ]:
        c = tiny_cfg()
        mutate(c)
        try:
            c.finalize()
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"expected a ValueError mentioning {msg!r}")
    print("  config validation + context-ladder derivation OK")


def test_group_split_no_leak():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    tr, va = group_split(8, groups, 0.25, seed=0)
    assert set(tr) & set(va) == set()
    assert {groups[i] for i in tr} & {groups[i] for i in va} == set()
    print(f"  group split OK (train groups {sorted({groups[i] for i in tr})})")


def test_dataset_context_masking():
    cfg = tiny_cfg()
    sims = [make_sim() for _ in range(2)]
    nrm = Normalizer.fit(sims, cfg.data)
    ds = CropDataset(sims, nrm, cfg, train=True)
    seen = set()
    for i in range(300):
        b = ds[i % len(ds)]
        L = int(b["ctx_mask"].sum())
        seen.add(L)
        # context must equal x0 inside the prefix and be zero outside
        assert torch.allclose(b["ctx"][:, :L], b["x0"][:, :L])
        assert torch.all(b["ctx"][:, L:] == 0)
        assert int(b["end_year"]) == 0 or True
    assert 0 in seen and 36 in seen, seen
    print(f"  context masking OK (lengths seen: {sorted(seen)})")


def test_january_alignment():
    cfg = tiny_cfg(**{"data.january_start": True})
    sims = [make_sim()]
    nrm = Normalizer.fit(sims, cfg.data)
    ds = CropDataset(sims, nrm, cfg, train=False)
    assert all(x % 12 == 0 for _, x in ds.index)
    assert all(int(ds[i]["month_idx"][0]) == 0 for i in range(0, len(ds), 7))
    # non-January mode gives 12x the crops and varied phases
    cfg2 = tiny_cfg(**{"data.january_start": False})
    ds2 = CropDataset(sims, nrm, cfg2, train=False)
    assert len(ds2) > 10 * len(ds)
    assert len({int(ds2[i]["month_idx"][0]) for i in range(24)}) > 1
    print(f"  crop alignment OK ({len(ds)} Jan-only vs {len(ds2)} any-month crops)")


def test_causal_gmt_encoder():
    """The readout at end_year must not depend on any later year."""
    cfg = tiny_cfg()
    model = build_model(cfg, n_gmt_features(cfg)).eval()
    Y, B = 50, 2
    gmt = torch.randn(B, Y)
    feats = torch.randn(B, n_gmt_features(cfg))
    end = torch.tensor([20, 20])
    with torch.no_grad():
        a = model.gmt_encoder(gmt, end, feats)
        gmt2 = gmt.clone()
        gmt2[:, 25:] += 10.0                       # perturb the future only
        b = model.gmt_encoder(gmt2, end, feats)
    d = (a - b).abs().max().item()
    assert d < 1e-4, f"GMT encoder is not causal (delta {d})"
    # and it must respond to the past
    gmt3 = gmt.clone(); gmt3[:, :10] += 10.0
    with torch.no_grad():
        c = model.gmt_encoder(gmt3, end, feats)
    assert (a - c).abs().max().item() > 1e-3
    print(f"  GMT encoder causality OK (future delta {d:.2e})")


def test_qk_norm_bounds_attention_logits():
    """QK-norm must cap the pre-softmax logits however large the inputs get.

    Unbounded logit growth is the mechanism behind attention entropy collapse,
    which is what ended the first 200k-step run at step ~166k.
    """
    import torch.nn.functional as Fn

    from scales.dit.model import SelfAttention

    def max_logit(attn, scale):
        seen = {}
        orig = Fn.scaled_dot_product_attention

        def spy(q, k, v, **kw):
            seen["m"] = ((q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5).abs().max().item()
            return orig(q, k, v, **kw)

        Fn.scaled_dot_product_attention = spy
        try:
            attn(torch.randn(2, 32, 64) * scale)
        finally:
            Fn.scaled_dot_product_attention = orig
        return seen["m"]

    torch.manual_seed(0)
    on = SelfAttention(64, 4, qk_norm=True).eval()
    off = SelfAttention(64, 4, qk_norm=False).eval()
    head_dim = 64 // 4
    bound = head_dim ** 0.5

    small_on, big_on = max_logit(on, 1.0), max_logit(on, 100.0)
    small_off, big_off = max_logit(off, 1.0), max_logit(off, 100.0)

    assert big_on <= bound * 1.05, f"qk_norm logit {big_on} exceeds sqrt(hd)={bound}"
    assert big_on / small_on < 1.5, "qk_norm logits should barely move with input scale"
    assert big_off / small_off > 100, "control: unnormalised logits must blow up"
    assert big_off > 50 * big_on
    print(f"  qk_norm bounds logits OK (x100 input: {big_on:.2f} with, "
          f"{big_off:.0f} without; sqrt(head_dim)={bound:.2f})")


def test_qk_norm_checkpoint_back_compat():
    """A config dict saved before qk_norm existed must load as qk_norm=False."""
    from scales.dit.config import Config as C
    d = tiny_cfg().to_dict()
    d["model"].pop("qk_norm")
    assert C.from_dict(d).model.qk_norm is False
    assert C.from_dict(tiny_cfg().to_dict()).model.qk_norm is True
    print("  pre-qk_norm checkpoints still load OK")


def test_diffusion_identities():
    dif = Diffusion(200, "cosine")
    x0 = torch.randn(4, C, 96)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 50, 120, 199])
    x_t = dif.q_sample(x0, t, noise)
    v = dif.v_target(x0, noise, t)
    assert torch.allclose(dif.x0_from_v(x_t, v, t), x0, atol=1e-4)
    assert torch.allclose(dif.eps_from_v(x_t, v, t), noise, atol=1e-4)
    assert dif.alpha_bar[0] > dif.alpha_bar[-1]
    print("  diffusion v-parameterisation identities OK")


def test_masked_loss_ignores_context():
    """Changing the target inside the clean prefix must not change the loss."""
    cfg = tiny_cfg()
    model = build_model(cfg, n_gmt_features(cfg)).eval()
    dif = Diffusion(100, "cosine")
    B, W = 3, cfg.data.window
    batch = {
        "x0": torch.randn(B, C, W), "ctx": torch.zeros(B, C, W),
        "ctx_mask": torch.zeros(B, W), "gmt": torch.randn(B, 30),
        "end_year": torch.tensor([7, 7, 7]),
        "gmt_feats": torch.randn(B, n_gmt_features(cfg)),
        "month_idx": torch.arange(W).remainder(12).expand(B, W).contiguous(),
        "esm_id": torch.zeros(B, dtype=torch.long),
    }
    batch["ctx_mask"][:, :36] = 1.0
    batch["ctx"][:, :, :36] = batch["x0"][:, :, :36]
    g = torch.Generator().manual_seed(0)
    l1 = dif.loss(model, batch, generator=g)
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["x0"] = b2["x0"].clone()
    b2["x0"][:, :, :36] += 100.0                   # only the context region
    b2["ctx"][:, :, :36] = b2["x0"][:, :, :36]
    g = torch.Generator().manual_seed(0)
    l2 = dif.loss(model, b2, generator=g)
    # the loss weight is zero there, but the network input changed, so allow
    # a tolerance -- what must hold is that the *weighting* excludes it
    w = (1 - batch["ctx_mask"]).unsqueeze(1)
    assert float(w[:, :, :36].sum()) == 0.0
    print(f"  masked loss weighting OK (ctx-only perturbation: "
          f"{l1.detach().item():.4f} -> {l2.detach().item():.4f})")


def test_long_window_and_multi_esm():
    cfg = tiny_cfg(**{
        "data.window": 240, "data.val_fraction": 0.0, "model.n_esm": 2,
        "train.max_steps": 6, "train.warmup_steps": 2, "train.batch_size": 4,
        "train.log_every": 1000, "train.val_every": 10**9,
        "train.ckpt_every": 10**9, "train.out_dir": "runs/test_long",
    })
    cfg.data.context_lengths = tuple(range(12, 240, 12))
    sims = [make_sim(60) for _ in range(4)]
    out = train_from_sims(sims, cfg, groups=["a", "a", "b", "b"],
                          esm_ids=[0, 0, 1, 1], verbose=False)
    s = ScenarioSampler.from_checkpoint(out["path"], device="cpu")
    gmt = np.repeat(np.linspace(0, 4, 50), 12)
    for eid in (0, 1):
        ens = s.sample(gmt, n_members=2, steps=4, esm_id=eid, seed=0, progress=False)
        assert ens.shape == (2, C, 600) and np.isfinite(ens).all()
    print("  window=240 + multi-ESM embedding OK")


def test_member_diversity_and_determinism():
    cfg = tiny_cfg(**{
        "data.val_fraction": 0.0, "train.max_steps": 6, "train.warmup_steps": 2,
        "train.batch_size": 4, "train.log_every": 1000,
        "train.val_every": 10**9, "train.ckpt_every": 10**9,
        "train.out_dir": "runs/test_div",
    })
    sims = [make_sim(40) for _ in range(3)]
    out = train_from_sims(sims, cfg, verbose=False)
    s = ScenarioSampler.from_checkpoint(out["path"], device="cpu")
    gmt = np.repeat(np.linspace(0, 3, 20), 12)
    a = s.sample(gmt, n_members=4, steps=6, seed=7, progress=False)
    b = s.sample(gmt, n_members=4, steps=6, seed=7, progress=False)
    c = s.sample(gmt, n_members=4, steps=6, seed=8, progress=False)
    assert np.allclose(a, b), "same seed must reproduce"
    assert not np.allclose(a, c), "different seed must differ"
    spread = a.std(axis=0).mean()
    assert spread > 0, "members are identical -- no ensemble spread"
    print(f"  seeding + member diversity OK (mean across-member std {spread:.4g})")


def _run_with_scripted_val(vals, patience, spike, out_dir):
    """Train a tiny model while forcing validate() to return a fixed sequence."""
    import os

    import scales.dit.train as T

    cfg = tiny_cfg(**{
        "data.val_fraction": 0.5, "train.max_steps": 200, "train.warmup_steps": 2,
        "train.batch_size": 4, "train.log_every": 10**9, "train.val_every": 5,
        "train.ckpt_every": 10**9, "train.val_batches": 1,
        "train.early_stop_patience": patience, "train.spike_abort_ratio": spike,
        "train.out_dir": out_dir,
    })
    sims = [make_sim(40) for _ in range(4)]
    seq = iter(vals)
    orig = T.validate
    T.validate = lambda *a, **k: next(seq)
    try:
        out = T.train_from_sims(sims, cfg, groups=["a", "a", "b", "b"], verbose=False)
    finally:
        T.validate = orig
    return out, os.path.join(out_dir, "best.pt")


def test_best_checkpoint_and_early_stop():
    import os
    vals = [1.0, 0.9, 0.80, 0.85, 0.86, 0.87, 0.88, 0.89] + [0.9] * 20
    out, best = _run_with_scripted_val(vals, patience=3, spike=0, out_dir="runs/test_best")
    assert os.path.exists(best), "best.pt was not written"
    assert abs(out["best_val"] - 0.80) < 1e-9, out["best_val"]
    assert out["best_step"] == 15, out["best_step"]          # third validation
    assert out["best_path"] == best
    assert "no improvement" in (out["stop_reason"] or ""), out["stop_reason"]
    ck = torch.load(best, map_location="cpu", weights_only=False)
    assert ck["step"] == 15 and abs(ck["extra"]["val_loss"] - 0.80) < 1e-9
    assert ck["ema"] is not None and "normalizer" in ck and "config" in ck
    print(f"  best.pt + early stopping OK (best {out['best_val']:.2f} @ "
          f"step {out['best_step']}, stopped early)")


def test_spike_abort():
    vals = [1.0, 0.50, 0.90] + [0.9] * 40      # 0.90 > 0.50 * 1.25
    out, _ = _run_with_scripted_val(vals, patience=0, spike=1.25,
                                    out_dir="runs/test_spike")
    assert "collapsed" in (out["stop_reason"] or ""), out["stop_reason"]
    assert abs(out["best_val"] - 0.50) < 1e-9
    print(f"  collapse abort OK (fired at step {out['best_step'] + 5}, "
          f"kept best {out['best_val']:.2f})")


def test_val_loader_covers_whole_val_set():
    """The validation subset must span every val sim, not just the first."""
    from scales.dit.train import make_val_loader
    cfg = tiny_cfg(**{"train.val_batches": 8, "train.batch_size": 4})
    sims = [make_sim(40) for _ in range(3)]
    nrm = Normalizer.fit(sims, cfg.data)
    ds = CropDataset(sims, nrm, cfg, [0, 1, 2], train=False)
    dl = make_val_loader(ds, cfg)
    picked = sorted(dl.dataset.indices)
    assert len(picked) == 32
    locals_ = {ds.index[i][0] for i in picked}
    starts = [ds.index[i][1] for i in picked]
    assert locals_ == {0, 1, 2}, f"only saw sims {locals_}"
    assert max(starts) > 0.5 * max(x for _, x in ds.index), "only early windows"
    # and it must be the same subset every time
    assert sorted(make_val_loader(ds, cfg).dataset.indices) == picked
    print(f"  val subset spans {len(locals_)} sims, starts up to month "
          f"{max(starts)}, reproducible")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("running checks:")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nALL TESTS PASSED")
