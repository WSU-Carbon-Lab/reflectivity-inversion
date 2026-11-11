"""Standardized perceptron visualizations: watch a reflectivity curve morph through the net.

Two figure types, identical in layout for every checkpoint so runs can be compared side by side:

1. ``<tag>_morph.png`` - "what does layer k still know about the curve?"
   For every stage (input -> mixer -> L1 ... -> output) a ridge decoder is fit from the
   layer's activations back to the (S, P) input curve on held-out rows. The decoded curve is
   the layer's view of the reflectivity: early layers reproduce every Kiessig fringe, deeper
   layers keep only what matters for the labels. The last row re-simulates reflectivity from
   the network's predicted film with refloxide, closing the loop curve -> labels -> curve.
   Right-hand columns track how much of the curve (overall and fringe detail) survives, and
   a PCA embedding coloured by film thickness shows the representation organizing itself.

2. ``<tag>_cards/<NN>_<layer>.png`` - one standardized "layer card" per block:
   receptive fields over q, activation map sorted by thickness, pre-activation & firing
   rate distributions, label tuning, PCA, and weight spectrum.

Usage:
    uv run python analysis/perceptron_viz.py --checkpoint checkpoints/<run>/best.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from diagnostics import (  # noqa: E402
    FIG_DPI, INK, MUTED, SERIES, _style, block_layout, load_model, load_val_rows, unroll,
)
import physics  # noqa: E402
from loss import LABEL_NAMES  # noqa: E402

HERO_THICKNESS = (80.0, 300.0, 650.0)
HERO_COLORS = (SERIES[0], SERIES[1], SERIES[2])


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def stage_list(model, s, p, q):
    """Return [(name, features (N, D), activation-or-None, op-or-None)] for every stage."""
    with torch.no_grad():
        stacked, blocks = unroll(model, s, p, q)
    stages = [("input", stacked[:, :2].flatten(1), None, None)]
    for b in blocks:
        stages.append((b["name"], b["post"].flatten(1), b["act"], b["op"]))
    return stacked, blocks, stages


def pick_heroes(y, targets=HERO_THICKNESS):
    idx = []
    for t in targets:
        order = torch.argsort((y[:, 0] - t).abs())
        idx.append(int(next(i for i in order.tolist() if i not in idx)))
    return idx


def fit_decoder(feats, target, tr, lam_grid=(1e-3, 1e-2, 1e-1, 1.0)):
    """Ridge decoder feats -> target (both numpy). Returns predict(feats) fn picked on a val split."""
    n_tr = int(0.8 * len(tr))
    fit, val = tr[:n_tr], tr[n_tr:]
    # Project wide layers onto their top principal components (fit split only) so the
    # decoder measures what the layer encodes rather than overfitting 800+ features.
    max_k = 512
    if feats.shape[1] > max_k:
        f0 = feats[fit] - feats[fit].mean(0)
        _, _, vt = np.linalg.svd(f0, full_matrices=False)
        proj, center = vt[:max_k].T, feats[fit].mean(0)
        feats_in = feats
        feats = (feats_in - center) @ proj
        inner = fit_decoder(feats, target, tr, lam_grid)
        return lambda f: inner((f - center) @ proj)
    mu, sd = feats[fit].mean(0), feats[fit].std(0) + 1e-6
    tm = target[fit].mean(0)

    def design(f):
        return (f - mu) / sd

    X = design(feats[fit])
    Y = target[fit] - tm
    dual = X.shape[1] > X.shape[0]
    G = X @ X.T if dual else X.T @ X
    scale = np.trace(G) / G.shape[0]
    best, best_err = None, np.inf
    for lam in lam_grid:
        reg = G + lam * scale * np.eye(G.shape[0])
        W = X.T @ np.linalg.solve(reg, Y) if dual else np.linalg.solve(reg, X.T @ Y)
        err = ((design(feats[val]) @ W - (target[val] - tm)) ** 2).mean()
        if err < best_err:
            best, best_err = W, err
    return lambda f: design(f) @ best + tm


def r2_per_dim(pred, true):
    ss_res = ((pred - true) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def high_pass(x, width=9):
    """Remove the smooth Fresnel trend along q so only fringe detail remains."""
    kernel = np.ones(width) / width
    pad = width // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    smooth = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), -1, xp)
    return x - smooth


def pca2(feats):
    f = feats - feats.mean(0)
    u, sv, _ = np.linalg.svd(f, full_matrices=False)
    var = sv**2 / (sv**2).sum()
    return u[:, :2] * sv[:2], var


def resimulate(labels, q, energy_ev):
    """Simulate (R_s, R_p) for predicted physical labels with refloxide (vacuum|film|SiO2|Si)."""
    from refloxide.model import FreeUniTensor, MaterialSLD, ReflectModel

    film_t, film_sigma, ox_t, dxx, dzz, bxx, bzz = [float(v) for v in labels]
    film_t, film_sigma, ox_t = max(film_t, 0.0), max(film_sigma, 0.0), max(ox_t, 0.0)
    vac = MaterialSLD("", 0.0, name="vacuum")(0.0, 0.0)
    film = FreeUniTensor(energies=[energy_ev], name="film")
    ch = film.channel_at(energy_ev)
    ch.delta_o.setp(dxx)
    ch.beta_o.setp(max(bxx, 0.0))
    ch.delta_e.setp(dzz)
    ch.beta_e.setp(max(bzz, 0.0))
    ox = MaterialSLD("SiO2", 2.2, name="oxide")(ox_t, ox_t / (2.0 * math.pi))
    si = MaterialSLD("Si", 2.33, name="silicon")(0.0, 1.2)
    res = ReflectModel(vac | film(film_t, film_sigma) | ox | si, energies=[energy_ev], parallel=False)(
        np.asarray(q, dtype=np.float64), energy_ev
    )
    return np.asarray(res.s), np.asarray(res.p)


def estimate_energy(q):
    """Photon energy [eV] of one curve; handles both random-angle and fixed-q curves."""
    return physics.estimate_energy(float(q[0]), float(q[-1]))


def input_to_display(model, stacked_sp):
    """Undo standardization so decoded curves are shown in regularizer units (log10 R or R q^4)."""
    if getattr(model, "normalize_inputs", False):
        m = model.input_mean[:2].numpy()
        s = model.input_std[:2].numpy()
        return stacked_sp * s + m
    return stacked_sp


# --------------------------------------------------------------------------------------
# Figure 1: morph
# --------------------------------------------------------------------------------------


def plot_morph(model, criterion, meta, data, path, n_rows=8192):
    s, p, q, y = (t[:n_rows] for t in data)
    stacked, blocks, stages = stage_list(model, s, p, q)
    target = stacked[:, :2].numpy()  # (N, 2, n_q) in model-input units
    n, _, n_q = target.shape
    flat_target = target.reshape(n, -1)
    hp_target = high_pass(target).reshape(n, -1)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    heroes = pick_heroes(y)
    te = np.setdiff1d(te, heroes)
    is_log = model.regularizer in ("log", "log_rq4")
    unit = r"$\log_{10}$" if is_log else r"$R\,q^4$"

    pred_norm = blocks[-1]["post"].detach()
    pred_phys = criterion.denormalize(pred_norm).numpy()

    n_st = len(stages) + 1  # + re-simulated output
    fig = plt.figure(figsize=(24, 2.9 * n_st + 1.2), layout="constrained")
    gs = fig.add_gridspec(n_st, 5, width_ratios=[1, 1, 1, 0.9, 0.85])
    q_idx = np.arange(n_q)

    keep_all, keep_hp, names = [], [], []
    for r, (name, feats, act, op) in enumerate(stages):
        f = feats.numpy()
        dec = fit_decoder(f, flat_target, tr)
        rec = dec(f)
        r2 = r2_per_dim(rec[te], flat_target[te]).reshape(2, n_q)
        r2_hp = r2_per_dim(high_pass(rec[te].reshape(-1, 2, n_q)).reshape(len(te), -1), hp_target[te])
        keep_all.append(float(np.clip(r2, -1, 1).mean()))
        keep_hp.append(float(np.clip(r2_hp, -1, 1).mean()))
        names.append(name)

        rec_disp = input_to_display(model, rec.reshape(n, 2, n_q))
        true_disp = input_to_display(model, target)
        for c, h in enumerate(heroes):
            ax = fig.add_subplot(gs[r, c])
            ax.plot(q_idx, true_disp[h, 0], color="0.75", lw=2.6, label="input S" if r == 0 else None)
            ax.plot(q_idx, true_disp[h, 1], color="0.85", lw=2.6, ls="-", label="input P" if r == 0 else None)
            ax.plot(q_idx, rec_disp[h, 0], color=HERO_COLORS[c], lw=1.4, label="decoded S" if r == 0 else None)
            ax.plot(q_idx, rec_disp[h, 1], color=HERO_COLORS[c], lw=1.1, ls="--", label="decoded P" if r == 0 else None)
            if not is_log:
                ax.set_yscale("symlog", linthresh=1e-9)
            _style(ax, f"{name}" + (f"  — film {y[h, 0]:.0f} Å" if r == 0 else ""))
            if c == 0:
                ax.set_ylabel(unit, fontsize=10.5)
            if r == 0 and c == 0:
                ax.legend(fontsize=9.5, frameon=False, ncol=2)
            ax.set_xticks([] if r < n_st - 1 else ax.get_xticks())

        # Column 4: reconstruction R^2 along q (where on the curve detail is kept).
        ax = fig.add_subplot(gs[r, 3])
        ax.plot(q_idx, r2[0], color=SERIES[0], lw=1.5, label="S")
        ax.plot(q_idx, r2[1], color=SERIES[1], lw=1.5, label="P")
        ax.set_ylim(-0.05, 1.02)
        _style(ax, f"curve kept: R² {keep_all[-1]:.2f} · fringes {keep_hp[-1]:.2f}")
        if r == 0:
            ax.legend(fontsize=9.5, frameon=False)

        # Column 5: PCA coloured by film thickness.
        ax = fig.add_subplot(gs[r, 4])
        emb, var = pca2(f[te])
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y[te, 0].numpy(), cmap="viridis", s=3, alpha=0.7, linewidths=0)
        _style(ax, f"PCA ({100*var[:2].sum():.0f}% var), colour = thickness")
        ax.set_xticks([])
        ax.set_yticks([])

    # Last row: the network's answer, re-simulated as a reflectivity curve.
    r = n_st - 1
    for c, h in enumerate(heroes):
        ax = fig.add_subplot(gs[r, c])
        qh = q[h].numpy()
        e = estimate_energy(qh)
        try:
            rs, rp = resimulate(pred_phys[h], qh, e)
            ok = True
        except Exception as exc:  # refloxide can reject non-physical predictions
            ok = False
            ax.text(0.5, 0.5, f"re-simulation failed:\n{exc}", ha="center", va="center", transform=ax.transAxes, fontsize=9.5)
        ax.semilogy(q_idx, s[h].numpy(), color="0.75", lw=2.6, label="measured s")
        ax.semilogy(q_idx, p[h].numpy(), color="0.85", lw=2.6, label="measured p")
        if ok:
            ax.semilogy(q_idx, rs, color=HERO_COLORS[c], lw=1.4, label="network s")
            ax.semilogy(q_idx, rp, color=HERO_COLORS[c], lw=1.1, ls="--", label="network p")
        _style(ax, f"re-simulated: pred {pred_phys[h, 0]:.0f} Å, truth {y[h, 0]:.0f} Å")
        ax.set_xlabel("q index (0 .. 255)")
        if c == 0:
            ax.set_ylabel("R")
            ax.legend(fontsize=9.5, frameon=False, ncol=2)

    ax = fig.add_subplot(gs[r, 3:])
    xs = np.arange(len(names))
    ax.plot(xs, keep_all, "-o", color=SERIES[0], lw=2, ms=5, label="whole curve")
    ax.plot(xs, keep_hp, "-s", color=SERIES[1], lw=2, ms=5, label="fringe detail (high-pass)")
    ax.set_xticks(xs, names, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(min(-0.05, min(keep_hp) - 0.05), 1.02)
    ax.set_ylabel("decoder R²")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "How much of the input curve each stage still encodes")

    fig.suptitle(
        f"Reflectivity curve morphing through the network — {meta['run']}/{meta['checkpoint']}  "
        f"({meta['activation']}, reg={meta['regularizer']}{', normalized' if meta.get('normalize_inputs') else ''}, epoch {meta['epoch']})",
        fontsize=15.5, color=INK,
    )
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return {"stages": names, "curve_r2": keep_all, "fringe_r2": keep_hp}


# --------------------------------------------------------------------------------------
# Figure 2: layer cards
# --------------------------------------------------------------------------------------


def receptive_fields(model, s, p, q, n_samples=128, chunk=16):
    """Mean |d unit / d (S,P)(q index)| per block -> list of arrays (units, n_q)."""
    x_all = model.prepare_inputs(s[:n_samples], p[:n_samples], q[:n_samples]).detach()
    layout = block_layout(model)
    fields = None
    for start in range(0, len(x_all), chunk):
        x = x_all[start:start + chunk].clone().requires_grad_(True)
        h = x
        outs = []
        for op, act in layout:
            if isinstance(op, nn.Linear) and h.dim() == 3:
                h = h.flatten(1)
            h = op(h)
            h = act(h) if act is not None else h
            outs.append(h.flatten(1))
        acc = []
        for o in outs:
            units = o.shape[1]
            rf = torch.zeros(units, x.shape[-1])
            # One backward pass per unit is fine at this size; vectorize via is_grads_batched.
            eye = torch.eye(units).unsqueeze(1).expand(units, o.shape[0], units)
            (g,) = torch.autograd.grad(o, x, grad_outputs=eye, retain_graph=True, is_grads_batched=True)
            rf = g[:, :, :2].abs().sum(dim=(1, 2))  # (units, n_q), summed over samples and S/P
            acc.append(rf)
        fields = acc if fields is None else [a + b for a, b in zip(fields, acc)]
    return [(f / len(x_all)).numpy() for f in fields]


def plot_layer_card(idx, blk, rf, y, splits, path, meta):
    name = blk["name"]
    post = blk["post"].flatten(1).numpy()
    pre = blk["pre"]
    # For the mixer, show per-channel units (channels x q) folded to channels.
    is_mixer = isinstance(blk["op"], nn.Conv1d)
    units_post = blk["post"].permute(0, 2, 1).reshape(-1, blk["post"].shape[1]).numpy() if is_mixer else post
    units_pre = pre.permute(0, 2, 1).reshape(-1, pre.shape[1]).numpy() if is_mixer else pre.numpy()
    has_act = blk["act"] is not None
    frac_on = (units_pre > 0).mean(0)
    ynp = y.numpy()
    thick = ynp[:, 0]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), layout="constrained")

    # (a) receptive fields: which q indices drive each unit (units sorted by peak position).
    ax = axes[0, 0]
    rf_n = rf / (rf.max(1, keepdims=True) + 1e-12)
    order = np.argsort(np.argmax(rf_n, 1))
    ax.imshow(rf_n[order], aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("q index (0 .. 255)")
    ax.set_ylabel("unit (sorted by peak q)")
    _style(ax, "Receptive fields: |∂unit/∂(S,P)| along the curve")

    # (b) activation map: rows = samples sorted by thickness, cols = units sorted by corr w/ thickness.
    ax = axes[0, 1]
    sub = np.argsort(thick)[:: max(1, len(thick) // 400)]
    a = units_post if not is_mixer else post
    a_sub = a[sub] if not is_mixer else post[sub]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.array([np.corrcoef(a_sub[:, j], thick[sub])[0, 1] if a_sub[:, j].std() > 0 else 0 for j in range(a_sub.shape[1])]))
    col_order = np.argsort(corr)
    lim = np.percentile(np.abs(a_sub), 99) + 1e-9
    cmap = "viridis" if has_act else "RdBu_r"
    vmin = 0 if has_act and blk["act"].__class__.__name__ == "ReLU" else -lim
    ax.imshow(a_sub[:, col_order], aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=lim)
    ax.set_xlabel("unit (sorted by correlation with thickness)")
    ax.set_ylabel(f"samples, thin → thick ({thick[sub].min():.0f}–{thick[sub].max():.0f} Å)")
    _style(ax, "Activations vs film thickness")

    # (c) pre-activation distribution + firing-rate histogram.
    ax = axes[0, 2]
    flat = units_pre.ravel()
    lo, hi = np.percentile(flat, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(flat[flat <= 0], bins=bins, color=SERIES[1], alpha=0.85, label="≤ 0")
    ax.hist(flat[flat > 0], bins=bins, color=SERIES[0], alpha=0.85, label="> 0")
    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.set_xlabel("pre-activation value")
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    inset = ax.inset_axes([0.6, 0.55, 0.37, 0.4])
    inset.hist(frac_on, bins=np.linspace(0, 1, 21), color=SERIES[2])
    inset.set_title("unit firing rate", fontsize=10)
    inset.tick_params(labelsize=8.5)
    dead = 100 * (frac_on == 0).mean()
    always = 100 * (frac_on > 0.99).mean()
    _style(ax, f"Pre-activations  ({dead:.0f}% dead, {always:.0f}% always on)" if has_act else "Linear outputs (no activation)")

    # (d) label tuning: which label each unit is most correlated with + probe R^2.
    ax = axes[1, 0]
    feats = post
    tr, va, te = splits
    from diagnostics import ridge_probe
    probe = ridge_probe(feats, (ynp - ynp.min(0)) / (np.ptp(ynp, 0) + 1e-12), splits)
    live = feats.std(0) > 1e-8
    C = np.zeros((len(LABEL_NAMES), feats.shape[1]))
    for k in range(len(LABEL_NAMES)):
        yk = ynp[:, k]
        fz = (feats[:, live] - feats[:, live].mean(0)) / feats[:, live].std(0)
        C[k, live] = np.abs((fz * ((yk - yk.mean()) / yk.std())[:, None]).mean(0))
    best = np.bincount(C[:, live].argmax(0), minlength=len(LABEL_NAMES)) / max(live.sum(), 1)
    xs = np.arange(len(LABEL_NAMES))
    ax.bar(xs - 0.2, best, 0.4, color=SERIES[4], label="share of units tuned to label")
    ax.bar(xs + 0.2, probe, 0.4, color=SERIES[0], label="linear-probe R²")
    ax.set_xticks(xs, LABEL_NAMES, rotation=25, ha="right", fontsize=10.5)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "What the layer encodes")

    # (e) PCA coloured by thickness.
    ax = axes[1, 1]
    emb, var = pca2(feats[te])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=thick[te], cmap="viridis", s=4, alpha=0.7, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.04, label="film thickness [Å]")
    ax.set_xticks([])
    ax.set_yticks([])
    _style(ax, f"PCA of activations (PC1+PC2 = {100*var[:2].sum():.0f}% var)")

    # (f) weight singular values + explained variance of activations.
    ax = axes[1, 2]
    W = blk["op"].weight.detach().reshape(blk["op"].weight.shape[0], -1).double()
    sv = torch.linalg.svdvals(W).numpy()
    ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], color=SERIES[0], lw=2, label="weight σ / σ_max")
    ax.semilogy(np.arange(1, min(len(var), len(sv)) + 1), var[: len(sv)] / var[0], color=SERIES[1], lw=2, label="activation PCA var / max")
    ax.set_xscale("log")
    ax.set_xlabel("component")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Spectra")

    fig.suptitle(
        f"Layer card {idx:02d}: {name}  ({blk['act'].__class__.__name__ if has_act else 'linear'})  —  "
        f"{meta['run']}/{meta['checkpoint']}, epoch {meta['epoch']}",
        fontsize=15.5, color=INK,
    )
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_layer_cards(model, meta, data, out_dir, n_rows=4096):
    s, p, q, y = (t[:n_rows] for t in data)
    with torch.no_grad():
        _, blocks = unroll(model, s, p, q)
    rfs = receptive_fields(model, s, p, q)
    n = len(y)
    perm = np.random.default_rng(0).permutation(n)
    splits = (perm[: int(0.6 * n)], perm[int(0.6 * n): int(0.8 * n)], perm[int(0.8 * n):])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (blk, rf) in enumerate(zip(blocks, rfs)):
        safe = blk["name"].replace(" ", "_").replace(">", "").replace("-", "_")
        plot_layer_card(i, blk, rf, y, splits, out_dir / f"{i:02d}_{safe}.png", meta)


def render_all(ckpt_path: Path, data, out_dir: Path, tag: str):
    model, criterion, meta = load_model(ckpt_path)
    morph = plot_morph(model, criterion, meta, data, out_dir / f"{tag}_morph.png")
    plot_layer_cards(model, meta, data, out_dir / f"{tag}_cards")
    return morph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-rows", type=int, default=8192)
    args = ap.parse_args()
    out = args.out_dir or PROJECT_ROOT / "analysis" / "runs" / args.checkpoint.parent.name
    out.mkdir(parents=True, exist_ok=True)
    data = load_val_rows(args.n_rows)
    m = render_all(args.checkpoint, data, out, args.checkpoint.stem)
    print("stage curve R²:", dict(zip(m["stages"], np.round(m["curve_r2"], 3))))
    print("stage fringe R²:", dict(zip(m["stages"], np.round(m["fringe_r2"], 3))))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
