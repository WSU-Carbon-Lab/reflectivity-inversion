"""Quantitative layer-by-layer health diagnostics for ReflectivityNet checkpoints.

Where ``activations.py`` *draws* a handful of rows flowing through the network,
this script *measures* how the network behaves across thousands of held-out
validation rows, and flags patterns that suggest the architecture or training
is fundamentally off:

  * Dead / always-on units      -> wasted capacity, or a layer that is effectively linear
  * Activation sparsity         -> how much of each layer is actually used per sample
  * Signal gain (RMS out / in)  -> where activations are intensified or squashed
  * Spike ratio (max / RMS)     -> a few units carrying huge values
  * Effective rank              -> representational collapse / bottlenecks
  * Unit redundancy             -> many neurons computing the same thing
  * Linear-probe R^2 per layer  -> where label information appears or is lost
  * Weight spectra              -> stable rank, spectral norm, collapsed linear chains
  * Gradient flow               -> vanishing / exploding gradients per layer
  * Input saliency vs q index   -> which parts of the curves the network actually reads

Usage:
    uv run python analysis/diagnostics.py --all              # every run in checkpoints/
    uv run python analysis/diagnostics.py --checkpoint checkpoints/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from loss import LABEL_NAMES, DepthProfileLoss, build_loss  # noqa: E402
from network import ReflectivityNet  # noqa: E402

# Rows 900,000+ (the last shard) are the validation split for every run trained with
# the default --val-split 0.10 on the full 1M dataset, so none were seen in training.
VAL_SHARD = PROJECT_ROOT / "data/reflectivity-1m/data/train-00009-of-00010.parquet"
OUT_ROOT = PROJECT_ROOT / "analysis" / "runs"

# Categorical slots (fixed order) from the dataviz reference palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5bd6", "#8c8c8c"]
INK = "#2b2b2b"
MUTED = "#8a8a85"

# Figure style: large enough to read when a figure is viewed at page width (e.g. on the Hub).
FIG_DPI = 220
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
ACTIVATIONS = (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.Tanh)

# Thresholds for automatically raised red flags.
FLAG = {
    "dead_frac": 0.20,         # >20% of units never fire on any validation row
    "always_on_frac": 0.50,    # >50% of units always fire -> layer behaves linearly
    "min_eff_rank": 7,         # fewer than 7 directions carry 99% of variance -> bottleneck
    "gain_hi": 4.0,            # activations grow > 4x through one block
    "gain_lo": 0.25,           # activations shrink > 4x through one block
    "spike": 25.0,             # single activation > 25x the layer RMS
    "redundant_frac": 0.10,    # >10% of live unit pairs have |corr| > 0.95
    "probe_drop": 0.03,        # linear-probe R^2 drops by more than this between layers
    "grad_spread": 100.0,      # largest / smallest per-layer grad-to-weight ratio
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_model(ckpt_path: Path):
    """Rebuild the exact architecture saved in a checkpoint (ReLU or GELU, any ReLU layout)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    sd = ckpt["model_state_dict"]
    activation = cfg.get("activation") or ("gelu" if cfg.get("model") == "netg" else "relu")
    act_cls = nn.GELU if activation == "gelu" else nn.ReLU

    # Checkpoints saved before --regularizer / --loss existed used log_rq4 + label MSE.
    regularizer = cfg.get("regularizer") or "log_rq4"
    loss_name = cfg.get("loss") or "mse"
    normalize = "input_mean" in sd

    model = ReflectivityNet(
        mix_channels=cfg.get("mix_channels", 8), regularizer=regularizer,
        activation=activation, normalize_inputs=normalize,
    )
    conv_w = sd["channel_mixer.0.weight"]
    model.channel_mixer = nn.Sequential(
        nn.Conv1d(conv_w.size(1), conv_w.size(0), kernel_size=1), act_cls(), nn.Flatten(1)
    )
    # Any mlp index without a weight was a parameter-free activation when saved.
    idxs = sorted(int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight"))
    modules = []
    for i in range(idxs[-1] + 1):
        w = sd.get(f"mlp.{i}.weight")
        modules.append(nn.Linear(w.shape[1], w.shape[0]) if w is not None else act_cls())
    model.mlp = nn.Sequential(*modules)
    model.load_state_dict(sd)
    model.eval()

    meta = {
        "run": ckpt_path.parent.name,
        "checkpoint": ckpt_path.name,
        "epoch": int(ckpt.get("epoch", -1)),
        "val_loss": float(ckpt.get("val_loss", float("nan"))),
        "planned_epochs": cfg.get("epochs"),
        "model": cfg.get("model") or "net",
        "activation": act_cls.__name__,
        "normalize_inputs": normalize,
        "lr_schedule": cfg.get("lr_schedule", "constant"),
        "run_name": cfg.get("run_name"),
        "regularizer": regularizer,
        "loss": loss_name,
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
    }
    return model, build_loss(loss_name), meta


def load_val_rows(n: int):
    """Load the first ``n`` rows of the held-out validation shard."""
    df = pl.read_parquet(VAL_SHARD, n_rows=n)
    s = torch.from_numpy(df["s"].to_numpy().copy())
    p = torch.from_numpy(df["p"].to_numpy().copy())
    q = torch.from_numpy(df["q"].to_numpy().copy())
    y = torch.from_numpy(df.select(LABEL_NAMES).to_numpy().copy())
    return s, p, q, y


# --------------------------------------------------------------------------------------
# Unrolling into blocks
# --------------------------------------------------------------------------------------


def block_layout(model):
    """Group modules into (linear op, activation or None) blocks in forward order."""
    seq = list(model.channel_mixer) + list(model.mlp)
    blocks, i = [], 0
    while i < len(seq):
        m = seq[i]
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            act = seq[i + 1] if i + 1 < len(seq) and isinstance(seq[i + 1], ACTIVATIONS) else None
            blocks.append((m, act))
            i += 2 if act is not None else 1
        else:
            i += 1
    return blocks


def block_name(op, idx):
    if isinstance(op, nn.Conv1d):
        return f"mixer {op.in_channels}->{op.out_channels}"
    return f"L{idx} {op.in_features}->{op.out_features}"


def unroll(model, s, p, q):
    """Return the stacked input plus (name, op, act, input, pre, post) for each block."""
    x = model.prepare_inputs(s, p, q)
    stacked = x
    out = []
    for idx, (op, act) in enumerate(block_layout(model)):
        if isinstance(op, nn.Linear) and x.dim() == 3:
            x = x.flatten(1)
        pre = op(x)
        post = act(pre) if act is not None else pre
        out.append(
            {"name": block_name(op, idx), "op": op, "act": act, "input": x, "pre": pre, "post": post}
        )
        x = post
    return stacked, out


def as_units(t):
    """View activations as (samples, units); Conv1d channels are treated per q index."""
    return t.permute(0, 2, 1).reshape(-1, t.size(1)) if t.dim() == 3 else t


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def effective_rank(feats: np.ndarray):
    """Participation ratio and #components for 99% variance of centered features."""
    f = feats - feats.mean(0, keepdims=True)
    sv = np.linalg.svd(f, compute_uv=False)
    lam = sv**2
    if lam.sum() <= 0:
        return 0.0, 0
    pr = float(lam.sum() ** 2 / (lam**2).sum())
    n99 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.99) + 1)
    return pr, n99


def redundancy(feats: np.ndarray, live: np.ndarray):
    """Fraction of live-unit pairs whose activations correlate with |r| > 0.95."""
    f = feats[:, live]
    if f.shape[1] < 2:
        return 0.0
    std = f.std(0)
    f = f[:, std > 1e-8]
    if f.shape[1] < 2:
        return 0.0
    c = np.corrcoef(f, rowvar=False)
    iu = np.triu_indices_from(c, k=1)
    return float((np.abs(c[iu]) > 0.95).mean())


def ridge_probe(feats: np.ndarray, y: np.ndarray, splits):
    """Linear-probe R^2 per label: fit ridge on train, pick lambda on val, score on test."""
    tr, va, te = splits
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
    X = (feats - mu) / sd
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    ym = y[tr].mean(0)
    Y = y - ym

    # Solve in whichever space is smaller (features or samples).
    Xtr, Ytr = X[tr], Y[tr]
    use_dual = Xtr.shape[1] > Xtr.shape[0]
    G = Xtr @ Xtr.T if use_dual else Xtr.T @ Xtr
    scale = np.trace(G) / G.shape[0]
    best, best_err = None, np.inf
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        reg = G + lam * scale * np.eye(G.shape[0])
        W = Xtr.T @ np.linalg.solve(reg, Ytr) if use_dual else np.linalg.solve(reg, Xtr.T @ Ytr)
        err = ((X[va] @ W - Y[va]) ** 2).mean()
        if err < best_err:
            best, best_err = W, err
    pred = X[te] @ best
    ss_res = ((pred - Y[te]) ** 2).sum(0)
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0) + 1e-12
    return (1.0 - ss_res / ss_tot).tolist()


def weight_stats(op):
    W = op.weight.detach().reshape(op.weight.size(0), -1).double()
    sv = torch.linalg.svdvals(W).numpy()
    fro2 = float((sv**2).sum())
    lam = sv**2
    return {
        "spectral_norm": float(sv[0]),
        "stable_rank": fro2 / float(sv[0] ** 2),
        "sv_participation": float(lam.sum() ** 2 / (lam**2).sum()),
        "max_rank": int(min(W.shape)),
        "singular_values": sv.tolist(),
    }


def chained_linear_rank(blocks):
    """Detect consecutive linear ops with no activation between them (they collapse to one)."""
    chains = []
    for i in range(len(blocks) - 1):
        op, act = blocks[i]
        nxt, _ = blocks[i + 1]
        if act is None and isinstance(op, nn.Linear) and isinstance(nxt, nn.Linear):
            prod = (nxt.weight @ op.weight).detach().double()
            sv = torch.linalg.svdvals(prod).numpy()
            chains.append(
                {
                    "blocks": [block_name(op, i), block_name(nxt, i + 1)],
                    "composite_shape": list(prod.shape),
                    "rank_bound": int(min(op.weight.shape[0], *prod.shape)),
                    "numerical_rank": int((sv > sv[0] * 1e-3).sum()),
                    "wide_layer_width": int(nxt.weight.shape[0]),
                }
            )
    return chains


def gradient_flow(model, criterion, s, p, q, y):
    """Per-layer weight gradient norm and grad/weight ratio on one batch of the training loss."""
    model.zero_grad()
    loss = criterion(model(s, p, q), y, q)
    loss.backward()
    rows = []
    for idx, (op, _) in enumerate(block_layout(model)):
        g = op.weight.grad.norm().item()
        w = op.weight.norm().item()
        rows.append({"name": block_name(op, idx), "grad_norm": g, "grad_to_weight": g / (w + 1e-12)})
    model.zero_grad()
    return rows


def input_saliency(model, s, p, q):
    """Mean |d output_k / d input| x std(input) -> array (7 labels, 3 channels, n_q)."""
    x = model.prepare_inputs(s, p, q).detach().requires_grad_(True)
    # Scale gradients by each input's natural variation so channels with very different
    # units (log10 R ~ -8..0 vs q ~ 0.01..0.5 1/Å) are compared fairly.
    x_std = x.detach().std(0)
    out = model.mlp(model.channel_mixer(x))
    sal = []
    for k in range(out.size(1)):
        (g,) = torch.autograd.grad(out[:, k].sum(), x, retain_graph=True)
        sal.append((g.abs() * x_std).mean(0))
    return torch.stack(sal).numpy()


def mixer_analysis(model, stacked):
    """How much of each mixer channel's variance comes from S, P, and raw q inputs."""
    conv = model.channel_mixer[0]
    w = conv.weight.detach().squeeze(-1)  # (mix, 3)
    x = stacked.permute(0, 2, 1).reshape(-1, 3)
    contrib = w.unsqueeze(0) * x.unsqueeze(1)  # (N*n_q, mix, 3)
    var = contrib.var(0)  # (mix, 3)
    frac = (var / var.sum(1, keepdim=True).clamp_min(1e-12)).numpy()
    # Polarization sensitivity: S-P difference relative to S+P common mode.
    ws, wp = w[:, 0], w[:, 1]
    anis = ((ws - wp).abs() / ((ws + wp).abs() + (ws - wp).abs() + 1e-12)).numpy()
    return {
        "weights": w.numpy().tolist(),
        "bias": conv.bias.detach().numpy().tolist(),
        "variance_frac_S_P_q": frac.tolist(),
        "polarization_contrast": anis.tolist(),
        "input_channel_mean": x.mean(0).tolist(),
        "input_channel_std": x.std(0).tolist(),
    }


@torch.no_grad()
def layer_metrics(stacked, blocks, y_norm, splits):
    rows = []
    ynp = y_norm.numpy()
    in_feats = stacked.flatten(1).numpy()
    probe_in = ridge_probe(in_feats, ynp, splits)
    for b in blocks:
        pre_u = as_units(b["pre"])
        post_u = as_units(b["post"])
        in_rms = b["input"].pow(2).mean().sqrt().item()
        post_rms = b["post"].pow(2).mean().sqrt().item()
        active = (pre_u > 0).float()
        frac_active = active.mean(0).numpy()
        has_act = b["act"] is not None
        post_np = post_u.numpy()
        live = frac_active > 0 if has_act else np.ones_like(frac_active, bool)

        unit_var = post_u.var(0).numpy()
        top = max(1, int(round(0.01 * len(unit_var))))
        var_top1 = float(np.sort(unit_var)[::-1][:top].sum() / max(unit_var.sum(), 1e-12))

        # Effective rank on per-sample feature vectors (flattened for the conv mixer).
        feats = b["post"].flatten(1).numpy()
        pr, n99 = effective_rank(feats)
        rows.append(
            {
                "name": b["name"],
                "width": int(feats.shape[1]),
                "has_activation": has_act,
                "activation": type(b["act"]).__name__ if has_act else None,
                "in_rms": in_rms,
                "pre_rms": b["pre"].pow(2).mean().sqrt().item(),
                "post_rms": post_rms,
                "gain": post_rms / (in_rms + 1e-12),
                "spike_ratio": b["post"].abs().max().item() / (post_rms + 1e-12),
                "sparsity": float(1.0 - active.mean().item()) if has_act else 0.0,
                "dead_frac": float((frac_active == 0).mean()) if has_act else 0.0,
                "rare_frac": float(((frac_active > 0) & (frac_active < 0.01)).mean()) if has_act else 0.0,
                "always_on_frac": float((frac_active > 0.99).mean()) if has_act else 1.0,
                "var_top1pct_units": var_top1,
                "eff_rank": pr,
                "eff_rank_frac": pr / feats.shape[1],
                "n99": n99,
                "redundant_frac": redundancy(post_np, live),
                "probe_r2": ridge_probe(feats, ynp, splits),
                "unit_frac_active": frac_active.tolist(),
            }
        )
    return probe_in, rows


def raise_flags(m):
    flags = []
    for L in m["layers"]:
        n = L["name"]
        if L["has_activation"]:
            if L["dead_frac"] > FLAG["dead_frac"]:
                flags.append(f"{n}: {100*L['dead_frac']:.0f}% of units are dead (never fire on {m['n_rows']} val rows)")
            if L["always_on_frac"] > FLAG["always_on_frac"]:
                flags.append(f"{n}: {100*L['always_on_frac']:.0f}% of units always fire -> the activation is doing almost nothing (layer ~linear)")
        if L is not m["layers"][-1] and L["n99"] < FLAG["min_eff_rank"]:
            flags.append(f"{n}: only {L['n99']} directions hold 99% of the variance (< {FLAG['min_eff_rank']} labels) "
                         f"-> bottleneck narrower than the target space")
        if L["gain"] > FLAG["gain_hi"]:
            flags.append(f"{n}: activations intensified {L['gain']:.1f}x (RMS {L['in_rms']:.2g} -> {L['post_rms']:.2g})")
        if L["gain"] < FLAG["gain_lo"]:
            flags.append(f"{n}: activations attenuated to {L['gain']:.2f}x (RMS {L['in_rms']:.2g} -> {L['post_rms']:.2g})")
        if L["spike_ratio"] > FLAG["spike"]:
            flags.append(f"{n}: spike ratio {L['spike_ratio']:.0f} (max |a| / RMS) -> a few units carry outsized values")
        if L["redundant_frac"] > FLAG["redundant_frac"]:
            flags.append(f"{n}: {100*L['redundant_frac']:.0f}% of live unit pairs are near-duplicates (|r| > 0.95)")
    prev_name, prev = "input", np.mean(m["probe_input"])
    for L in m["layers"]:
        cur = np.mean(L["probe_r2"])
        if prev - cur > FLAG["probe_drop"]:
            flags.append(f"{L['name']}: linear-probe mean R^2 drops {prev:.3f} -> {cur:.3f} vs {prev_name} (information lost or entangled)")
        prev_name, prev = L["name"], cur
    for c in m["linear_chains"]:
        flags.append(
            f"{c['blocks'][0]} and {c['blocks'][1]} have no activation between them: they collapse to one "
            f"{c['composite_shape'][0]}x{c['composite_shape'][1]} map of rank <= {c['rank_bound']} "
            f"(numerical rank {c['numerical_rank']}); the {c['wide_layer_width']}-wide layer can only carry a "
            f"{c['rank_bound']}-dim signal and the pair is no more expressive than a single linear layer"
        )
    ratios = [g["grad_to_weight"] for g in m["gradient_flow"] if g["grad_to_weight"] > 0]
    if ratios and max(ratios) / min(ratios) > FLAG["grad_spread"]:
        flags.append(f"gradient/weight ratio spans {max(ratios)/min(ratios):.0f}x across layers -> uneven learning speed")
    sal = np.array(m["saliency"]).sum(0)  # (3, n_q)
    ch_tot = sal.sum(1)
    if ch_tot[2] > 0.5 * ch_tot.sum():
        flags.append(f"raw q channel receives {100*ch_tot[2]/ch_tot.sum():.0f}% of input saliency -> network keys on the q grid more than on reflectivity")
    sp = sal[:2].sum(0)
    dead_q = float((sp < 0.02 * sp.max()).mean())
    if dead_q > 0.25:
        flags.append(f"{100*dead_q:.0f}% of q indices get <2% of peak S/P saliency -> large parts of the curve are ignored")
    mix = m["mixer"]
    if max(mix["polarization_contrast"]) < 0.2:
        flags.append("mixer weights treat S and P almost identically -> little access to the S/P anisotropy that sets delta_xx vs delta_zz")
    return flags


def analyze(ckpt_path: Path, data, n_sal: int, n_grad: int):
    model, criterion, meta = load_model(ckpt_path)
    s, p, q, y = data
    n = len(y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    splits = (perm[: int(0.6 * n)], perm[int(0.6 * n) : int(0.8 * n)], perm[int(0.8 * n) :])

    with torch.no_grad():
        stacked, blocks = unroll(model, s, p, q)
        out = blocks[-1]["post"]
        y_norm = criterion.normalize(y)
        profile = DepthProfileLoss()
        val_metrics = {
            "label_mse": float((out - y_norm).pow(2).mean()),
            "profile_l2": float(profile(out, y, q)),
            "thickness_mae_A": float((criterion.denormalize(out)[:, 0] - y[:, 0]).abs().mean()),
        }
    probe_in, layers = layer_metrics(stacked, blocks, y_norm, splits)
    # Output-layer probe on the actual prediction (not refit) for comparison.
    out_r2 = 1 - ((out - y_norm) ** 2).sum(0) / ((y_norm - y_norm.mean(0)) ** 2).sum(0)

    m = {
        **meta,
        "n_rows": n,
        "val_metrics": val_metrics,
        "output_r2_per_label": out_r2.tolist(),
        "probe_input": probe_in,
        "input_eff_rank": effective_rank(stacked.flatten(1).numpy())[0],
        "layers": layers,
        "weights": [
            {"name": block_name(op, i), **weight_stats(op)} for i, (op, _) in enumerate(block_layout(model))
        ],
        "linear_chains": chained_linear_rank(block_layout(model)),
        "gradient_flow": gradient_flow(model, criterion, s[:n_grad], p[:n_grad], q[:n_grad], y[:n_grad]),
        "saliency": input_saliency(model, s[:n_sal], p[:n_sal], q[:n_sal]).tolist(),
        "mixer": mixer_analysis(model, stacked),
    }
    m["flags"] = raise_flags(m)
    return m


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------


def _style(ax, title):
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#4a4a46", labelsize=10.5)


def plot_layer_health(m, path):
    L = m["layers"]
    names = [l["name"] for l in L]
    x = np.arange(len(L))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")

    ax = axes[0, 0]
    w = 0.26
    ax.bar(x - w, [100 * l["dead_frac"] for l in L], w, color=SERIES[1], label="dead (never fires)")
    ax.bar(x, [100 * l["sparsity"] for l in L], w, color=SERIES[0], label="mean sparsity")
    ax.bar(x + w, [100 * (l["always_on_frac"] if l["has_activation"] else np.nan) for l in L], w, color=SERIES[3], label="always on (~linear)")
    for i, l in enumerate(L):
        if not l["has_activation"]:
            ax.text(i, 3, "no act.", ha="center", fontsize=10, color=MUTED, rotation=90)
    ax.set_ylabel("% of units")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Unit usage")

    ax = axes[0, 1]
    ax.plot(x, [l["gain"] for l in L], "-o", color=SERIES[0], lw=2, ms=6, label="gain (RMS out / in)")
    ax.plot(x, [l["spike_ratio"] / 10 for l in L], "-s", color=SERIES[1], lw=2, ms=6, label="spike ratio / 10")
    ax.axhline(1, color=MUTED, ls=":", lw=1)
    ax.set_yscale("log")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Signal intensification per block")

    ax = axes[0, 2]
    ax.plot(x, [100 * l["eff_rank_frac"] for l in L], "-o", color=SERIES[2], lw=2, ms=6, label="eff. rank / width")
    ax.plot(x, [100 * l["redundant_frac"] for l in L], "-s", color=SERIES[4], lw=2, ms=6, label="near-duplicate unit pairs")
    for i, l in enumerate(L):
        ax.annotate(f"{l['eff_rank']:.0f}", (i, 100 * l["eff_rank_frac"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10, color=INK)
    ax.set_ylabel("%")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Rank & redundancy (numbers = participation ratio)")

    ax = axes[1, 0]
    probes = np.array([m["probe_input"]] + [l["probe_r2"] for l in L])  # (stages, 7)
    xs = np.arange(len(probes))
    for k, lab in enumerate(LABEL_NAMES):
        ax.plot(xs, probes[:, k], "-o", color=SERIES[k], lw=2, ms=4, label=lab)
    ax.plot(xs, probes.mean(1), color=INK, lw=2.5, ls="--", label="mean")
    ax.set_xticks(xs, ["input"] + names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(min(0, probes.min()) - 0.02, 1.02)
    ax.set_ylabel("held-out R²")
    ax.legend(fontsize=9.5, frameon=False, ncol=2)
    _style(ax, "Linear probe: label information available at each stage")

    ax = axes[1, 1]
    g = m["gradient_flow"]
    ax.bar(np.arange(len(g)), [r["grad_to_weight"] for r in g], color=SERIES[0], width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(g)), [r["name"] for r in g], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("‖∇W‖ / ‖W‖")
    _style(ax, f"Gradient flow (one batch, loss={m['loss']})")

    ax = axes[1, 2]
    for i, wst in enumerate(m["weights"]):
        sv = np.array(wst["singular_values"])
        ax.plot(np.linspace(0, 1, len(sv)), sv / sv[0], color=SERIES[i % len(SERIES)], lw=1.8, label=f"{wst['name']} (stable rank {wst['stable_rank']:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("singular value index (fraction of max rank)")
    ax.set_ylabel("σ / σ_max")
    ax.legend(fontsize=9.5, frameon=False)
    _style(ax, "Weight singular-value spectra")

    for a in axes[0]:
        a.set_xticks(x, names, rotation=35, ha="right", fontsize=10)
    vm = m["val_metrics"]
    fig.suptitle(
        f"{m['run']}/{m['checkpoint']}  (epoch {m['epoch']}, {m['activation']}, reg={m['regularizer']}, loss={m['loss']})   "
        f"val label MSE {vm['label_mse']:.2e} | profile L2 {vm['profile_l2']:.3f} | thickness MAE {vm['thickness_mae_A']:.1f} Å",
        fontsize=14.5, color=INK,
    )
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_saliency(m, path):
    sal = np.array(m["saliency"])  # (7, 3, n_q)
    n_q = sal.shape[2]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]})

    ax = axes[0, 0]
    norm = sal[:, :2].sum(1)
    norm = norm / norm.max(1, keepdims=True)
    im = ax.imshow(norm, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(7), LABEL_NAMES, fontsize=10.5)
    ax.set_xlabel("q index (0 .. 255)")
    fig.colorbar(im, ax=ax, fraction=0.025, label="relative |∂y/∂(S,P)|")
    _style(ax, "Where along the curve each label reads from (S+P saliency, row-normalized)")

    ax = axes[1, 0]
    tot = sal.sum(0)
    for c, (lab, col) in enumerate(zip(("S channel", "P channel", "raw q channel"), (SERIES[0], SERIES[1], SERIES[2]))):
        ax.plot(np.arange(n_q), tot[c], color=col, lw=2, label=lab)
    ax.set_xlabel("q index (0 .. 255)")
    ax.set_ylabel("Σ_labels mean |∂y/∂x|")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Input saliency by channel")

    ax = axes[0, 1]
    frac = np.array(m["mixer"]["variance_frac_S_P_q"])
    mix = np.arange(frac.shape[0])
    bottom = np.zeros(len(mix))
    for c, (lab, col) in enumerate(zip(("S", "P", "q"), (SERIES[0], SERIES[1], SERIES[2]))):
        ax.bar(mix, frac[:, c], bottom=bottom, color=col, width=0.65, label=lab, edgecolor="white", linewidth=1)
        bottom += frac[:, c]
    alive = [m["layers"][0]["unit_frac_active"][i] for i in mix]
    ax.set_xticks(mix, [f"ch{i}\n{100*a:.0f}% on" for i, a in enumerate(alive)], fontsize=10)
    ax.set_ylabel("share of pre-activation variance")
    ax.legend(fontsize=10.5, frameon=False, ncol=3)
    _style(ax, "Mixer: what each channel listens to")

    ax = axes[1, 1]
    ax.bar(mix, m["mixer"]["polarization_contrast"], color=SERIES[4], width=0.65)
    ax.set_ylim(0, 1)
    ax.set_xticks(mix, [f"ch{i}" for i in mix], fontsize=10.5)
    ax.set_ylabel("|w_S − w_P| / (|w_S + w_P| + |w_S − w_P|)")
    _style(ax, "Mixer S/P contrast (0 = sum, 1 = difference)")

    fig.suptitle(f"Input sensitivity — {m['run']}/{m['checkpoint']} (epoch {m['epoch']})", fontsize=14.5, color=INK)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_evolution(ms, path):
    """Track layer health across the periodic checkpoints of one run."""
    ms = sorted([m for m in ms if m["checkpoint"] != "best.pt"], key=lambda m: m["epoch"])
    if len(ms) < 2:
        return
    names = [l["name"] for l in ms[0]["layers"]]
    epochs = [m["epoch"] for m in ms]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    for metric, ax, title, scale in (
        ("dead_frac", axes[0], "Dead units (%)", 100),
        ("eff_rank_frac", axes[1], "Effective rank / width (%)", 100),
        ("gain", axes[2], "Gain (RMS out / in)", 1),
    ):
        for i, n in enumerate(names):
            ax.plot(epochs, [scale * m["layers"][i][metric] for m in ms], "-o", color=SERIES[i % len(SERIES)], lw=2, ms=5, label=n)
        ax.set_xlabel("epoch")
        _style(ax, title)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=9.5, frameon=False)
    fig.suptitle(f"Layer health over training — {ms[0]['run']}", fontsize=14.5, color=INK)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_summary(best, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    for i, m in enumerate(best):
        col = SERIES[i % len(SERIES)]
        lab = f"{m['run']} ({m['loss']}, {m['regularizer']})"
        stages = ["input"] + [l["name"].split(" ")[0] for l in m["layers"]]
        xs = np.arange(len(stages))
        axes[0].plot(xs, [np.mean(m["probe_input"])] + [np.mean(l["probe_r2"]) for l in m["layers"]], "-o", color=col, lw=2, ms=5, label=lab)
        axes[1].plot(xs[1:], [100 * l["dead_frac"] for l in m["layers"]], "-o", color=col, lw=2, ms=5, label=lab)
        axes[2].plot(xs[1:], [l["eff_rank"] for l in m["layers"]], "-o", color=col, lw=2, ms=5, label=lab)
    for ax, t in zip(axes, ("Mean linear-probe R²", "Dead units (%)", "Effective rank (participation ratio)")):
        ax.set_xticks(xs, stages, fontsize=10.5)
        _style(ax, t)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=10, frameon=False)
    fig.suptitle("All runs — best.pt", fontsize=14.5, color=INK)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Reporting / driver
# --------------------------------------------------------------------------------------


def run_status(run_dir: Path):
    ckpts = sorted(run_dir.glob("*.pt"))
    planned = None
    last = 0
    for c in ckpts:
        if c.name.startswith("epoch_"):
            last = max(last, int(c.stem.split("_")[1]))
    try:
        cfg = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False).get("config", {})
        planned = cfg.get("epochs")
    except FileNotFoundError:
        pass
    complete = planned is not None and last >= planned
    return complete, last, planned


def write_summary(best, statuses, path):
    lines = ["# Network diagnostics summary", ""]
    lines.append(f"Measured on {best[0]['n_rows']} held-out validation rows (shard 9). Each run's `best.pt`.\n")
    lines.append("| run | status | loss | reg | best epoch | label MSE | profile L2 | thick. MAE Å | mean probe R² (in → last hidden) | worst dead % | flags |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in best:
        st = statuses[m["run"]]
        status = "complete" if st[0] else f"stopped ({st[1]}/{st[2]})" if st[1] else f"no periodic ckpt (planned {st[2]})"
        vm = m["val_metrics"]
        dead = max(100 * l["dead_frac"] for l in m["layers"])
        pr_in = np.mean(m["probe_input"])
        pr_last = np.mean(m["layers"][-2]["probe_r2"])
        lines.append(
            f"| {m['run']} | {status} | {m['loss']} | {m['regularizer']} | {m['epoch']} | {vm['label_mse']:.2e} | "
            f"{vm['profile_l2']:.3f} | {vm['thickness_mae_A']:.1f} | {pr_in:.3f} → {pr_last:.3f} | {dead:.0f} | {len(m['flags'])} |"
        )
    lines.append("")
    for m in best:
        lines.append(f"## {m['run']}  ({m['activation']}, reg={m['regularizer']}, loss={m['loss']}, epoch {m['epoch']})\n")
        lines.append("| block | width | dead % | always-on % | sparsity % | gain | spike | eff. rank (PR / n99) | redundant % | probe R² (mean / thickness) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        lines.append(f"| input | {768} | – | – | – | – | – | {m['input_eff_rank']:.1f} | – | {np.mean(m['probe_input']):.3f} / {m['probe_input'][0]:.3f} |")
        for l in m["layers"]:
            act = l["has_activation"]
            lines.append(
                f"| {l['name']} | {l['width']} | {100*l['dead_frac']:.1f}" + ("" if act else "*") +
                f" | {100*l['always_on_frac']:.1f}" + ("" if act else "*") +
                f" | {100*l['sparsity']:.1f} | {l['gain']:.2f} | {l['spike_ratio']:.1f} | {l['eff_rank']:.1f} / {l['n99']} | "
                f"{100*l['redundant_frac']:.1f} | {np.mean(l['probe_r2']):.3f} / {l['probe_r2'][0]:.3f} |"
            )
        lines.append("\n(* = no activation after this block)\n")
        lines.append("Flags:" if m["flags"] else "Flags: none")
        for f in m["flags"]:
            lines.append(f"- {f}")
        lines.append("")
    path.write_text("\n".join(lines))


def checkpoints_for(run_dir: Path, every: bool):
    best = run_dir / "best.pt"
    out = [best] if best.exists() else []
    if every:
        out += sorted(run_dir.glob("epoch_*.pt"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="Analyze a single checkpoint")
    ap.add_argument("--all", action="store_true", help="Analyze every run in checkpoints/")
    ap.add_argument("--completed-only", action="store_true", help="With --all, skip runs that stopped early")
    ap.add_argument("--best-only", action="store_true", help="Skip periodic epoch_*.pt checkpoints")
    ap.add_argument("--n-rows", type=int, default=6144, help="Validation rows for statistics")
    ap.add_argument("--n-saliency", type=int, default=1024)
    ap.add_argument("--n-grad", type=int, default=1024)
    ap.add_argument("--no-unroll", action="store_true", help="Skip the perceptron morph / layer-card figures")
    ap.add_argument("--legacy-unroll", action="store_true", help="Also render the old activations.py figures")
    ap.add_argument("--force", action="store_true", help="Recompute even if results are newer than the checkpoint")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--runs", nargs="+", help="With --all, only these checkpoint ids")
    args = ap.parse_args()
    if not args.checkpoint and not args.all:
        ap.error("pass --checkpoint <path> or --all")

    torch.set_grad_enabled(True)
    print(f"Loading {args.n_rows} validation rows from {VAL_SHARD.name} ...")
    data = load_val_rows(args.n_rows)
    _viz_cache = {}

    def viz_data():
        if "d" not in _viz_cache:
            _viz_cache["d"] = load_val_rows(8192)
        return _viz_cache["d"]

    if args.checkpoint:
        run_dirs = {args.checkpoint.parent: [args.checkpoint]}
    else:
        run_dirs = {
            d: checkpoints_for(d, not args.best_only)
            for d in sorted((PROJECT_ROOT / "checkpoints").iterdir())
            if d.is_dir() and (d / "best.pt").exists() and (not args.runs or d.name in args.runs)
        }

    statuses, best_metrics = {}, []
    for run_dir, ckpts in run_dirs.items():
        statuses[run_dir.name] = run_status(run_dir)
        if args.completed_only and not statuses[run_dir.name][0]:
            print(f"skip {run_dir.name}: incomplete {statuses[run_dir.name]}")
            continue
        run_out = args.out_dir / run_dir.name
        run_out.mkdir(parents=True, exist_ok=True)
        run_ms = []
        for ck in ckpts:
            tag = ck.stem
            jpath = run_out / f"{tag}.json"
            if jpath.exists() and not args.force and jpath.stat().st_mtime > ck.stat().st_mtime:
                m = json.loads(jpath.read_text())
                print(f"cached  {run_dir.name}/{ck.name}")
            else:
                print(f"analyze {run_dir.name}/{ck.name} ...", flush=True)
                m = analyze(ck, data, args.n_saliency, args.n_grad)
                jpath.write_text(json.dumps(m))
                plot_layer_health(m, run_out / f"{tag}_layer_health.png")
                plot_saliency(m, run_out / f"{tag}_saliency.png")
                if (tag == "best" or ck == ckpts[-1]) and not args.no_unroll:
                    # Standardized perceptron figures: curve morph + per-layer cards.
                    from perceptron_viz import render_all
                    m["morph"] = render_all(ck, viz_data(), run_out, tag)
                    jpath.write_text(json.dumps(m))
                if tag == "best" and args.legacy_unroll:
                    subprocess.run(
                        [sys.executable, str(PROJECT_ROOT / "analysis/activations.py"), "--checkpoint", str(ck),
                         "--out-dir", str(run_out), "--include-ood"],
                        check=False, stdout=subprocess.DEVNULL,
                    )
            run_ms.append(m)
            if tag == "best":
                best_metrics.append(m)
            for f in m["flags"]:
                print(f"    ! {f}")
        plot_evolution(run_ms, run_out / "evolution.png")

    if best_metrics:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_summary(best_metrics, statuses, args.out_dir / "SUMMARY.md")
        plot_summary(best_metrics, args.out_dir / "summary.png")
        print(f"\nWrote {args.out_dir / 'SUMMARY.md'} and {args.out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
