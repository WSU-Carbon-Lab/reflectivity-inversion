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
    uv run python -m refltron.multi_channel_mlp.analysis.diagnostics --all              # every run in checkpoints/
    uv run python -m refltron.multi_channel_mlp.analysis.diagnostics --checkpoint checkpoints/<run>/best.pt
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


from refltron.common.loss import LABEL_NAMES, DepthProfileLoss, build_loss
from refltron.common.paths import PROJECT_ROOT
from refltron.multi_channel_mlp.network import ReflectivityNet

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
    cfg = ckpt.get("config")
    if not cfg:
        cfg = {}
    sd = ckpt["model_state_dict"]

    # Older checkpoints did not store the activation; "netg" was the GELU model.
    activation = cfg.get("activation")
    if not activation:
        if cfg.get("model") == "netg":
            activation = "gelu"
        else:
            activation = "relu"
    if activation == "gelu":
        act_cls = nn.GELU
    else:
        act_cls = nn.ReLU

    # Checkpoints saved before --regularizer / --loss existed used log_rq4 + label MSE.
    regularizer = cfg.get("regularizer")
    if not regularizer:
        regularizer = "log_rq4"
    loss_name = cfg.get("loss")
    if not loss_name:
        loss_name = "mse"
    normalize = "input_mean" in sd

    model = ReflectivityNet(
        mix_channels=cfg.get("mix_channels", 8), regularizer=regularizer,
        activation=activation, normalize_inputs=normalize,
    )
    conv_w = sd["channel_mixer.0.weight"]
    model.channel_mixer = nn.Sequential(
        nn.Conv1d(conv_w.size(1), conv_w.size(0), kernel_size=1), act_cls(), nn.Flatten(1)
    )

    # Rebuild the MLP layer by layer. Every saved "mlp.<i>.weight" is a Linear layer;
    # any index in between without a weight was a parameter-free activation.
    last_index = 0
    for key in sd:
        if key.startswith("mlp.") and key.endswith(".weight"):
            index = int(key.split(".")[1])
            if index > last_index:
                last_index = index
    model.mlp = nn.Sequential()
    for i in range(last_index + 1):
        w = sd.get(f"mlp.{i}.weight")
        if w is not None:
            model.mlp.append(nn.Linear(w.shape[1], w.shape[0]))
        else:
            model.mlp.append(act_cls())
    model.load_state_dict(sd)
    model.eval()

    model_name = cfg.get("model")
    if not model_name:
        model_name = "net"
    meta = {
        "run": ckpt_path.parent.name,
        "checkpoint": ckpt_path.name,
        "epoch": int(ckpt.get("epoch", -1)),
        "val_loss": float(ckpt.get("val_loss", float("nan"))),
        "planned_epochs": cfg.get("epochs"),
        "model": model_name,
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
    blocks = []
    i = 0
    while i < len(seq):
        m = seq[i]
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            # A linear op owns the activation right after it, if there is one.
            act = None
            if i + 1 < len(seq) and isinstance(seq[i + 1], ACTIVATIONS):
                act = seq[i + 1]
            blocks.append((m, act))
            if act is not None:
                i += 2
            else:
                i += 1
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
    blocks = block_layout(model)
    for idx in range(len(blocks)):
        op = blocks[idx][0]
        act = blocks[idx][1]
        if isinstance(op, nn.Linear) and x.dim() == 3:
            x = x.flatten(1)
        pre = op(x)
        if act is not None:
            post = act(pre)
        else:
            post = pre
        out.append(
            {"name": block_name(op, idx), "op": op, "act": act, "input": x, "pre": pre, "post": post}
        )
        x = post
    return stacked, out


def as_units(t):
    """View activations as (samples, units); Conv1d channels are treated per q index."""
    if t.dim() == 3:
        # (batch, channels, n_q) -> (batch * n_q, channels): every q point counts as a sample.
        return t.permute(0, 2, 1).reshape(-1, t.size(1))
    return t


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
    # Every unit pair once: the entries above the diagonal of the correlation matrix.
    iu = np.triu_indices_from(c, k=1)
    return float((np.abs(c[iu]) > 0.95).mean())


def ridge_probe(feats: np.ndarray, y: np.ndarray, splits):
    """Linear-probe R^2 per label: fit ridge on train, pick lambda on val, score on test."""
    tr = splits[0]
    va = splits[1]
    te = splits[2]
    mu = feats[tr].mean(0)
    sd = feats[tr].std(0) + 1e-6
    X = (feats - mu) / sd
    X = np.concatenate([X, np.ones((len(X), 1))], 1)  # extra column of ones = bias term
    ym = y[tr].mean(0)
    Y = y - ym

    # Solve in whichever space is smaller (features or samples).
    Xtr = X[tr]
    Ytr = Y[tr]
    use_dual = Xtr.shape[1] > Xtr.shape[0]
    if use_dual:
        G = Xtr @ Xtr.T
    else:
        G = Xtr.T @ Xtr
    scale = np.trace(G) / G.shape[0]
    best = None
    best_err = np.inf
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        reg = G + lam * scale * np.eye(G.shape[0])
        if use_dual:
            W = Xtr.T @ np.linalg.solve(reg, Ytr)
        else:
            W = np.linalg.solve(reg, Xtr.T @ Ytr)
        err = ((X[va] @ W - Y[va]) ** 2).mean()
        if err < best_err:
            best = W
            best_err = err
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
        op = blocks[i][0]
        act = blocks[i][1]
        nxt = blocks[i + 1][0]
        if act is None and isinstance(op, nn.Linear) and isinstance(nxt, nn.Linear):
            prod = (nxt.weight @ op.weight).detach().double()
            sv = torch.linalg.svdvals(prod).numpy()
            chains.append(
                {
                    "blocks": [block_name(op, i), block_name(nxt, i + 1)],
                    "composite_shape": list(prod.shape),
                    "rank_bound": int(min(op.weight.shape[0], prod.shape[0], prod.shape[1])),
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
    blocks = block_layout(model)
    for idx in range(len(blocks)):
        op = blocks[idx][0]
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
        # Gradient of output k (summed over rows) with respect to every input value.
        gradients = torch.autograd.grad(out[:, k].sum(), x, retain_graph=True)
        g = gradients[0]
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
    ws = w[:, 0]
    wp = w[:, 1]
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
        frac_active = active.mean(0).numpy()  # fraction of samples on which each unit fires
        has_act = b["act"] is not None
        post_np = post_u.numpy()

        # Unit-usage numbers only mean something for blocks followed by an activation.
        if has_act:
            live = frac_active > 0
            activation_name = type(b["act"]).__name__
            sparsity = float(1.0 - active.mean().item())
            dead_frac = float((frac_active == 0).mean())
            rare_frac = float(((frac_active > 0) & (frac_active < 0.01)).mean())
            always_on_frac = float((frac_active > 0.99).mean())
        else:
            live = np.ones_like(frac_active, bool)
            activation_name = None
            sparsity = 0.0
            dead_frac = 0.0
            rare_frac = 0.0
            always_on_frac = 1.0

        # Share of the layer's variance carried by its top 1% most variable units.
        unit_var = post_u.var(0).numpy()
        top = max(1, int(round(0.01 * len(unit_var))))
        largest_first = np.sort(unit_var)[::-1]
        var_top1 = float(largest_first[:top].sum() / max(unit_var.sum(), 1e-12))

        # Effective rank on per-sample feature vectors (flattened for the conv mixer).
        feats = b["post"].flatten(1).numpy()
        pr, n99 = effective_rank(feats)
        rows.append(
            {
                "name": b["name"],
                "width": int(feats.shape[1]),
                "has_activation": has_act,
                "activation": activation_name,
                "in_rms": in_rms,
                "pre_rms": b["pre"].pow(2).mean().sqrt().item(),
                "post_rms": post_rms,
                "gain": post_rms / (in_rms + 1e-12),
                "spike_ratio": b["post"].abs().max().item() / (post_rms + 1e-12),
                "sparsity": sparsity,
                "dead_frac": dead_frac,
                "rare_frac": rare_frac,
                "always_on_frac": always_on_frac,
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

    # Linear-probe R^2 should not drop much from one stage to the next.
    prev_name = "input"
    prev = np.mean(m["probe_input"])
    for L in m["layers"]:
        cur = np.mean(L["probe_r2"])
        if prev - cur > FLAG["probe_drop"]:
            flags.append(f"{L['name']}: linear-probe mean R^2 drops {prev:.3f} -> {cur:.3f} vs {prev_name} (information lost or entangled)")
        prev_name = L["name"]
        prev = cur

    for c in m["linear_chains"]:
        flags.append(
            f"{c['blocks'][0]} and {c['blocks'][1]} have no activation between them: they collapse to one "
            f"{c['composite_shape'][0]}x{c['composite_shape'][1]} map of rank <= {c['rank_bound']} "
            f"(numerical rank {c['numerical_rank']}); the {c['wide_layer_width']}-wide layer can only carry a "
            f"{c['rank_bound']}-dim signal and the pair is no more expressive than a single linear layer"
        )

    ratios = []
    for g in m["gradient_flow"]:
        if g["grad_to_weight"] > 0:
            ratios.append(g["grad_to_weight"])
    if len(ratios) > 0 and max(ratios) / min(ratios) > FLAG["grad_spread"]:
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
    # Split the rows 60 / 20 / 20 into probe-training, probe-validation and probe-test rows.
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

    weights = []
    layout = block_layout(model)
    for i in range(len(layout)):
        op = layout[i][0]
        entry = {"name": block_name(op, i)}
        entry.update(weight_stats(op))
        weights.append(entry)

    m = dict(meta)  # start from the checkpoint description, then add the measurements
    m["n_rows"] = n
    m["val_metrics"] = val_metrics
    m["output_r2_per_label"] = out_r2.tolist()
    m["probe_input"] = probe_in
    m["input_eff_rank"] = effective_rank(stacked.flatten(1).numpy())[0]
    m["layers"] = layers
    m["weights"] = weights
    m["linear_chains"] = chained_linear_rank(layout)
    m["gradient_flow"] = gradient_flow(model, criterion, s[:n_grad], p[:n_grad], q[:n_grad], y[:n_grad])
    m["saliency"] = input_saliency(model, s[:n_sal], p[:n_sal], q[:n_sal]).tolist()
    m["mixer"] = mixer_analysis(model, stacked)
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
    names = []
    for layer in L:
        names.append(layer["name"])
    x = np.arange(len(L))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")

    ax = axes[0, 0]
    w = 0.26
    dead = []
    sparsity = []
    always_on = []
    for layer in L:
        dead.append(100 * layer["dead_frac"])
        sparsity.append(100 * layer["sparsity"])
        if layer["has_activation"]:
            always_on.append(100 * layer["always_on_frac"])
        else:
            always_on.append(100 * np.nan)  # no bar: "always on" is meaningless without an activation
    ax.bar(x - w, dead, w, color=SERIES[1], label="dead (never fires)")
    ax.bar(x, sparsity, w, color=SERIES[0], label="mean sparsity")
    ax.bar(x + w, always_on, w, color=SERIES[3], label="always on (~linear)")
    for i in range(len(L)):
        if not L[i]["has_activation"]:
            ax.text(i, 3, "no act.", ha="center", fontsize=10, color=MUTED, rotation=90)
    ax.set_ylabel("% of units")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Unit usage")

    ax = axes[0, 1]
    gains = []
    spikes = []
    for layer in L:
        gains.append(layer["gain"])
        spikes.append(layer["spike_ratio"] / 10)
    ax.plot(x, gains, "-o", color=SERIES[0], lw=2, ms=6, label="gain (RMS out / in)")
    ax.plot(x, spikes, "-s", color=SERIES[1], lw=2, ms=6, label="spike ratio / 10")
    ax.axhline(1, color=MUTED, ls=":", lw=1)
    ax.set_yscale("log")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Signal intensification per block")

    ax = axes[0, 2]
    rank_fracs = []
    redundant = []
    for layer in L:
        rank_fracs.append(100 * layer["eff_rank_frac"])
        redundant.append(100 * layer["redundant_frac"])
    ax.plot(x, rank_fracs, "-o", color=SERIES[2], lw=2, ms=6, label="eff. rank / width")
    ax.plot(x, redundant, "-s", color=SERIES[4], lw=2, ms=6, label="near-duplicate unit pairs")
    for i in range(len(L)):
        ax.annotate(f"{L[i]['eff_rank']:.0f}", (i, 100 * L[i]["eff_rank_frac"]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=10, color=INK)
    ax.set_ylabel("%")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Rank & redundancy (numbers = participation ratio)")

    ax = axes[1, 0]
    probe_rows = [m["probe_input"]]
    for layer in L:
        probe_rows.append(layer["probe_r2"])
    probes = np.array(probe_rows)  # (stages, 7)
    xs = np.arange(len(probes))
    for k in range(len(LABEL_NAMES)):
        ax.plot(xs, probes[:, k], "-o", color=SERIES[k], lw=2, ms=4, label=LABEL_NAMES[k])
    ax.plot(xs, probes.mean(1), color=INK, lw=2.5, ls="--", label="mean")
    ax.set_xticks(xs, ["input"] + names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(min(0, probes.min()) - 0.02, 1.02)
    ax.set_ylabel("held-out R²")
    ax.legend(fontsize=9.5, frameon=False, ncol=2)
    _style(ax, "Linear probe: label information available at each stage")

    ax = axes[1, 1]
    g = m["gradient_flow"]
    ratios = []
    grad_names = []
    for row in g:
        ratios.append(row["grad_to_weight"])
        grad_names.append(row["name"])
    ax.bar(np.arange(len(g)), ratios, color=SERIES[0], width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(g)), grad_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("‖∇W‖ / ‖W‖")
    _style(ax, f"Gradient flow (one batch, loss={m['loss']})")

    ax = axes[1, 2]
    for i in range(len(m["weights"])):
        wst = m["weights"][i]
        sv = np.array(wst["singular_values"])
        ax.plot(np.linspace(0, 1, len(sv)), sv / sv[0], color=SERIES[i % len(SERIES)], lw=1.8,
                label=f"{wst['name']} (stable rank {wst['stable_rank']:.1f})")
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
    channel_colors = [SERIES[0], SERIES[1], SERIES[2]]

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
    channel_names = ["S channel", "P channel", "raw q channel"]
    for c in range(3):
        ax.plot(np.arange(n_q), tot[c], color=channel_colors[c], lw=2, label=channel_names[c])
    ax.set_xlabel("q index (0 .. 255)")
    ax.set_ylabel("Σ_labels mean |∂y/∂x|")
    ax.legend(fontsize=10.5, frameon=False)
    _style(ax, "Input saliency by channel")

    ax = axes[0, 1]
    frac = np.array(m["mixer"]["variance_frac_S_P_q"])
    mix = np.arange(frac.shape[0])
    bottom = np.zeros(len(mix))
    short_names = ["S", "P", "q"]
    for c in range(3):
        ax.bar(mix, frac[:, c], bottom=bottom, color=channel_colors[c], width=0.65, label=short_names[c],
               edgecolor="white", linewidth=1)
        bottom += frac[:, c]  # stack the next channel on top
    tick_labels = []
    for i in mix:
        alive = m["layers"][0]["unit_frac_active"][i]
        tick_labels.append(f"ch{i}\n{100*alive:.0f}% on")
    ax.set_xticks(mix, tick_labels, fontsize=10)
    ax.set_ylabel("share of pre-activation variance")
    ax.legend(fontsize=10.5, frameon=False, ncol=3)
    _style(ax, "Mixer: what each channel listens to")

    ax = axes[1, 1]
    ax.bar(mix, m["mixer"]["polarization_contrast"], color=SERIES[4], width=0.65)
    ax.set_ylim(0, 1)
    channel_ticks = []
    for i in mix:
        channel_ticks.append(f"ch{i}")
    ax.set_xticks(mix, channel_ticks, fontsize=10.5)
    ax.set_ylabel("|w_S − w_P| / (|w_S + w_P| + |w_S − w_P|)")
    _style(ax, "Mixer S/P contrast (0 = sum, 1 = difference)")

    fig.suptitle(f"Input sensitivity — {m['run']}/{m['checkpoint']} (epoch {m['epoch']})", fontsize=14.5, color=INK)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def checkpoint_epoch(m):
    """The epoch a diagnostics result belongs to (used to sort results by epoch)."""
    return m["epoch"]


def plot_evolution(ms, path):
    """Track layer health across the periodic checkpoints of one run."""
    periodic = []
    for m in ms:
        if m["checkpoint"] != "best.pt":
            periodic.append(m)
    periodic.sort(key=checkpoint_epoch)
    if len(periodic) < 2:
        return
    names = []
    for layer in periodic[0]["layers"]:
        names.append(layer["name"])
    epochs = []
    for m in periodic:
        epochs.append(m["epoch"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    metric_keys = ["dead_frac", "eff_rank_frac", "gain"]
    titles = ["Dead units (%)", "Effective rank / width (%)", "Gain (RMS out / in)"]
    scales = [100, 100, 1]
    for j in range(3):
        ax = axes[j]
        for i in range(len(names)):
            values = []
            for m in periodic:
                values.append(scales[j] * m["layers"][i][metric_keys[j]])
            ax.plot(epochs, values, "-o", color=SERIES[i % len(SERIES)], lw=2, ms=5, label=names[i])
        ax.set_xlabel("epoch")
        _style(ax, titles[j])
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=9.5, frameon=False)
    fig.suptitle(f"Layer health over training — {periodic[0]['run']}", fontsize=14.5, color=INK)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_summary(best, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    for i in range(len(best)):
        m = best[i]
        col = SERIES[i % len(SERIES)]
        lab = f"{m['run']} ({m['loss']}, {m['regularizer']})"
        stages = ["input"]
        probe_means = [np.mean(m["probe_input"])]
        dead = []
        ranks = []
        for layer in m["layers"]:
            stages.append(layer["name"].split(" ")[0])
            probe_means.append(np.mean(layer["probe_r2"]))
            dead.append(100 * layer["dead_frac"])
            ranks.append(layer["eff_rank"])
        xs = np.arange(len(stages))
        axes[0].plot(xs, probe_means, "-o", color=col, lw=2, ms=5, label=lab)
        axes[1].plot(xs[1:], dead, "-o", color=col, lw=2, ms=5, label=lab)
        axes[2].plot(xs[1:], ranks, "-o", color=col, lw=2, ms=5, label=lab)
    titles = ["Mean linear-probe R²", "Dead units (%)", "Effective rank (participation ratio)"]
    for j in range(3):
        axes[j].set_xticks(xs, stages, fontsize=10.5)
        _style(axes[j], titles[j])
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=10, frameon=False)
    fig.suptitle("All runs — best.pt", fontsize=14.5, color=INK)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Reporting / driver
# --------------------------------------------------------------------------------------


def run_status(run_dir: Path):
    """(finished?, last saved epoch, planned epochs) of a run folder."""
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
        if st[0]:
            status = "complete"
        elif st[1]:
            status = f"stopped ({st[1]}/{st[2]})"
        else:
            status = f"no periodic ckpt (planned {st[2]})"
        vm = m["val_metrics"]
        dead_values = []
        for layer in m["layers"]:
            dead_values.append(100 * layer["dead_frac"])
        dead = max(dead_values)
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
            # A star marks numbers that are meaningless because no activation follows the block.
            if l["has_activation"]:
                marker = ""
            else:
                marker = "*"
            lines.append(
                f"| {l['name']} | {l['width']} | {100*l['dead_frac']:.1f}" + marker +
                f" | {100*l['always_on_frac']:.1f}" + marker +
                f" | {100*l['sparsity']:.1f} | {l['gain']:.2f} | {l['spike_ratio']:.1f} | {l['eff_rank']:.1f} / {l['n99']} | "
                f"{100*l['redundant_frac']:.1f} | {np.mean(l['probe_r2']):.3f} / {l['probe_r2'][0]:.3f} |"
            )
        lines.append("\n(* = no activation after this block)\n")
        if m["flags"]:
            lines.append("Flags:")
        else:
            lines.append("Flags: none")
        for f in m["flags"]:
            lines.append(f"- {f}")
        lines.append("")
    path.write_text("\n".join(lines))


def checkpoints_for(run_dir: Path, every: bool):
    """best.pt of a run, plus its numbered epoch_*.pt checkpoints if ``every``."""
    out = []
    best = run_dir / "best.pt"
    if best.exists():
        out.append(best)
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
    viz_rows = None  # 8192 rows for the perceptron figures, loaded the first time they are needed

    # Which checkpoints to analyze, grouped by run folder: {run folder: [checkpoint paths]}.
    run_dirs = {}
    if args.checkpoint:
        run_dirs[args.checkpoint.parent] = [args.checkpoint]
    else:
        for d in sorted((PROJECT_ROOT / "checkpoints").iterdir()):
            if not d.is_dir() or not (d / "best.pt").exists():
                continue
            if args.runs and d.name not in args.runs:
                continue
            run_dirs[d] = checkpoints_for(d, not args.best_only)

    statuses = {}
    best_metrics = []
    for run_dir in run_dirs:
        ckpts = run_dirs[run_dir]
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
                    from refltron.multi_channel_mlp.analysis.perceptron_viz import render_all
                    if viz_rows is None:
                        viz_rows = load_val_rows(8192)
                    m["morph"] = render_all(ck, viz_rows, run_out, tag)
                    jpath.write_text(json.dumps(m))
                if tag == "best" and args.legacy_unroll:
                    subprocess.run(
                        [sys.executable, "-m", "refltron.multi_channel_mlp.analysis.activations", "--checkpoint", str(ck),
                         "--out-dir", str(run_out), "--include-ood"],
                        check=False, stdout=subprocess.DEVNULL, cwd=PROJECT_ROOT,
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
