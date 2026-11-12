"""Unroll ReflectivityNet stage by stage and visualize its internal activations."""

import argparse
from pathlib import Path

import matplotlib

# Save figures directly to PNG files without opening an interactive window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


from refltron.common.loss import LABEL_NAMES, LOSS_CHOICES, DepthProfileLoss, build_loss
from refltron.common.paths import PROJECT_ROOT
from refltron.multi_channel_mlp.metrics import build_ood_test_cases, pick_sample_near_thickness
from refltron.multi_channel_mlp.network import REGULARIZER_CHOICES, ReflectivityNet
from refltron.multi_channel_mlp.train import load_dataset

from refltron.multi_channel_mlp.analysis.diagnostics import ACTIVATIONS, VAL_SHARD, load_model

# Human-readable labels for each regularizer mode in ReflectivityNet.
REGULARIZER_LABELS = {
    "log": (
        "Input ch0: log10(s)",
        "Input ch1: log10(p)",
        r"$\log_{10}(s)$",
        r"$\log_{10}(p)$",
        r"$\log_{10}(R)$",
        "Stage 1a: Log Transform vs Array Index",
    ),
    "rq4": (
        "Input ch0: s * q^4",
        "Input ch1: p * q^4",
        r"$s \cdot q^4$",
        r"$p \cdot q^4$",
        r"$R \cdot q^4$",
        "Stage 1a: Fresnel R·q^4 vs Array Index",
    ),
    "log_rq4": (
        "Input ch0: log10(s*q^4)",
        "Input ch1: log10(p*q^4)",
        r"$\log_{10}(s \cdot q^4)$",
        r"$\log_{10}(p \cdot q^4)$",
        r"$\log_{10}(R \cdot q^4)$",
        "Stage 1a: Fresnel Log Transform vs Array Index",
    ),
}


def main():
    args = parse_args()

    # 1. Load the trained ReflectivityNet weights and matching loss function.
    ckpt_path = find_checkpoint(args.checkpoint)
    model, criterion, loss_name, ckpt = load_trained_model(
        ckpt_path, args.regularizer, args.loss
    )
    print(
        f"Loaded checkpoint: {ckpt_path} "
        f"(epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.2e}, "
        f"regularizer={model.regularizer!r}, loss={loss_name!r})"
    )

    # 2. Load example training rows (and optional OOD benchmark cases), sorted thin -> thick.
    row_tags, s, p, q, targets, highlight_pos = load_example_rows(
        args.data, args.num_rows, args.rows, args.include_ood
    )

    # 3. Unroll the network: run every single operation step by step and save each stage.
    stages = unroll_network(model, s, p, q)

    # 4. Print a plain-English summary table and save the two stage-by-stage figures.
    print_stage_summary(model, stages, criterion, loss_name, targets, row_tags)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig1_path = args.out_dir / "01_curves_and_mixer.png"
    fig2_path = args.out_dir / "02_flatten_and_mlp.png"

    plot_curves_and_mixer(
        fig1_path, model, stages, row_tags, targets, highlight_pos, ckpt_path
    )
    plot_flatten_and_mlp(
        fig2_path, model, stages, criterion, loss_name, row_tags, targets, ckpt_path
    )
    print(f"\nSaved stage visualizations:\n  1. {fig1_path}\n  2. {fig2_path}")


@torch.no_grad()
def unroll_network(model, s, p, q):
    """Break ReflectivityNet.forward() apart into every individual tensor stage."""
    model.eval()
    stages = {}

    # Stage 0: Raw inputs straight from the dataset, each of shape (batch, 256).
    stages["raw_s"] = s
    stages["raw_p"] = p
    stages["raw_q"] = q

    # Stage 1: Apply model.regularizer ("log", "rq4", or "log_rq4") and stack (S, P, q)
    # into 3 channels of shape (batch, 3, 256).
    stacked_input = model.prepare_inputs(s, p, q)
    stages["stacked_input"] = stacked_input

    # Stage 2: Pointwise 1x1 convolution mixes the 3 channels into mix_channels at each
    # index 0..255 independently, followed by optional ReLU before Flatten.
    conv_layer = model.channel_mixer[0]
    mixer_pre_relu = conv_layer(stacked_input)
    mixer_has_relu = isinstance(model.channel_mixer[1], ACTIVATIONS)
    if mixer_has_relu:
        mixer_post_relu = model.channel_mixer[1](mixer_pre_relu)
        flatten_layer = model.channel_mixer[2]
    else:
        mixer_post_relu = mixer_pre_relu
        flatten_layer = model.channel_mixer[1]

    stages["mixer_has_relu"] = mixer_has_relu
    stages["mixer_pre_relu"] = mixer_pre_relu
    stages["mixer_post_relu"] = mixer_post_relu

    # Stage 3: Flatten concatenates all mixer channels end-to-end (e.g. 8 * 256 = 2048).
    flattened = flatten_layer(mixer_post_relu)
    stages["flattened"] = flattened

    # Stages 4..N: Walk through the hidden MLP layers before the final Linear(100 -> 7).
    # Handles both (Linear + ReLU) pairs and bare Linear layers (when ReLU is commented out).
    hidden = flattened
    mlp_layers = []
    hidden_modules = list(model.mlp[:-1])
    mod_idx = 0
    layer_num = 1
    while mod_idx < len(hidden_modules):
        module = hidden_modules[mod_idx]
        if not isinstance(module, nn.Linear):
            hidden = module(hidden)
            mod_idx += 1
            continue

        pre_relu = module(hidden)
        has_relu = (
            mod_idx + 1 < len(hidden_modules)
            and isinstance(hidden_modules[mod_idx + 1], ACTIVATIONS)
        )
        if has_relu:
            post_relu = hidden_modules[mod_idx + 1](pre_relu)
            mod_idx += 2
        else:
            post_relu = pre_relu
            mod_idx += 1

        mlp_layers.append(
            {
                "name": f"MLP Layer {layer_num} ({module.in_features} -> {module.out_features})",
                "has_relu": has_relu,
                "pre_relu": pre_relu,
                "post_relu": post_relu,
            }
        )
        hidden = post_relu
        layer_num += 1

    stages["mlp_layers"] = mlp_layers

    # Final Stage: Linear(100 -> 7) output layer (no ReLU so [0, 1] & OOD values aren't clipped).
    output_layer = model.mlp[-1]
    output_norm = output_layer(hidden)
    stages["output_norm"] = output_norm

    # Verify our unrolled calculation matches model(s, p, q) to floating-point precision.
    direct_output = model(s, p, q)
    assert torch.allclose(output_norm, direct_output, atol=1e-5)

    return stages


def format_range(low, high):
    """Format [min, max] cleanly whether values are O(1) or tiny O(1e-8)."""
    scale = max(abs(low), abs(high))
    if 0.0 < scale < 1e-2:
        return f"[{low:+9.2e}, {high:+9.2e}]"
    return f"[{low:+9.2f}, {high:+9.2f}]"


def print_stage_summary(model, stages, criterion, loss_name, targets, row_tags):
    """Print tensor shapes, value ranges, ReLU survival rates, and decoded predictions."""
    s_name = REGULARIZER_LABELS[model.regularizer][0]
    p_name = REGULARIZER_LABELS[model.regularizer][1]
    print(f"\nInspected rows (sorted thin -> thick): {row_tags}")
    header = (
        f"{'Stage':<34} | {'Shape':<14} | {'Pre-ReLU [min, max]':<23} | "
        f"{'Post-ReLU [min, max]':<23} | {'Active > 0':<10} | {'Always Dead'}"
    )
    print("\n" + header)
    print("-" * len(header))

    x = stages["stacked_input"]
    channel_names = (s_name, p_name, "Input ch2: q")
    for ch_idx in range(len(channel_names)):
        ch_name = channel_names[ch_idx]
        vals = x[:, ch_idx, :]
        rng = format_range(vals.min().item(), vals.max().item())
        print(f"{ch_name:<34} | {str(tuple(vals.shape)):<14} | {rng:<23} | {'(no ReLU)':<23} | {'--':<10} | --")

    pre = stages["mixer_pre_relu"]
    post = stages["mixer_post_relu"]
    mixer_has_relu = stages["mixer_has_relu"]
    if mixer_has_relu:
        mixer_suffix = "Conv1d + ReLU"
    else:
        mixer_suffix = "Conv1d, no ReLU"
    for ch in range(pre.size(1)):
        ch_pre = pre[:, ch, :]
        ch_post = post[:, ch, :]
        pre_rng = format_range(ch_pre.min().item(), ch_pre.max().item())
        if mixer_has_relu:
            active_pct = 100.0 * (ch_post > 0).float().mean().item()
            # A position is dead when it is exactly 0 for every row.
            always_zero = ch_post.abs().max(dim=0).values == 0
            dead_pct = 100.0 * always_zero.float().mean().item()
            post_rng = format_range(ch_post.min().item(), ch_post.max().item())
            print(
                f"{f'Mixer ch{ch} ({mixer_suffix})':<34} | {str(tuple(ch_post.shape)):<14} | "
                f"{pre_rng:<23} | {post_rng:<23} | {active_pct:6.1f}%    | {dead_pct:5.1f}%"
            )
        else:
            print(
                f"{f'Mixer ch{ch} ({mixer_suffix})':<34} | {str(tuple(ch_post.shape)):<14} | "
                f"{pre_rng:<23} | {'(no ReLU)':<23} | {'--':<10} | --"
            )

    flat = stages["flattened"]
    flat_rng = format_range(flat.min().item(), flat.max().item())
    if mixer_has_relu:
        flat_active = 100.0 * (flat > 0).float().mean().item()
        always_zero = flat.abs().max(dim=0).values == 0
        flat_dead = 100.0 * always_zero.float().mean().item()
        print(
            f"{'Flatten (8 x 256 -> 2048)':<34} | {str(tuple(flat.shape)):<14} | "
            f"{'(from mixer)':<23} | {flat_rng:<23} | {flat_active:6.1f}%    | {flat_dead:5.1f}%"
        )
    else:
        print(
            f"{'Flatten (8 x 256 -> 2048)':<34} | {str(tuple(flat.shape)):<14} | "
            f"{flat_rng:<23} | {'(no ReLU)':<23} | {'--':<10} | --"
        )

    for layer in stages["mlp_layers"]:
        l_pre = layer["pre_relu"]
        l_post = layer["post_relu"]
        pre_rng = format_range(l_pre.min().item(), l_pre.max().item())
        if layer["has_relu"]:
            active_pct = 100.0 * (l_post > 0).float().mean().item()
            # A neuron is dead when it is never above 0 for any row.
            never_positive = l_post.max(dim=0).values <= 0
            dead_pct = 100.0 * never_positive.float().mean().item()
            post_rng = format_range(l_post.min().item(), l_post.max().item())
            print(
                f"{layer['name']:<34} | {str(tuple(l_post.shape)):<14} | "
                f"{pre_rng:<23} | {post_rng:<23} | {active_pct:6.1f}%    | {dead_pct:5.1f}%"
            )
        else:
            print(
                f"{layer['name'] + ' [no ReLU]':<34} | {str(tuple(l_post.shape)):<14} | "
                f"{pre_rng:<23} | {'(no ReLU)':<23} | {'--':<10} | --"
            )

    out = stages["output_norm"]
    out_rng = format_range(out.min().item(), out.max().item())
    print(
        f"{'Output Linear (100 -> 7)':<34} | {str(tuple(out.shape)):<14} | "
        f"{out_rng:<23} | {'(no ReLU)':<23} | {'--':<10} | --"
    )

    pred_phys = criterion.denormalize(out)
    # Always report the depth-profile distance, whatever loss the run trained with.
    if type(criterion) is DepthProfileLoss:
        profile_eval = criterion
    else:
        profile_eval = DepthProfileLoss()
    profile_l2 = profile_eval.per_sample_loss(out, targets, stages["raw_q"])

    print(f"\nThickness & depth-profile L2 check (decoded via loss={loss_name!r}):")
    for i in range(len(row_tags)):
        tag = row_tags[i]
        true_t = targets[i, 0].item()
        pred_t = pred_phys[i, 0].item()
        prof_err = profile_l2[i].item()
        print(
            f"  {tag:<12}: true = {true_t:6.1f} Å  |  network = {pred_t:6.1f} Å  "
            f"(error {pred_t - true_t:+7.1f} Å)  |  Profile L2 = {prof_err:.3e}"
        )


def plot_curves_and_mixer(path, model, stages, row_tags, targets, highlight_pos, ckpt_path):
    """Figure 1: Visualize curve stages (Raw R -> Regularizer -> 8 Mixer Channels)."""
    labels = REGULARIZER_LABELS[model.regularizer]
    s_tex = labels[2]
    p_tex = labels[3]
    y_tex = labels[4]
    stage1a_title = labels[5]

    fig = plt.figure(figsize=(18.0, 12.5), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0])

    top_grid = grid[0, :].subgridspec(1, 3)
    ax_raw = fig.add_subplot(top_grid[0, 0])
    ax_rq4 = fig.add_subplot(top_grid[0, 1])
    ax_stack = fig.add_subplot(top_grid[0, 2])

    raw_s = stages["raw_s"].numpy()
    raw_p = stages["raw_p"].numpy()
    raw_q = stages["raw_q"].numpy()
    stacked = stages["stacked_input"].numpy()
    idx_axis = np.arange(raw_q.shape[1])

    hero_tag = row_tags[highlight_pos]
    hero_t = targets[highlight_pos, 0].item()

    # Panel 1A: Stage 0 — Raw polarized reflectivity curves vs physical q.
    for i in range(len(row_tags)):
        if i == highlight_pos:
            continue
        ax_raw.plot(raw_q[i], raw_s[i], color="tab:blue", alpha=0.18, linewidth=0.9)
        ax_raw.plot(raw_q[i], raw_p[i], color="tab:red", alpha=0.18, linewidth=0.9)
    ax_raw.plot(raw_q[highlight_pos], raw_s[highlight_pos], color="tab:blue", lw=2.0, label=f"s ({hero_tag}, {hero_t:.0f} Å)")
    ax_raw.plot(raw_q[highlight_pos], raw_p[highlight_pos], color="tab:red", lw=2.0, label=f"p ({hero_tag}, {hero_t:.0f} Å)")
    ax_raw.set_yscale("log")
    ax_raw.set_xlabel(r"physical $q$ [$\mathrm{\AA}^{-1}$]")
    ax_raw.set_ylabel("reflectivity")
    ax_raw.set_title(f"Stage 0: Raw Inputs (s, p, q) — {len(row_tags)} Inspected Rows")
    ax_raw.grid(True, which="both", alpha=0.3)
    ax_raw.legend(loc="upper right", fontsize=8.5)

    # Panel 1B: Stage 1a — Regularized (S, P) plotted against array index 0..255.
    for i in range(len(row_tags)):
        if i == highlight_pos:
            continue
        ax_rq4.plot(idx_axis, stacked[i, 0], color="tab:blue", alpha=0.18, linewidth=0.9)
        ax_rq4.plot(idx_axis, stacked[i, 1], color="tab:red", alpha=0.18, linewidth=0.9)
    ax_rq4.plot(idx_axis, stacked[highlight_pos, 0], color="tab:blue", lw=2.0, label=f"S = {s_tex}")
    ax_rq4.plot(idx_axis, stacked[highlight_pos, 1], color="tab:red", lw=2.0, label=f"P = {p_tex}")
    ax_rq4.set_xlabel("array index i (0 .. 255) seen by the network")
    ax_rq4.set_ylabel(y_tex)
    ax_rq4.set_title(stage1a_title)
    ax_rq4.grid(True, alpha=0.3)
    ax_rq4.legend(loc="upper right", fontsize=8.5)

    # Panel 1C: Stage 1b — All 3 stacked input channels on the same y-axis.
    channel_colors = ("tab:blue", "tab:red", "tab:green")
    channel_labels = (f"ch0 (S): {s_tex}", f"ch1 (P): {p_tex}", r"ch2: $q$ [$\mathrm{\AA}^{-1}$]")
    ax_stack.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    for ch_idx in range(3):
        color = channel_colors[ch_idx]
        label = channel_labels[ch_idx]
        ch_min = stacked[:, ch_idx, :].min(axis=0)
        ch_max = stacked[:, ch_idx, :].max(axis=0)
        ax_stack.fill_between(idx_axis, ch_min, ch_max, color=color, alpha=0.16)
        ax_stack.plot(idx_axis, stacked[highlight_pos, ch_idx], color=color, lw=2.0, label=label)
    ax_stack.set_xlabel("array index i (0 .. 255)")
    ax_stack.set_ylabel("channel value entering Conv1d")
    ax_stack.set_title("Stage 1b: Stacked 3-Channel Tensor (S, P, q)")
    ax_stack.grid(True, alpha=0.3)
    ax_stack.legend(loc="center right", fontsize=8.5)

    # Panels 2 & 3 (2x4 grid): Stage 2 — Unrolled Conv1d(3 -> 8, kernel_size=1) (+ optional ReLU).
    weights = model.channel_mixer[0].weight.squeeze(-1).detach().cpu().numpy()  # (8, 3)
    biases = model.channel_mixer[0].bias.detach().cpu().numpy()                 # (8,)
    mixer_pre = stages["mixer_pre_relu"].numpy()
    mixer_post = stages["mixer_post_relu"].numpy()
    mixer_has_relu = stages["mixer_has_relu"]

    for ch in range(weights.shape[0]):
        ax = fig.add_subplot(grid[1 + ch // 4, ch % 4])
        active_pct = 100.0 * float((mixer_post[:, ch, :] > 0).mean())
        is_dead = mixer_has_relu and not bool((mixer_post[:, ch, :] != 0).any())

        if is_dead:
            ax.set_facecolor("#fff4f4")

        for i in range(len(row_tags)):
            if i == highlight_pos:
                continue
            if mixer_has_relu:
                ax.plot(idx_axis, mixer_pre[i, ch], color="0.65", linestyle="--", lw=0.8, alpha=0.45)
            ax.plot(idx_axis, mixer_post[i, ch], color="tab:blue", linestyle="-", lw=0.8, alpha=0.25)

        if mixer_has_relu:
            ax.plot(idx_axis, mixer_pre[highlight_pos, ch], color="0.30", linestyle="--", lw=1.4, label="Pre-ReLU (Conv1d)")
            ax.plot(idx_axis, mixer_post[highlight_pos, ch], color="tab:blue", linestyle="-", lw=2.1, label="Post-ReLU")
        else:
            ax.plot(idx_axis, mixer_post[highlight_pos, ch], color="tab:blue", linestyle="-", lw=2.1, label="Conv1d output (no ReLU)")
        ax.axhline(0.0, color="crimson", linestyle=":", linewidth=1.1)

        # Pad the bottom of each subplot slightly so the learned formula box stays clear of the curves.
        y_lo, y_hi = ax.get_ylim()
        if y_hi > y_lo:
            span = y_hi - y_lo
        else:
            span = 1.0
        ax.set_ylim(y_lo - 0.16 * span, y_hi + 0.06 * span)

        # This channel's learned weights for S, P and q, and its bias.
        w_s = weights[ch][0]
        w_p = weights[ch][1]
        w_q = weights[ch][2]
        b_c = biases[ch]
        formula = f"z = {w_s:+.2f}·S {w_p:+.2f}·P {w_q:+.2f}·q {b_c:+.2f}"
        ax.text(
            0.03, 0.04, formula,
            transform=ax.transAxes, fontsize=7.8, family="monospace",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "0.8"},
        )

        if not mixer_has_relu:
            status = "linear (no ReLU)"
        elif is_dead:
            status = "DEAD (0% active)"
        else:
            status = f"{active_pct:.0f}% active"
        if is_dead:
            title_color = "#a61c1c"
        else:
            title_color = "0.1"
        ax.set_title(f"Stage 2 — Mixer Ch {ch}: {status}", color=title_color, fontsize=10.5)
        ax.set_xlabel("array index i (0 .. 255)")
        if ch % 4 == 0:
            ax.set_ylabel("activation")
        ax.grid(True, alpha=0.3)
        if ch == 0 or ch == 2:
            ax.legend(loc="upper right", fontsize=7.8)

    fig.suptitle(
        f"Unrolling ReflectivityNet (Part 1: Curve Stages & Channel Mixer, regularizer={model.regularizer!r})  —  "
        f"{ckpt_path.parent.name}/{ckpt_path.name}",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_flatten_and_mlp(path, model, stages, criterion, loss_name, row_tags, targets, ckpt_path):
    """Figure 2: Visualize Flatten (2048), all hidden MLP layers, and the 7-label output."""
    n_mlp = len(stages["mlp_layers"])
    total_rows = 1 + n_mlp + 1  # Flatten + hidden layers + Output layer
    height_ratios = [1.10] + [1.0] * n_mlp + [1.35]

    fig, axes = plt.subplots(
        total_rows, 2,
        figsize=(17.5, 20.0),
        gridspec_kw={"width_ratios": [2.6, 1.15], "height_ratios": height_ratios},
        layout="constrained",
    )

    row_labels = []
    for i in range(len(row_tags)):
        row_labels.append(f"{row_tags[i]} ({targets[i, 0].item():.0f} Å)")
    y_ticks = np.arange(len(row_tags))

    # Shared colormap for ReLU activations: exact zeros show up as clean light gray.
    relu_cmap = plt.get_cmap("viridis").copy()
    relu_cmap.set_bad("#f1f3f5")

    # --- Row 0: Stage 3 — Flatten (8 x 256 -> 2048) ---
    ax_flat = axes[0, 0]
    ax_flat_bar = axes[0, 1]
    flat_np = stages["flattened"].numpy()
    mixer_has_relu = stages["mixer_has_relu"]

    if mixer_has_relu:
        flat_masked = np.ma.masked_where(flat_np <= 0.0, flat_np)
        if np.any(flat_np > 0):
            flat_vmax = float(np.percentile(flat_np[flat_np > 0], 99))
        else:
            flat_vmax = 1.0
        im_flat = ax_flat.imshow(flat_masked, aspect="auto", interpolation="nearest", cmap=relu_cmap, vmin=0.0, vmax=flat_vmax)
        fig.colorbar(im_flat, ax=ax_flat, pad=0.012, fraction=0.025, label="post-ReLU")
        flat_note = "[gray = 0 after ReLU]"
    else:
        lim = max(float(np.percentile(np.abs(flat_np), 99)), 1e-6)
        im_flat = ax_flat.imshow(flat_np, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im_flat, ax=ax_flat, pad=0.012, fraction=0.025, label="linear value")
        flat_note = "[linear, no ReLU]"

    n_mix = stages["mixer_post_relu"].size(1)
    n_q = stages["mixer_post_relu"].size(2)
    for ch in range(1, n_mix):
        ax_flat.axvline(ch * n_q - 0.5, color="0.35", linestyle="--", linewidth=0.9)

    # Label each mixer channel's block of the flattened vector at its centre.
    ch_centers = []
    ch_tags = []
    for ch in range(n_mix):
        ch_centers.append(ch * n_q + n_q // 2)
        block = flat_np[:, ch * n_q : (ch + 1) * n_q]
        if mixer_has_relu and not np.any(block != 0):
            ch_tags.append(f"Ch {ch} (dead)")
        else:
            ch_tags.append(f"Ch {ch}")
    top_ax = ax_flat.secondary_xaxis("top")
    top_ax.set_xticks(ch_centers, ch_tags, fontsize=8.0)
    top_ax.tick_params(length=0)
    tick_labels = top_ax.get_xticklabels()
    for ch in range(len(tick_labels)):
        if "dead" in ch_tags[ch]:
            tick_labels[ch].set_color("#a61c1c")

    ax_flat.set_yticks(y_ticks, row_labels, fontsize=7.8)
    ax_flat.set_xlabel(f"flattened feature index (0 .. {flat_np.shape[1] - 1})")
    ax_flat.set_title(f"Stage 3: Flatten ({n_mix} × {n_q} → {flat_np.shape[1]})  {flat_note}")

    mixer_post = stages["mixer_post_relu"].numpy()
    ch_active_pct = []
    bar_colors = []
    bar_names = []
    for ch in range(n_mix):
        pct = 100.0 * float((mixer_post[:, ch, :] > 0).mean())
        ch_active_pct.append(pct)
        if mixer_has_relu and pct == 0.0:
            bar_colors.append("#d9534f")  # dead channel
        else:
            bar_colors.append("tab:blue")
        bar_names.append(f"Ch {ch}")
    ax_flat_bar.bar(np.arange(n_mix), ch_active_pct, color=bar_colors, width=0.65)
    ax_flat_bar.set_xticks(np.arange(n_mix), bar_names, fontsize=8.5)
    ax_flat_bar.set_ylim(0, 112)
    ax_flat_bar.set_ylabel("% > 0")
    if mixer_has_relu:
        ax_flat_bar.set_title("Surviving Fraction per Mixer Channel")
    else:
        ax_flat_bar.set_title("Positive Fraction per Mixer Channel (no ReLU)")
    ax_flat_bar.grid(True, axis="y", alpha=0.3)

    # --- Rows 1..n_mlp: Hidden MLP Layers ---
    for idx in range(len(stages["mlp_layers"])):
        layer = stages["mlp_layers"][idx]
        ax_hm = axes[1 + idx, 0]
        ax_hist = axes[1 + idx, 1]
        pre_np = layer["pre_relu"].numpy()
        post_np = layer["post_relu"].numpy()
        has_relu = layer["has_relu"]

        if has_relu:
            masked_post = np.ma.masked_where(post_np <= 0.0, post_np)
            if np.any(post_np > 0):
                vmax = float(np.percentile(post_np[post_np > 0], 99))
            else:
                vmax = 1.0
            im = ax_hm.imshow(masked_post, aspect="auto", interpolation="nearest", cmap=relu_cmap, vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax_hm, pad=0.012, fraction=0.025, label="post-ReLU")
            active_pct = 100.0 * float((post_np > 0).mean())
            never_positive = post_np.max(axis=0) <= 0  # neurons that are 0 for every row
            dead_pct = 100.0 * float(never_positive.mean())
            hm_title = f"Stage {4 + idx}: {layer['name']} + ReLU  —  {active_pct:.1f}% active, {dead_pct:.1f}% dead across rows"
        else:
            lim = max(float(np.percentile(np.abs(post_np), 99)), 1e-6)
            im = ax_hm.imshow(post_np, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-lim, vmax=lim)
            fig.colorbar(im, ax=ax_hm, pad=0.012, fraction=0.025, label="linear")
            hm_title = f"Stage {4 + idx}: {layer['name']} (linear — ReLU disabled)"

        ax_hm.set_yticks(y_ticks, row_labels, fontsize=7.8)
        ax_hm.set_xlabel(f"neuron index (0 .. {post_np.shape[1] - 1})")
        ax_hm.set_title(hm_title)

        # Clip extreme outliers at the 2nd/98th percentile so the 0 threshold is easy to see.
        flat_pre = pre_np.ravel()
        lo_clip = float(np.percentile(flat_pre, 2.0))
        if has_relu:
            hi_clip = float(flat_pre.max())
        else:
            hi_clip = float(np.percentile(flat_pre, 98.0))
        clipped_pre = np.clip(flat_pre, lo_clip, max(hi_clip, lo_clip + 1e-6))

        neg_vals = clipped_pre[clipped_pre <= 0.0]
        pos_vals = clipped_pre[clipped_pre > 0.0]
        bins = np.linspace(min(lo_clip, -0.1), max(hi_clip, 0.5), 36)
        if has_relu:
            neg_label = "≤ 0 (zeroed by ReLU)"
            pos_label = "> 0 (kept by ReLU)"
        else:
            neg_label = "≤ 0 (kept — no ReLU)"
            pos_label = "> 0 (kept — no ReLU)"
        ax_hist.hist(neg_vals, bins=bins, color="#e07a7a", alpha=0.85, label=neg_label)
        ax_hist.hist(pos_vals, bins=bins, color="tab:blue", alpha=0.85, label=pos_label)
        ax_hist.axvline(0.0, color="0.15", linestyle="--", linewidth=1.1)
        ax_hist.set_xlabel("Linear output")
        ax_hist.set_ylabel("count")
        ax_hist.set_title(f"Linear range: [{flat_pre.min():.1f}, {flat_pre.max():.1f}]")
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper left", fontsize=7.5)

    # --- Final Row: Output Layer Linear(100 -> 7) vs Ground Truth ---
    ax_out = axes[-1, 0]
    ax_thick = axes[-1, 1]
    pred_norm = stages["output_norm"].numpy()
    true_norm = criterion.normalize(targets).numpy()
    err_norm = pred_norm - true_norm
    pred_phys = criterion.denormalize(stages["output_norm"]).numpy()
    true_phys = targets.numpy()

    im_out = ax_out.imshow(err_norm, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    fig.colorbar(im_out, ax=ax_out, pad=0.012, fraction=0.025, label="pred − truth")
    for r in range(pred_norm.shape[0]):
        for c in range(pred_norm.shape[1]):
            if abs(err_norm[r, c]) > 0.22:
                text_color = "white"  # dark cell: light text
            else:
                text_color = "0.12"
            ax_out.text(
                c, r, f"{pred_norm[r, c]:.2f} ({true_norm[r, c]:.2f})",
                ha="center", va="center", fontsize=7.6, color=text_color,
            )
    ax_out.set_xticks(np.arange(len(LABEL_NAMES)), LABEL_NAMES, rotation=14, ha="right")
    ax_out.set_yticks(y_ticks, row_labels, fontsize=7.8)
    ax_out.set_title(
        f"Stage {4 + n_mlp}: Output Linear(100 → 7)  [cell text: network (truth) in [0, 1]; color = signed error]"
    )

    # Companion panel: True vs Predicted film_thickness [Å] across inspected rows.
    bar_h = 0.36
    ax_thick.barh(y_ticks - bar_h / 2, true_phys[:, 0], height=bar_h, color="0.60", label="truth")
    ax_thick.barh(y_ticks + bar_h / 2, pred_phys[:, 0], height=bar_h, color="tab:blue", label="network")
    ax_thick.invert_yaxis()
    ax_thick.set_yticks(y_ticks, row_tags, fontsize=8.0)
    ax_thick.set_xlabel("film_thickness [Å]")
    ax_thick.set_title(f"Decoded Film Thickness (loss={loss_name!r})")
    ax_thick.grid(True, axis="x", alpha=0.3)
    ax_thick.legend(loc="lower right", fontsize=8.0)

    fig.suptitle(
        f"Unrolling ReflectivityNet (Part 2: Flatten, MLP & Output, regularizer={model.regularizer!r}, loss={loss_name!r})  —  "
        f"{ckpt_path.parent.name}/{ckpt_path.name}",
        fontsize=13.5,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def find_checkpoint(requested_path):
    """Return the requested checkpoint or the most recently saved best.pt."""
    if requested_path is not None:
        return Path(requested_path)
    newest = None
    for path in (PROJECT_ROOT / "checkpoints").glob("*/best.pt"):
        if newest is None or path.stat().st_mtime >= newest.stat().st_mtime:
            newest = path
    if newest is None:
        raise FileNotFoundError("No checkpoints/*/best.pt found. Pass --checkpoint <path>.")
    return newest


def load_trained_model(ckpt_path, regularizer_override=None, loss_override=None):
    """Build the exact saved architecture (ReLU or GELU) + matching loss and load weights."""
    model, criterion, meta = load_model(ckpt_path)
    if regularizer_override:
        model.regularizer = regularizer_override
    if loss_override:
        loss_name = loss_override
        criterion = build_loss(loss_override)
    else:
        loss_name = meta["loss"]
    ckpt = {"epoch": meta["epoch"], "val_loss": meta["val_loss"]}
    return model, criterion, loss_name, ckpt


def load_example_rows(data_path, num_rows, explicit_rows, include_ood=False):
    """Load example training rows (plus optional OOD cases) sorted thin -> thick."""
    if explicit_rows:
        raw_indices = list(explicit_rows)
    else:
        raw_indices = list(range(num_rows))
    max_needed = max(raw_indices) + 1
    if data_path is None:
        data_path = str(VAL_SHARD)  # default: the held-out validation shard
    dataset = load_dataset(data_path, max_samples=max_needed)

    # Highlight the in-distribution training row closest to ~200 Å in the curve plots.
    hero_dataset_idx = pick_sample_near_thickness(dataset, raw_indices, target_thickness=200.0)

    # One entry per row in each list: name, highlighted?, s, p, q, labels.
    tags = []
    is_hero = []
    s_rows = []
    p_rows = []
    q_rows = []
    target_rows = []
    for i in raw_indices:
        row = dataset[i]
        tags.append(f"Row #{i}")
        is_hero.append(i == hero_dataset_idx)
        s_rows.append(row[0])
        p_rows.append(row[1])
        q_rows.append(row[2])
        target_rows.append(row[3])

    if include_ood:
        ood_cases = build_ood_test_cases()
        ood_tags = ("OOD 0Å", "OOD 2000Å")
        for k in range(2):
            tags.append(ood_tags[k])
            is_hero.append(False)
            s_rows.append(ood_cases[k]["s"])
            p_rows.append(ood_cases[k]["p"])
            q_rows.append(ood_cases[k]["q"])
            target_rows.append(ood_cases[k]["target"])

    # Sort all rows from thinnest to thickest true film_thickness (ties keep their order).
    thicknesses = []
    for target in target_rows:
        thicknesses.append(target[0].item())
    order = np.argsort(thicknesses, kind="stable")

    row_tags = []
    highlight_pos = None
    for position in range(len(order)):
        k = order[position]
        row_tags.append(tags[k])
        if is_hero[k] and highlight_pos is None:
            highlight_pos = position
    sorted_s = []
    sorted_p = []
    sorted_q = []
    sorted_targets = []
    for k in order:
        sorted_s.append(s_rows[k])
        sorted_p.append(p_rows[k])
        sorted_q.append(q_rows[k])
        sorted_targets.append(target_rows[k])
    s = torch.stack(sorted_s)
    p = torch.stack(sorted_p)
    q = torch.stack(sorted_q)
    targets = torch.stack(sorted_targets)
    return row_tags, s, p, q, targets, highlight_pos


def parse_args():
    """Parse optional command-line settings."""
    parser = argparse.ArgumentParser(description="Unroll ReflectivityNet and plot layer activations.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt checkpoint (default: newest best.pt)")
    parser.add_argument("--regularizer", choices=REGULARIZER_CHOICES, default=None, help="Override checkpoint regularizer")
    parser.add_argument("--loss", choices=LOSS_CHOICES, default=None, help="Override checkpoint loss function")
    parser.add_argument("--data", default=None, help="Parquet path or glob")
    parser.add_argument("--num-rows", type=int, default=10, help="Number of rows to inspect (default: first 10 held-out validation rows)")
    parser.add_argument("--rows", type=int, nargs="+", default=None, help="Specific row indices to inspect instead of 0..N-1")
    parser.add_argument("--include-ood", action="store_true", help="Also unroll the 0 Å bare-oxide and 2000 Å thick-film OOD benchmarks")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis" / "runs" / "legacy_activations", help="Directory for output PNG plots")
    return parser.parse_args()


if __name__ == "__main__":
    main()
