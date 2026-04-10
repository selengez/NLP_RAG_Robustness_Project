"""
Generate figures

"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("results/summary.csv")


PROMPT_LABELS = {
    "baseline":           "Baseline",
    "verification_aware": "Verification-aware",
    "uncertainty_aware":  "Uncertainty-aware",
}
STRATEGY_LABELS = {
    "direct_false":         "Direct False",
    "subtle_contradiction": "Subtle Contradiction",
    "confident_lie":        "Confident Lie",
    "partial_truth":        "Partial Truth",
}
COLOURS = {
    "Baseline":             "#d62728",   # red
    "Verification-aware":   "#1f77b4",   # blue
    "Uncertainty-aware":    "#2ca02c",   # green
}
MARKERS = {
    "Baseline":             "o",
    "Verification-aware":   "s",
    "Uncertainty-aware":    "^",
}

df["prompt_label"]   = df["prompt_mode"].map(PROMPT_LABELS)
df["strategy_label"] = df["poison_strategy"].map(STRATEGY_LABELS)

strategies = list(STRATEGY_LABELS.values())
prompts    = list(PROMPT_LABELS.values())

# ── Figure 1: Accuracy by strategy ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
axes = axes.flatten()

for ax, strategy in zip(axes, strategies):
    sub = df[df["strategy_label"] == strategy]
    for prompt in prompts:
        pts = sub[sub["prompt_label"] == prompt].sort_values("poison_rate")
        ax.plot(
            pts["poison_rate"],
            pts["accuracy"],
            marker=MARKERS[prompt],
            color=COLOURS[prompt],
            label=prompt,
            linewidth=2,
            markersize=7,
        )
    ax.set_title(strategy, fontsize=12, fontweight="bold")
    ax.set_xlabel("Poison rate", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75])
    ax.set_ylim(-0.02, 0.80)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Accuracy by Poison Strategy and Prompt Mode", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig_accuracy.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved fig_accuracy")

# ── Figure 2: Poison acceptance rate heatmaps (advanced prompts only) ─────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
rates_ordered = [0.0, 0.25, 0.50, 0.75]

for ax, prompt in zip(axes, ["Verification-aware", "Uncertainty-aware"]):
    matrix = np.zeros((len(strategies), len(rates_ordered)))
    sub = df[df["prompt_label"] == prompt]
    for i, strat in enumerate(strategies):
        for j, rate in enumerate(rates_ordered):
            row = sub[(sub["strategy_label"] == strat) &
                      (sub["poison_rate"] == rate)]
            if not row.empty:
                matrix[i, j] = row["poison_acceptance_rate"].values[0]

    im = ax.imshow(matrix, vmin=0, vmax=0.5, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(rates_ordered)))
    ax.set_xticklabels([f"{r:.0%}" for r in rates_ordered], fontsize=10)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)
    ax.set_xlabel("Poison rate", fontsize=10)
    ax.set_title(prompt, fontsize=12, fontweight="bold")

    for i in range(len(strategies)):
        for j in range(len(rates_ordered)):
            val = matrix[i, j]
            colour = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=colour, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Poison acceptance rate")

fig.suptitle("Poison Acceptance Rate Heatmap (Advanced Prompts)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig_par_heatmap.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved fig_par_heatmap")

# ── Figure 3: Conflict detection rate – baseline vs. advanced at rate 0.75 ────
fig, ax = plt.subplots(figsize=(9, 4))

x = np.arange(len(strategies))
width = 0.25

for k, prompt in enumerate(prompts):
    sub = df[(df["prompt_label"] == prompt) & (df["poison_rate"] == 0.75)]
    vals = [
        sub[sub["strategy_label"] == s]["conflict_detection_rate"].values[0]
        for s in strategies
    ]
    bars = ax.bar(x + k * width, vals, width,
                  label=prompt, color=list(COLOURS.values())[k],
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(strategies, fontsize=10)
ax.set_ylabel("Conflict detection rate", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("Conflict Detection Rate at 75% Poisoning", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "fig_cdr.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved fig_cdr")

print(f"\nAll figures written to {OUT}/")

# ── Figure 4: FEVER vs HotpotQA accuracy comparison  ──────
HOTPOT_SUMMARY = Path("results/hotpot/summary.csv")
if HOTPOT_SUMMARY.exists():
    df_hot = pd.read_csv(HOTPOT_SUMMARY)
    df_hot["prompt_label"]   = df_hot["prompt_mode"].map(PROMPT_LABELS)
    df_hot["strategy_label"] = df_hot["poison_strategy"].map(STRATEGY_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    rates_ordered = [0.0, 0.25, 0.50, 0.75]

    for ax, prompt in zip(axes, prompts):
        sub_fever = df[df["prompt_label"] == prompt].sort_values("poison_rate")
        sub_hot   = df_hot[df_hot["prompt_label"] == prompt].sort_values("poison_rate")

        # Mean accuracy per poison rate across all strategies
        fever_mean = sub_fever.groupby("poison_rate")["accuracy"].mean()
        hot_mean   = sub_hot.groupby("poison_rate")["accuracy"].mean()

        ax.plot(fever_mean.index, fever_mean.values,
                marker="o", color="#1f77b4", linewidth=2, markersize=7, label="FEVER")
        ax.plot(hot_mean.index,   hot_mean.values,
                marker="s", color="#ff7f0e", linewidth=2, markersize=7,
                linestyle="--", label="HotpotQA")

        ax.set_title(prompt, fontsize=12, fontweight="bold")
        ax.set_xlabel("Poison rate", fontsize=10)
        ax.set_ylabel("Mean accuracy (across strategies)", fontsize=9)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75])
        ax.set_ylim(-0.02, 0.80)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, frameon=False)

    fig.suptitle("FEVER vs HotpotQA: Mean Accuracy by Prompt Mode", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fever_vs_hotpot.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved fig_fever_vs_hotpot")
else:
    print("Skipped fig_fever_vs_hotpot (no results/hotpot/summary.csv yet)")
