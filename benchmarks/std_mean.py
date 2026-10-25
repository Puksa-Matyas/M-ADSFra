
import numpy as np
import matplotlib.pyplot as plt


algorithms = ["HST", "RZS", "SPE", "TRAKR", "TranAD"]

means = {
    "Precision": [0.4471, 0.5157, 0.6789, 0.7129, 0.6742],
    "Recall":    [0.7013, 0.7554, 0.8377, 0.8506, 0.9416],
    "F-score":   [0.5305, 0.5876, 0.7382, 0.7656, 0.754],
    "Accuracy":  [0.9641, 0.9635, 0.9814, 0.9839, 0.9692],
    "MCC":       [0.5429, 0.6008, 0.742, 0.7684, 0.7697],
}

stds = {
    "Precision": [0.346, 0.363, 0.3058, 0.2974, 0.3062],
    "Recall":    [0.4607, 0.4254, 0.3483, 0.3351, 0.229],
    "F-score":   [0.3765, 0.3708, 0.3132, 0.3037, 0.2679],
    "Accuracy":  [0.0335, 0.0397, 0.0164, 0.0149, 0.0385],
    "MCC":       [0.3786, 0.3695, 0.3137, 0.3041, 0.255],
}


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

metrics = list(means.keys())
x = np.arange(len(metrics))

# Narrow bars and distinct hatch patterns
bar_width = 0.105
hatches = ["///", "\\\\\\", "...", "xxx", "---"]

fig, ax = plt.subplots(figsize=(11, 6))

for i, algorithm in enumerate(algorithms):
    positions = x + (i - 2) * 0.14

    values = np.array([means[m][i] for m in metrics])
    deviations = np.array([stds[m][i] for m in metrics])

    bars = ax.bar(
        positions,
        values,
        width=bar_width,
        label=algorithm,
        hatch=hatches[i],
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )

    # Thin I-shaped error bars: mean ± standard deviation
    lower = np.maximum(values - deviations, 0)
    upper = np.minimum(values + deviations, 1)

    ax.errorbar(
        positions,
        values,
        yerr=[values - lower, upper - values],
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.9,
        zorder=5,
        clip_on=True,
    )

ax.set_ylabel("Score", labelpad=8)

ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Keep the metric scale within its valid range
ax.set_ylim(0.0, 1.0)
ax.set_yticks(np.arange(0.0, 1.01, 0.1))

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.55,
    alpha=0.35,
    zorder=0,
)
ax.set_axisbelow(True)

ax.legend(
    title="Algorithm",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=5,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.5,
)

fig.tight_layout()

fig.savefig(
    "smd_detector_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    "smd_detector_comparison.pdf",
    bbox_inches="tight",
)

plt.show()
