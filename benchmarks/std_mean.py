
import numpy as np
import matplotlib.pyplot as plt


algorithms = ["HST", "RZS", "RC-SRE", "TRAKR", "TranAD"]


'''================SMD+SMAP+MSL================
means = {
    "Precision": [0.5185, 0.5286, 0.7234, 0.7549, 0.7161],
    "Recall":    [0.6932, 0.7954, 0.8364, 0.8296, 0.9139],
    "Accuracy":  [0.9663, 0.9630, 0.9822, 0.9841, 0.9719],
    "F1-score":  [0.5580, 0.6104, 0.7579, 0.7702, 0.7651],
    "MCC":       [0.5719, 0.6246, 0.7626, 0.7756, 0.7794],
}

stds = {
    "Precision": [0.3430, 0.3402, 0.2939, 0.2825, 0.2962],
    "Recall":    [0.4326, 0.3969, 0.3229, 0.3237, 0.2339],
    "Accuracy":  [0.0314, 0.0393, 0.0172, 0.0149, 0.0371],
    "F1-score":  [0.3515, 0.3478, 0.2893, 0.2840, 0.2565],
    "MCC":       [0.3509, 0.3452, 0.2884, 0.2817, 0.2424],
}
'''

'''================SMD================
means = {
    "Precision": [0.7804, 0.5761, 0.8865, 0.9089, 0.8777],
    "Recall":    [0.6633, 0.9421, 0.8320, 0.7526, 0.8072],
    "Accuracy":  [0.9746, 0.9608, 0.9852, 0.9850, 0.9823],
    "F1-score":  [0.6587, 0.6938, 0.8299, 0.7873, 0.8076],
    "MCC":       [0.6779, 0.7116, 0.8380, 0.8020, 0.8166],
}

stds = {
    "Precision": [0.1547, 0.2398, 0.1669, 0.1393, 0.1836],
    "Recall":    [0.3161, 0.2183, 0.2111, 0.2712, 0.2269],
    "Accuracy":  [0.0205, 0.0388, 0.0200, 0.0149, 0.0294],
    "F1-score":  [0.2168, 0.2348, 0.1618, 0.2005, 0.2075],
    "MCC":       [0.1934, 0.2212, 0.1481, 0.1804, 0.1868],
}

'''

''' ================SMAP+MSL================
means = {
    "Precision": [0.4471, 0.5157, 0.6789, 0.7129, 0.6742],
    "Recall":    [0.7013, 0.7554, 0.8377, 0.8506, 0.9416],
    "Accuracy":  [0.9641, 0.9635, 0.9814, 0.9839, 0.9692],
    "F1-score":   [0.5305, 0.5876, 0.7382, 0.7656, 0.754],
    "MCC":       [0.5429, 0.6008, 0.742, 0.7684, 0.7697],
}

stds = {
    "Precision": [0.346, 0.363, 0.3058, 0.2974, 0.3062],
    "Recall":    [0.4607, 0.4254, 0.3483, 0.3351, 0.229],
    "Accuracy":  [0.0335, 0.0397, 0.0164, 0.0149, 0.0385],
    "F1-score":   [0.3765, 0.3708, 0.3132, 0.3037, 0.2679],
    "MCC":       [0.3786, 0.3695, 0.3137, 0.3041, 0.255],
}
'''


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 17,
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
