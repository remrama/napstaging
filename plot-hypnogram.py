"""Plot hypnogram(s)?"""
import argparse
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


utils.set_matplotlib_style()


root_dir = Path(utils.config["root_dir"])
datasets = utils.config["datasets"]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, choices=datasets)
parser.add_argument("-p", "--participant", type=int, required=True)
args = parser.parse_args()

dataset = args.dataset
participant = args.participant

participant_id = f"sub-{participant:03d}"

import_path = root_dir / "derivatives" / dataset / participant_id / f"{participant_id}_hypno.tsv"
export_path = import_path.with_suffix(".png")

df = pd.read_csv(import_path, sep="\t")

# Convert hypnogram stages to integers to ensure proper y-axis order.
plot_order = ["N3", "N2", "N1", "R", "W"]
legend_order = ["W", "R", "N1", "N2", "N3"]
stage_labels = dict(N1="N1", N2="N2", N3="SWS", R="REM", W="Awake")
# cmap = cc.cm.get("CET_CBD2")
cmap = cc.cm.get("bwy")
stage_palette = {
    "N1": cmap(0.30),
    "N2": cmap(0.15),
    "N3": cmap(0.),
    "R": cmap(1.),
    "W": cmap(0.5),
}
# blues = utils.cmap2hex("blues", 4)
# stage_palette = {
#     "N1": blues[1],
#     "N2": blues[2],
#     "N3": blues[3],
#     "R": "indianred",
#     "W": "gray",
# }

stage_colors = [stage_palette[s] for s in plot_order]
proba_columns = [f"proba_{s}" for s in plot_order]
stack_kwargs = dict(alpha = 0.9)
n_stages = len(plot_order)

step_cmap = cc.cm.get("gwv")
step_kwargs = {
    "yasa": dict(lw=1, color=step_cmap(1.), ls="solid", zorder=1, label="YASA"),
    "human": dict(lw=1, color=step_cmap(0.), ls="dashed", zorder=2, label="Human"),
}

times = df["duration"].multiply(df["epoch"]).to_numpy()
times = times / 60

figsize = (4, 2)

fig, (ax0, ax1) = plt.subplots(
    nrows=2,
    figsize=figsize,
    sharex=True,
    sharey=False,
    gridspec_kw={"height_ratios": [2, 1]},
)

# Overlay YASA and human hypnograms.
for scorer in ["human", "yasa"]:
    values = df[scorer].map(plot_order.index).to_numpy()
    # values_rem = np.ma.masked_not_equal(values, plot_order.index("R"))
    ax0.step(times, values, **step_kwargs[scorer])

# YASA hypnodensity.
probas = df[proba_columns].T.to_numpy()
ax1.stackplot(times, probas, colors=stage_colors, **stack_kwargs)

ax0.set_yticks(range(n_stages))
ax0.set_yticklabels([stage_labels[s] for s in plot_order])
ax0.set_ylabel("Sleep Stage")
ax0.spines[["top", "right"]].set_visible(False)
ax0.tick_params(axis="both", which="both", direction="out", top=False, right=False)
ax0.set_ybound(upper=n_stages)
ax0.set_xbound(lower=0, upper=times.max())

ax1.set_ylabel("Probability")
ax1.set_xlabel("Time (minutes)")
ax1.tick_params(axis="both", which="both", direction="out", top=False, right=False)
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(1 / n_stages))
ax1.xaxis.set_major_locator(plt.MultipleLocator(30))
ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
ax1.grid(which="minor", axis="y")

ax0.legend(
    title="Scorer",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.2),
    borderaxespad=0,
    labelspacing=.01,
    ncol=2,
    # fontsize=6,
)
legend_handles = [
    plt.matplotlib.patches.Patch(
        label=stage_labels[s], facecolor=stage_palette[s], edgecolor="black", linewidth=0.5
    ) for s in legend_order
]
legend = ax1.legend(handles=legend_handles,
    loc="upper left", bbox_to_anchor=(1, 1),
    # title="Stage",
    # handlelength=1, handleheight=.3,
    # handletextpad=,
    borderaxespad=0,
    labelspacing=.01,
    # columnspacing=,
    ncol=1,
    fontsize=6,
)

fig.align_ylabels()


# Export.
utils.export_mpl(export_path)
