"""Plot hypnogram(s)?"""
import argparse
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import utils


root_dir = Path(utils.config["root_dir"])
datasets = utils.config["datasets"]

export_path = root_dir / "derivatives" / "heatmap.png"

# Stack all hypnogram dataframes into one dataframe.
dataframe_list = []
for d in datasets:
    dataset_dir = root_dir / "derivatives" / d
    participants = utils.get_participant_list(d)
    for p in participants:
        p_id = f"sub-{p:03d}"
        import_path = dataset_dir / p_id / f"{p_id}_hypno.tsv"
        hypno = pd.read_csv(import_path, sep="\t").assign(dataset=d, participant_id=p_id)
        dataframe_list.append(hypno)

df = pd.concat(dataframe_list, ignore_index=True)



true = df["human"].to_numpy()
pred = df["yasa"].to_numpy()

labels = ["W", "N1", "N2", "N3", "R"]
plot_labels = ["Wake", "N1", "N2", "SWS", "REM"]
cm = confusion_matrix(true, pred, labels=labels, normalize="true")
cm = pd.DataFrame(cm,
    index=pd.Index(plot_labels, name="Human"),
    columns=pd.Index(plot_labels, name="YASA"),
)
cm = cm.multiply(100)

fig, ax = plt.subplots(figsize=(3.5, 3), constrained_layout=True)

sns.heatmap(cm,
    ax=ax, square=True, cmap=cc.cm.blues, annot=True, fmt=".0f", linewidth=0.5,
)



utils.export_mpl(export_path)


# fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
# ax = sns.boxplot(
#     ax=ax,
#     data=subjs.reset_index(),
#     x="dataset",
#     y="kappa",
#     notch=True,
#     color="cornflowerblue",
#     saturation=1,
# )
# ax.set_xlabel("Dataset")
# ax.set_ylabel("Kappa")
# ax.spines[["top", "right"]].set_visible(False)
