"""Plot hypnogram(s)?"""
import argparse
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as skm

import utils


root_dir = Path(utils.config["root_dir"])
datasets = utils.config["datasets"]

export_path = root_dir / "derivatives" / "accuracy.png"

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


def scorer(df):
    t, p = zip(*df.values)  # Same as (df["col1"], df["col2"]) but teensy bit faster
    return {
        "accuracy": skm.accuracy_score(t, p, normalize=True),
        "balanced_acc": skm.balanced_accuracy_score(t, p, adjusted=False),
        "kappa": skm.cohen_kappa_score(t, p, labels=None, weights=None),
        "mcc": skm.matthews_corrcoef(t, p),
        "precision": skm.precision_score(
            t, p, average="weighted", zero_division=0
        ),
        "recall": skm.recall_score(t, p, average="weighted", zero_division=0),
        "fbeta": skm.fbeta_score(
            t, p, beta=1, average="weighted", zero_division=0
        ),
    }

subjs = df.groupby(["dataset", "participant_id"])[["human", "yasa"]].apply(scorer).apply(pd.Series)
dsets = subjs.groupby("dataset").agg(["mean", "sem"]).stack(0).rename_axis(["dataset", "metric"])


fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
ax = sns.boxplot(
    ax=ax,
    data=subjs.reset_index(),
    x="dataset",
    y="accuracy",
    notch=True,
    color="cornflowerblue",
    saturation=1,
)
ax.set_ybound(upper=1)
ax.set_xlabel("Dataset")
ax.set_ylabel("Accuracy")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
ax.spines[["top", "right"]].set_visible(False)


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
