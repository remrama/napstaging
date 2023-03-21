"""Run all analysis scripts in order."""
import os
import subprocess
import sys

from tqdm import tqdm

import utils


def run_command(command):
    """Run shell command and exit upon failure."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit()


for d in (pbar := tqdm(utils.config["datasets"])):
    pbar.set_description(d)
    calc_command = f"python calc-hypnograms.py --dataset {d}"
    run_command(calc_command)
    participants = utils.get_participant_list(d)
    for p in tqdm(participants, leave=False):
        plot_command = f"python plot-hypnogram.py --dataset {d} --participant {p}"
        run_command(plot_command)
