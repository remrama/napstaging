"""Run all analysis scripts in order."""
import subprocess
import sys

from tqdm import tqdm

import utils


def run_command(command):
    """Run shell command and exit upon failure."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit()

