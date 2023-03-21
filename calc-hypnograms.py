"""
for other dataset
labels = {'Fp1' 'AF7' 'AF3' 'F1' 'F3' 'F5' 'F7' 'FT7' 'FC5' 'FC3' 'FC1' 'C1' 'C3' 'C5' 'T7' 'TP7' 'CP5' 'CP3' 'CP1' 'P1' 'P3' 'P5' 'P7' 'P9' 'PO7' 'PO3' 'O1' 'Iz' 'Oz' 'POz' 'Pz' ...
    'CPz' 'Fpz' 'Fp2' 'AF8' 'AF4' 'AFz' 'Fz' 'F2' 'F4' 'F6' 'F8' 'FT8' 'FC6' 'FC4' 'FC2' 'FCz' 'Cz' 'C2' 'C4' 'C6' 'T8' 'TP8' 'CP6' 'CP4' 'CP2' 'P2' 'P4' 'P6' 'P8' 'P10' 'PO8' 'PO4' 'O2'};
channels = 1:64;
%labels = {'C3' 'O2' 'C4' 'O1' 'FP1' 'FP2' 'F7' 'F3' 'FZ' 'F4' 'F8' 'T3' 'CZ' 'T4' 'T5' 'P3' 'PZ' 'P4' 'T6'}; % Electrode Names for variable output
%channels = [3 4 5 6 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]; % Corresponding channels in EEG dataset, should match order of names in variable above
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import yasa

import utils


mne.set_log_level(False)
yasa.io.set_log_level(False)

root_dir = Path(utils.config["root_dir"])
datasets = utils.config["datasets"]


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, choices=datasets)
args = parser.parse_args()

dataset = args.dataset


# Get a list of all the EEG filepaths.
dataset_dir = root_dir / "sourcedata" / dataset
if dataset == "antony2018classy":
    filepaths = dataset_dir.glob("*/sleep/TIB.set")
elif dataset == "antony2018tonehero":
    filepaths = dataset_dir.glob("*/EEGLAB_Datasets/TIB/TIB.set")
elif dataset == "konkoly202Xalpha":
    filepaths = dataset_dir.glob("*.set")

epoch_length = 30

if dataset.startswith("antony2018"):
    channels = dict(eeg_channel="EEG 000")
elif dataset == "konkoly202Xalpha":
    channels = dict(eeg_channel="C4", eog_channel="R-HEOG", emg_channel="EMG")

# Loop over each EEG file, get human hypnogram, and generate YASA hypnogram.
for fp in tqdm(list(filepaths), desc=dataset, leave=False):

    # Get human hypnogram.
    if dataset.startswith("antony2018"):
        hypno_manual_fp = fp.with_stem("stages").with_suffix(".txt")
        hypno_manual_int = np.loadtxt(hypno_manual_fp)
        hypno_mapping = {
            -2: "Uns",
            -1: "Art",
            0: "W",
            1: "N1",
            2: "N2",
            3: "N3",
            4: "N3",
            5: "R",
        }
    elif dataset == "konkoly202Xalpha":
        hypno_manual_fp = fp.with_name(f"{fp.stem}KK").with_suffix(".mat")
        mat = loadmat(hypno_manual_fp)
        hypno_int = mat["stageData"]["stages"][0, 0].flatten()
        hypno_mapping = {
            0: "W",
            1: "N1",
            2: "N2",
            3: "N3",
            5: "R",
            7: "Uns",
        }
    hypno_manual_str = yasa.hypno_int_to_str(hypno_manual_int, mapping_dict=hypno_mapping)

    participant_number = [int(x) for x in fp.parts if x.isdigit()]
    assert len(participant_number) == 1
    participant_number = participant_number[0]
    participant_id = f"sub-{participant_number:03d}"

    # Load raw data.
    if fp.suffix == ".set":
        raw = mne.io.read_raw_eeglab(fp)
    else:
        raise ValueError(f"Unexpected EEG filetype: {fp.suffix}")

    # Drop to the only channels needed for staging.
    raw.pick([eeg_channel])
    # Load data.
    raw.load_data()
    # Downsample to 100 Hz.
    raw.resample(100)
    # Apply a bandpass filter to all channels.
    raw.filter(0.1, 40)

    # Perform YASA's automatic sleep staging.
    sls = yasa.SleepStaging(raw, **channels)

    hypno_yasa_str = sls.predict()
    hypno_proba = sls.predict_proba().add_prefix("proba_")

    # Generate events dataframe for hypnogram.
    n_epochs = len(hypno_yasa_str)
    if dataset == "antony2018tonehero" and participant_number == 321:
        hypno_manual_str = hypno_manual_str[:n_epochs]
    hypno_yasa_int = yasa.hypno_str_to_int(hypno_yasa_str)
    hypno_events = {
        "onset": [epoch_length*i for i in range(n_epochs)],
        "duration": epoch_length,
        # "value" : hypno_yasa_int,
        "yasa" : hypno_yasa_str,
        "human" : hypno_manual_str,
        # "scorer": f"YASA-v{yasa.__version__}",
        # "eeg_channel": eeg_channel,
        # "manual": hypno
    }
    hypno = pd.DataFrame.from_dict(hypno_events).rename_axis("epoch").join(hypno_proba)

    # Export.
    export_path = root_dir / "derivatives" / dataset / participant_id / f"{participant_id}_hypno.tsv"
    utils.export_tsv(hypno, export_path)
