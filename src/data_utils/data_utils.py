import pandas as pd
import os
import numpy as np
import json

def balance(pat_data, split_by="split", variable="Label"):
    """
    Balance the dataset by oversampling minority classes.

    Parameters:
    - pat_data (DataFrame): DataFrame containing the dataset.
    - split_by (str): The column name for splitting the dataset.
    - variable (str): The variable to balance.

    Returns:
    - balanced_data (DataFrame): DataFrame with balanced classes.
    """
    values = np.unique(pat_data[variable])
    for split in np.unique(pat_data[split_by]):
        split_data = pat_data[pat_data[split_by] == split]
        v, c = np.unique(split_data[variable], return_counts=True)
        if len(v) != len(values):
            raise ValueError("Split rejected, a split group does not contain samples of all variables")
        max_count = np.max(c)
        for val, count in zip(v, c):
            diff = max_count - count
            if diff == 0:
                continue
            over_sampled = split_data[split_data[variable] == val].sample(diff, replace=True)
            pat_data = pd.concat([pat_data, over_sampled])
    return pat_data



def get_data_csv(dataset="Melanoma", groups=["Melanoma"], high_quality_only=True, pfs=True, config_path="/data_nfs/je30bery/melanoma_data/config.json"):    
    """
    Load clinical data from a CSV file based on specified criteria.

    Parameters:
    - dataset (str): The dataset name.
    - groups (list): List of groups to filter -> ["Melanoma"] or ["Melanoma", "Nevi"]
    - high_quality_only (bool): Whether to include only high-quality samples -> False for CNN training, false for analyses
    - config_path (str): The path to the JSON configuration file.

    Returns:
    - data (DataFrame): DataFrame containing filtered clinical data.
    """
    with open(config_path, 'r') as f:
        configs = json.load(f)
    data = pd.read_csv(configs["clinical_data"])
    if high_quality_only:
        data = data[data["High-quality segmentation result"] == True]
    if pfs:
        data = data[data["PFS < 5"].isin([True, False])]
    return data


def get_val(PFS_data, histo_id, column):
    if histo_id in PFS_data.index:
        return PFS_data.loc[histo_id, column]
    else:
        return np.nan

def coarse_loc(x):
    if "thorakal" in x:
        return 0
    if "abdominal" in x or "adbominal" in x:
        return 1
    if "Rücken" in x:
        return 2
    if "Arm" in x or "Capillitium" in x:
        return 3
    if "Bein" in x:
        return 4

def left_right(x):
    if "rechts" in x:
        return -1
    if "links" in x:
        return 1
    else:
        return 0