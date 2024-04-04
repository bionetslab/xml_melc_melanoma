import pandas as pd
import os
import numpy as np
import json

def get_high_quality_samples(path):    
    """ 
    Get high-quality samples

    Parameters:
    - path (str): The path to the dictionairy containing quality information.
    Returns:
    - high_quality (array): Array containing high-quality sample fovs.
    """
    quality_dict = np.load(path, allow_pickle=True).item()
    k = np.array(list(quality_dict.keys()))
    v = np.array(list(quality_dict.values()))
    high_quality = k[np.where(v == "2")]
    return high_quality
    
def get_data_csv(dataset="Melanoma", groups=["Melanoma"], high_quality_only=True, config_path="/data_nfs/je30bery/melanoma_data/config.json"):    
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

    data = data[data["Dataset"] == dataset]
    data = data[~data["file_path"].isna()]
    data = data[data["Group"].isin(groups)]

    data["Collection year"] = data["Collection data"].apply(lambda x: x[-4:])
    data["Imaging year"] = data["file_path"].apply(lambda x: x.split("_")[2][:4])

    data = data[(data["Group"] == "Nevus") | (data["Tumor stage"].notna())]


    if high_quality_only:
        high_quality = get_high_quality_samples(configs["high_quality_samples"])
        data = data[data["file_path"].isin(high_quality)]
    float_ts = {np.nan: 0,
                'T1a':1/4,
                'T1b':1/4,
                'T2a':2/4,
                'T2b':2/4,
                'T3a':3/4,
                'T3b':3/4,
                'T4a':4/4,
                'T4b':4/4,
                'T4b N1b':4/4}
    data["Float tumor stage"] = data["Tumor stage"].apply(lambda x: float_ts[x])

    return data