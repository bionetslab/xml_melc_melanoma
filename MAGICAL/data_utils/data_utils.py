import pandas as pd
import os
import numpy as np


def get_high_quality_samples(base):
    quality_dict = np.load(os.path.join(base, "je30bery/melanoma_data/qualtiy_assessment/quality.npy"), allow_pickle=True).item()
    k = np.array(list(quality_dict.keys()))
    v = np.array(list(quality_dict.values()))
    high_quality = k[np.where(v == "2")]
    return high_quality
    
def get_data_csv(base="/data/bionets/", dataset="Melanoma", group="Melanoma", filter_column="Tumor stage", filter_value=None):

    path = os.path.join(base, "datasets/melc/melc_clinical_data.csv")
    data = pd.read_csv(path)

    data = data[data["Dataset"] == dataset]
    data = data[~data["file_path"].isna()]
    data = data[data["Group"] == group]

    #if filter_column:
    #    if filter_value:
    #        data = data[data[filter_column] == filter_value]
    #    else:   
    #        data = data[~data[filter_column].isna()]

    data["Collection year"] = data["Collection data"].apply(lambda x: x[-4:])
    data["Imaging year"] = data["file_path"].apply(lambda x: x.split("_")[2][:4])
    #data = data[~data["Tumor stage"].isna()]

    high_quality = get_high_quality_samples(base)
    data = data[data["file_path"].isin(high_quality)]
    data = data.set_index("file_path").drop(["Melanoma_13_201907111415_3", "Melanoma_13_201907111415_4"], axis=0).reset_index()
    float_ts = {np.nan: 0,
        'T1a':1/8,
    'T1b':2/8,
    'T2a':3/8,
    'T2b':4/8,
    'T3a':5/8,
    'T3b':6/8,
    'T4a':7/8,
    'T4b':8/8,
    'T4b N1b':8/8}
    data["Float tumor stage"] = data["Tumor stage"].apply(lambda x: float_ts[x])

    return data