import pandas as pd
import os
import numpy as np

def get_high_quality_samples(base):
    quality_dict = np.load(os.path.join(base,"je30bery/melanoma_data/data/metadata/patient_data/quality.npy"), allow_pickle=True).item()
    k = np.array(list(quality_dict.keys()))
    v = np.array(list(quality_dict.values()))
    high_quality = k[np.where(v == "2")]
    return high_quality
    
def get_data_csv(dataset="Melanoma", groups=["Melanoma"], filter_column="Tumor stage", filter_value=None, high_quality_only=True):
    base = "/data/bionets" if "ramses" in os.uname()[1] else "/data_nfs/"

    path = os.path.join(base, "datasets/melc/melc_clinical_data.csv")
    data = pd.read_csv(path)

    data = data[data["Dataset"] == dataset]
    data = data[~data["file_path"].isna()]
    data = data[data["Group"].isin(groups)]

    data["Collection year"] = data["Collection data"].apply(lambda x: x[-4:])
    data["Imaging year"] = data["file_path"].apply(lambda x: x.split("_")[2][:4])

    data = data[(data["Group"] == "Nevus") | (data["Tumor stage"].notna())]


    if high_quality_only:
        high_quality = get_high_quality_samples(base)
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