import numpy as np
import pandas as pd
import torch as t
import multiprocessing as mp
from trainer import Trainer
from sklearn.utils import shuffle
import os
import json

from data import MelanomaData
from model import EfficientnetWithFinetuning



def main():
    """
    data = pd.read_csv("/data_nfs/datasets/melc/melc_clinical_data.csv")
    dataset = "Melanoma"
    data = data[data["Dataset"] == dataset]
    data = data[~data["file_path"].isna()]
    data = data[data["Group"].isin(["Eczema", "T-Cell Lymphoma"])]
    

    """
    data = pd.read_csv("/data_nfs/datasets/melc/melc_clinical_data.csv")
    dataset = "Melanoma"
    data = data[data["Dataset"] == dataset]
    data = data[~data["file_path"].isna()]
    data = data[data["Group"] == "Melanoma"]
    data = data.dropna(subset=["Tumor stage"], axis=0)

    splits = {"train": 0.8, "val": 0.2} #, "test": 0.1}
    data["split"] = np.random.choice(list(splits.keys()), len(data), p=list(splits.values()))
    
    float_ts = {'T1a':1/8,
    'T1b':2/8,
    'T2a':3/8,
    'T2b':4/8,
    'T3a':5/8,
    'T3b':6/8,
    'T4a':7/8,
    'T4b':8/8,
    'T4b N1b':8/8}
    
    data["Float tumor stage"] = data["Tumor stage"].apply(lambda x: float_ts[x])
            
    with open(f'/data_nfs/je30bery/melanoma_data/model/{dataset.lower()}_means.json', 'r') as fp:
        means = json.load(fp)
    markers = list(means.keys())
        
    vdl = t.utils.data.DataLoader(MelanomaData(markers, dataset, data[data["split"] == "train"], mode="train"), batch_size=20, shuffle=True)
    tdl = t.utils.data.DataLoader(MelanomaData(markers, dataset, data[data["split"] == "val"], mode="val"), batch_size=20, shuffle=True)

    print("loaded data")
    
    model = EfficientnetWithFinetuning(indim=len(markers))
    model.load_state_dict(t.load("/data_nfs/je30bery/melanoma_data/model/training/saved_models/model_2023-11-20 16:27:56.558630_f1=0.9260606060606061_acc=0.9166666666666666_11.pt"), strict=False)

    #crit = t.nn.CrossEntropyLoss()
    #crit = t.nn.BCEWithLogitsLoss()
    #crit = t.nn.BCELoss()
    crit = t.nn.MSELoss()
    optim = t.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    summary_writer_name = "tumor_stage_regressor=1e-5_pretraining_2" # TODO

    trainer = Trainer(model, 
                    crit,
                    optim, 
                    tdl,
                    vdl,
                    device="cuda:1",
                    summary_writer_name=summary_writer_name)
    print("evoked trainer")
    
    res = trainer.fit(epochs=50)
    print(res)
    model.eval()
    print('done')

    
    """
    
    for excluded_marker in markers:
        #less_markers = [m for m in markers if m != excluded_marker]
        less_markers = [excluded_marker] # [m for m in markers if m != excluded_marker]

        print(less_markers)

        #assert len(less_markers) == 1, "marker was not properly excluded"
                
        vdl = t.utils.data.DataLoader(MelanomaData(less_markers, vdata, mode="val"), batch_size=20, shuffle=True)
        tdl = t.utils.data.DataLoader(MelanomaData(less_markers, tdata, mode="train"), batch_size=20, shuffle=True)

        print("loaded data")
        
        model = EfficientnetWithFinetuning(indim=len(less_markers))
    
        #model = t.load("/data_nfs/je30bery/melanoma_data/model/model_2023-11-07 17:08:22.583593.pt")
        #model.eval()
        
        
        #crit = t.nn.CrossEntropyLoss()
        #crit = t.nn.BCEWithLogitsLoss()
        crit = t.nn.BCELoss()
        optim = t.optim.Adam(model.parameters(), lr=5e-5) #, weight_decay=1e-5)

        
        summary_writer_name = "_".join(less_markers) # TODO
        #summary_writer_name = less_markers[0]

        trainer = Trainer(model, 
                        crit,
                        optim, 
                        tdl,
                        vdl,
                        device="cuda:0",
                        summary_writer_name=summary_writer_name)
        print("evoked trainer")
        
        res = trainer.fit(epochs=30)
        print(res)
        model.eval()
        print('done')
    """


"""
def balance_and_shuffle(data):
    paths = [os.path.join(data, sample) for sample in os.listdir(data) if os.path.isdir(os.path.join(data, sample))]
    classes, counts = np.unique([os.path.basename(p).split("_")[0] for p in paths], return_counts=True)
    relative_counts = counts / np.max(counts)
    balanced_samples = list()
    for i, c in enumerate(relative_counts):
        class_samples = [p for p in paths if classes[i] in p]
        if c == 1:
            balanced_samples += class_samples
        else:
            balanced_samples += list(np.random.choice(class_samples, int(len(class_samples) * 1 / c)))

    return shuffle(balanced_samples)

vdata = balance_and_shuffle(os.path.join(data, "validation"))
tdata = balance_and_shuffle(os.path.join(data, "training"))

"""

if __name__ == "__main__":
    main()