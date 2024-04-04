import numpy as np
import pandas as pd
import torch as t
import multiprocessing as mp
from trainer import Trainer
from sklearn.utils import shuffle
import os
import json
from data_utils import get_data_csv

from data import MelanomaData
from model import EfficientnetWithFinetuning

def balance(pat_data, split_by="split", variable="Label"):
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


def get_data(data, splits = {"train": 0.8, "val": 0.2}, balance_by="Group"):
    subsets = list()
    data = data.dropna(subset=[balance_by], axis=0)
    groups = np.unique(data[balance_by])
    while True:
        try:
            for group in groups:
                subset = data[data[balance_by] == group].copy()
                histo_ids = np.unique(subset["Histo-ID"])
                split = np.random.choice(list(splits.keys()), len(histo_ids), p=list(splits.values()))
                histo_split = {hist_id: split[i] for i, hist_id in enumerate(histo_ids)}
                subset["split"] = subset["Histo-ID"].apply(lambda x: histo_split[x])
                subsets.append(subset)
            data = pd.concat(subsets, axis=0)
            data = balance(data, "split", balance_by)
            break
        except ValueError:
            continue
    return data

def main(classifier = True):
    if classifier:
        data = get_data_csv(dataset="Melanoma", groups=["Melanoma", "Nevus"], high_quality_only=False)
        data = get_data(data, splits = {"train": 0.8, "val": 0.2}, balance_by="Group")
    else:
        data = get_data_csv(dataset="Melanoma", groups=["Melanoma"], high_quality_only=False)
        data = get_data(data, splits = {"train": 0.8, "val": 0.2}, balance_by="Float tumor stage")

    with open(f'../data/dataset_statistics/melanoma_means.json', 'r') as fp:
        means = json.load(fp)
    markers = list(means.keys())

    vdl = t.utils.data.DataLoader(MelanomaData(markers, classifier, data[data["split"] == "train"], mode="train"), batch_size=20, shuffle=True)
    tdl = t.utils.data.DataLoader(MelanomaData(markers, classifier, data[data["split"] == "val"], mode="val"), batch_size=20, shuffle=True)

    return vdl
    
    model = EfficientnetWithFinetuning(indim=len(markers))
    #model.load_state_dict(t.load("/data_nfs/je30bery/melanoma_data/model/training/saved_models/model_2023-11-20 16:27:56.558630_f1=0.9260606060606061_acc=0.9166666666666666_11.pt"), strict=False)

    crit = t.nn.CrossEntropyLoss()

    if classifier:
        crit = t.nn.BCEWithLogitsLoss()
    else:
        crit = t.nn.MSELoss()

    optim = t.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    summary_writer_name = "classifier_new" # TODO

    trainer = Trainer(model, 
                    crit,
                    optim, 
                    tdl,
                    vdl,
                    device="cuda:1",
                    summary_writer_name=summary_writer_name)
    print("evoked trainer")
    
    res = trainer.fit(epochs=30)
    model.eval()
    t.save(model.state_dict(), f"../model/model_{datetime.datetime.now()}.pt")


    print('done')


    
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
if __name__ == "__main__":
    main()