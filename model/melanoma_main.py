import numpy as np
import pandas as pd
import torch as t
import multiprocessing as mp
from trainer import Trainer
from sklearn.utils import shuffle
import os

from data import MelanomaData
from model import EfficientnetWithFinetuning

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

def main():
    data = "/data_nfs/datasets/melc/melanoma/processed"
    vdata = balance_and_shuffle(os.path.join(data, "validation"))
    tdata = balance_and_shuffle(os.path.join(data, "training"))

    #markers = ["Propidium", "CD274", "Melan-A", "phase", "CD95", "Bcl-2"]
    markers = ["Propidium", "ADAM10-PE", "Melan-A", "CD63"]
    markers = ["HLA-DR", "CD3", "CD8", "ADAM10-PE", "CD63", "Melan-A"]
   
    vdl = t.utils.data.DataLoader(MelanomaData(markers, vdata, mode="val"), batch_size=20)
    tdl = t.utils.data.DataLoader(MelanomaData(markers, tdata, mode="train"), batch_size=20)

    print("loaded data")
    
    model = EfficientnetWithFinetuning(indim=len(markers))
    crit = t.nn.CrossEntropyLoss()
    optim = t.optim.Adam(model.parameters(), lr=1e-4) #, weight_decay=1e-5)

    
    trainer = Trainer(model, 
                     crit,
                     optim, 
                     tdl,
                     vdl,
                     device="cuda:0")
    print("evoked trainer")
    
    res = trainer.fit(epochs=200)
    print(res)
    model.eval()
    print('done')


if __name__ == "__main__":
    main()