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

    
    #markers = ["HLA-DR", "CD3", "CD8", "ADAM10-PE", "CD63", "Melan-A"]        
    #markers = ["CD63", "Melan-A", "PMEL17"]    


    data = "/data_nfs/datasets/melc/melanoma/processed/training"
    markers = ['ADAM10', 'Bcl-2', 'CD10', 'CD107a', 'CD13', 'CD138', 'CD14', 'CD1a', 'CD2', 'CD25', 'CD271', 'CD3', 'CD36', 'CD4', 'CD44', 'CD45', 'CD45RA', 'CD45RO', 'CD5', 'CD56', 'CD6', 'CD63', 'CD66abce', 'CD7', 'CD71', 'CD8', 'CD9', 'CD95', 'Collagen IV', 'Cytokeratin-14', 'EBF-P', 'EGFR', 'EGFR-AF488', 'HLA-ABC', 'HLA-DR', 'KIP1', 'Ki67', 'L302', 'MCSP', 'Melan-A', 'Nestin-AF488', 'Notch-1', 'Notch-3', 'PPARgamma', 'PPB', 'RIM3', 'TAP73', 'Vimentin', 'p63', 'phospho-Connexin']
    
    vdl = t.utils.data.DataLoader(MelanomaData(markers, vdata, mode="val"), batch_size=20, shuffle=True)
    tdl = t.utils.data.DataLoader(MelanomaData(markers, tdata, mode="train"), batch_size=20, shuffle=True)

    print("loaded data")
    
    model = EfficientnetWithFinetuning(indim=len(markers))

    #model = t.load("/data_nfs/je30bery/melanoma_data/model/model_2023-11-07 17:08:22.583593.pt")
    #model.eval()
    
    
    #crit = t.nn.CrossEntropyLoss()
    #crit = t.nn.BCEWithLogitsLoss()
    crit = t.nn.BCELoss()
    optim = t.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    summary_writer_name = "all_with_weight_decay" # TODO

    trainer = Trainer(model, 
                    crit,
                    optim, 
                    tdl,
                    vdl,
                    device="cuda:0",
                    summary_writer_name=summary_writer_name)
    print("evoked trainer")
    
    res = trainer.fit(epochs=20)
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


if __name__ == "__main__":
    main()