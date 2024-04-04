import numpy as np
import pandas as pd
import torch as t
import multiprocessing as mp
from trainer import Trainer
from sklearn.utils import shuffle
import os
import json
from data_utils import get_data_csv
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from data import MelanomaData
from model import EfficientnetWithFinetuning, ResNetWithFinetuning, VGGWithFinetuning, EfficientnetWithFinetuningWithVGGClassifier

t.manual_seed(0)

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


def get_data(data, splits = {"train": 0.8, "val": 0.2}, balance_by="Group"):
    """
    Split and balance the dataset for training and validation. 
    Different field of views with the same Histo-ID (-> same patient) are kept within one split

    Parameters:
    - data (DataFrame): DataFrame containing the dataset.
    - splits (dict): Dictionary containing train-validation split ratios.
    - balance_by (str): The column name for balancing the dataset.

    Returns:
    - balanced_data (DataFrame): DataFrame with balanced classes for training and validation.
    """
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
    """
    Main function for training the classifier.

    Parameters:
    - classifier (bool): set to True for 1st experiment (Melanoma vs. Nevi) and to False for 2nd experiment (coarse tumor stage prediction)
    """
    classifier = False
    weight_decay = 1e-7
    lr = 9e-5
    batch_size = 32
    num_epochs = 50
    device = "cuda:0"
    
    summary_writer_name = f"classifier_only_lr={lr}_weight_decay={weight_decay}_epochs={num_epochs}_dropout_50"

    if classifier:
        data = get_data_csv(dataset="Melanoma", groups=["Melanoma", "Nevus"], high_quality_only=False)
        data = get_data(data, splits = {"train": 0.8, "val": 0.2}, balance_by="Group")
        data.to_csv("split.csv")
    else:
        data = pd.read_csv("split.csv")
        #data = get_data_csv(dataset="Melanoma", groups=["Melanoma"], high_quality_only=False)
        data = data[data["Group"] == "Melanoma"]
        data["Coarse tumor stage"] = data["Float tumor stage"] > 0.5
        data = balance(data, split_by="split", variable="Coarse tumor stage")        

    with open(f'../data/dataset_statistics/melanoma_means.json', 'r') as fp:
        means = json.load(fp)
    markers = list(means.keys())

    vdl = t.utils.data.DataLoader(MelanomaData(markers, classifier, data[data["split"] == "train"], mode="train"), batch_size=batch_size, shuffle=True)
    tdl = t.utils.data.DataLoader(MelanomaData(markers, classifier, data[data["split"] == "val"], mode="val"), batch_size=batch_size, shuffle=True)
    
    model = EfficientnetWithFinetuningWithVGGClassifier(indim=len(markers))
    #model.load_state_dict(t.load("/data_nfs/je30bery/melanoma_data/model/finetuned_effnet_with_LR_reduction_on_plateau.pt"), strict=False)

    crit = t.nn.BCELoss()
    optim = t.optim.Adam(model.vggclassifier.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optim, patience=5)
    #scheduler = CosineAnnealingLR(optim, T_max=num_epochs, eta_min=1e-7)
    #scheduler = StepLR(optim, 5, gamma=0.5)

    print(summary_writer_name)
    trainer = Trainer(model, 
                    crit,
                    optim, 
                    scheduler,
                    tdl,
                    vdl,
                    device=device,
                    summary_writer_name=summary_writer_name)
    print("evoked trainer")
    
    trainer.fit(epochs=num_epochs)
    model.eval()
    t.save(model.state_dict(), f"../model/effnet_tumor_stage_{datetime.datetime.now()}.pt")
    print('done')

if __name__ == "__main__":
    main()