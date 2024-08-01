import numpy as np
import pandas as pd
import torch as t
import os
import json
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

import sys
sys.path.append("..")
from src import *

t.manual_seed(0)


def get_data(data, splits = {"train": 0.8, "val": 0.2}, balance_by="Group"):
    """
    Split and balance the dataset for training and validation. 
    Different field of views with the same Histo ID (-> same patient) are kept within one split

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
                histo_ids = np.unique(subset["Histo ID"])
                split = np.random.choice(list(splits.keys()), len(histo_ids), p=list(splits.values()))
                histo_split = {hist_id: split[i] for i, hist_id in enumerate(histo_ids)}
                subset["split"] = subset["Histo ID"].apply(lambda x: histo_split[x])
                subsets.append(subset)
            data = pd.concat(subsets, axis=0)
            data = balance(data, "split", balance_by)
            break
        except ValueError:
            continue
    return data

def main():    
    """
    Main function for training the classifier.

    Parameters:
    - pretraining (bool): set to True for 1st experiment (Melanoma vs. Nevi) and to False for 2nd experiment (coarse tumor stage prediction)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Config path')
    parser.add_argument('--device', type=str, default="cuda:0", required=False, help='Cuda device')
    parser.add_argument('--pretraining', action='store_true', help='Enable pre-training mode.')
    parser.add_argument('--finetuning', action='store_false', dest='pretraining', help='Enable fine-tuning mode.')
    parser.add_argument('--index', required=False, default="all", help='Experiment index')
    parser.add_argument('--val_samples', required=False, nargs='+', default=None, help='Hold-out set')
    parser.add_argument('--leave_out', required=False, nargs='+', default=None, help='Experiment 2c')

    print("RUNNING WITH:")    
    args = parser.parse_args()
    config_path = args.config_path
    print(config_path)
    leave_out = args.leave_out
    print(leave_out)
    pretraining = args.pretraining
    print(pretraining)
    device = args.device
    print(device)
    idx = args.index
    print(idx)
    val_samples = args.val_samples
    print(val_samples)    
    
    if pretraining:
        print("PRE-TRAINING")
    else:
        print("FINE-TUNING")
    
    weight_decay = 0    
    if pretraining:
        lr = 0.0005    
        num_epochs = 1
    else:
        lr = 0.001
        num_epochs = 5
    batch_size = 20
    patience = 15    

    with open(config_path, "r") as f:
        configs = json.load(f)
        dataset_statistics = configs["dataset_statistics"]
        checkpoint_path = configs["downloaded_model_weights"]
        pretrained_model_path = configs["pretrained_model_path"]


    if pretraining:
        summary_writer_name = f"pretraining_left_out={leave_out}_split={str(idx)}_lr={lr}_wd={weight_decay}_bs={batch_size}"
    else:
        summary_writer_name = f"finetuning_left_out={leave_out}_split={str(idx)}_lr={lr}_wd={weight_decay}_bs={batch_size}"
    
    if pretraining:
        data = get_data_csv(dataset="Melanoma", groups=["Melanoma", "Nevus"], high_quality_only=False, config_path=config_path, pfs=False)
        data["Histo ID"] = data["Histo ID"].astype(str)
        data = data.set_index("Histo ID")
        data["split"] = "train" 
        if val_samples:
            data.loc[val_samples, "split"] = "val"
            data = balance(data, variable="Group")

        print(len(data[(data["split"] == "train") & (data["Group"] == "Nevus")]))
        print(len(data[(data["split"] == "train") & (data["Group"] == "Melanoma")]))
        print(len(data[(data["split"] == "val") & (data["Group"] == "Nevus")]))
        print(len(data[(data["split"] == "val") & (data["Group"] == "Melanoma")]))
        
        if leave_out:
            data = data.drop(leave_out)
            print("dropped", leave_out)
        #data.to_csv(f"split_{idx}.csv")
    else:
        data = get_data_csv(dataset="Melanoma", groups=["Melanoma"], high_quality_only=False, config_path=config_path, pfs=True)
        data["Histo ID"] = data["Histo ID"].astype(str)
        data = data.set_index("Histo ID")
        data["split"] = "train"
        if val_samples:
            data.loc[val_samples, "split"] = "val"
            train_data = balance(data[data["split"] == "train"], variable="PFS < 5")
            data = pd.concat([data[data["split"] == "val"], train_data])

        print(len(data[(data["split"] == "train") & (data["PFS < 5"] == 0)]))
        print(len(data[(data["split"] == "train") & (data["PFS < 5"] == 1)]))
        print(len(data[(data["split"] == "val") & (data["PFS < 5"] == 0)]))
        print(len(data[(data["split"] == "val") & (data["PFS < 5"] == 1)]))

            
    data.reset_index(inplace=True)

    with open(os.path.join(dataset_statistics, f'melanoma_means.json'), 'r') as fp:
        means = json.load(fp)
    markers = list(means.keys())

    tdl = t.utils.data.DataLoader(MelanomaData(markers, pretraining, data[data["split"] == "train"], mode="train", config_path=config_path), batch_size=batch_size, shuffle=True)
    if val_samples:
        vdl = t.utils.data.DataLoader(MelanomaData(markers, pretraining, data[data["split"] == "val"], mode="val", config_path=config_path), batch_size=batch_size, shuffle=True)
    else:
        vdl = None
    
    if pretraining:
        model = ResNet18_pretrained(indim=len(markers), cam=False, checkpoint_path=checkpoint_path)
    else:
        model = ResNet18_pretrained_for_finetuning(indim=len(markers), cam=False, checkpoint_path=checkpoint_path)
        if leave_out:
            model.load_state_dict(t.load(f"./{leave_out}.pt"), strict=False)
            print("loading", f"./{leave_out}.pt")
        else:
            model.load_state_dict(t.load(pretrained_model_path), strict=False)

    crit = t.nn.BCELoss()
    if pretraining:
        optim = t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # only pass classifier parameters to optimizer
        optim = t.optim.Adam(model.res.fc.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = StepLR(optim, patience, gamma=0.1)

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
    
    #t.save(model.state_dict(), f"../model/finetuned_{idx}_{datetime.datetime.now()}_left_out={leave_out}.pt")
    print('done')

if __name__ == "__main__":
    main()