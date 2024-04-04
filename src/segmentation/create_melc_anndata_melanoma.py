import matplotlib.pyplot as plt
import json
import os
import cv2
import time
import numpy as np
from csbdeep.utils import Path, normalize
import pandas as pd
import anndata as ad
from tqdm import tqdm
import pickle
import sys
sys.path.append("/data_nfs/je30bery/ALS_MELC_Data_Analysis/segmentation/")
sys.path.append("/data_nfs/je30bery/ALS_MELC_Data_Analysis/marker_expression/")
from melc_segmentation import MELC_Segmentation
from initial_analysis import ExpressionAnalyzer
import anndata as ad
import warnings
warnings.filterwarnings("ignore")


def main():
    # this file creates the segmentation files and expression data from the MELC melanoma images
    
    with open("../config.json") as f:
        config = json.load(f)

    data_path = config["melanoma_data"]
    seg_results_path = config["segmentation_results"]

    seg = MELC_Segmentation(data_path, membrane_markers=["CD45-"]) 


    with open(config["antibody_gene_mapping"]) as f:
        antibody_gene_symbols = json.load(f)

    os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)
    os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)

    segment = "cell"
    result_dict = dict()


    EA = ExpressionAnalyzer(data_path=data_path, segmentation_results_dir_path=seg_results_path, membrane_markers=["CD45-"], markers_of_interest=list(antibody_gene_symbols.keys()))
    EA.run(segment=segment, profile=None)

    expression_data = EA.expression_data.sort_index()

    for i, fov in enumerate(tqdm(seg.fields_of_view)):
        if os.path.exists(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}_{fov}.pickle")):
            continue
        if "ipynb" in fov:
            continue

        seg.field_of_view = fov
        # load segmentation files if they already exist
        if os.path.exists(os.path.join(seg_results_path, f"{fov}_nuclei.pickle")):
            with open(os.path.join(seg_results_path, f"{fov}_nuclei.pickle"), "rb") as handle:
                where_nuc = pickle.load(handle)
            with open(os.path.join(seg_results_path, f"{fov}_cell.pickle"), "rb") as handle:
                where_cell = pickle.load(handle)
            nuc = np.load(os.path.join(seg_results_path, f"{fov}_nuclei.npy"))
            cell = np.load(os.path.join(seg_results_path, f"{fov}_cells.npy"))
        else:
            try:
                nuc, cell, where_nuc, where_cell = seg.run()
            except:
                print("skipping", fov, " - no segmentation file")
                continue
        where_dict = where_nuc if segment == "nuclei" else where_cell   
        where_dict = dict(sorted(where_dict.items()))   
                
        group =  np.unique(expression_data.loc[fov]["Group"].astype(str).values)[0]
        pat_id = np.unique(expression_data.loc[fov]["Sample"].astype(str).values)[0]

        exp_fov = expression_data.loc[fov].copy()
        exp_fov = exp_fov.drop(["Sample", "Group"], axis=1)
        

        adata = ad.AnnData(exp_fov)
        adata.var = pd.DataFrame(np.array(exp_fov.columns), columns=["gene_symbol"])
        
        adata.obsm["cellLabelInImage"] = np.array([int(a) for a in list(exp_fov.index)])

        #adata.varm["antibody"] = pd.DataFrame(exp_fov.columns, columns=["antibody"])
        adata.obsm["cellSize"] = np.array([len(where_dict[k][0]) for k in where_dict])      
        adata.obsm["Group"] = np.array([group] * len(adata.obsm["cellSize"]))
                
        adata.uns["patient_id"] = pat_id

        adata.obsm["patient_label"] = np.array([pat_id] * len(adata.obsm["cellSize"]))
        adata.obsm["field_of_view"] = np.array([fov] * exp_fov.shape[0]) 
        adata.uns["cell_coordinates"] = where_dict
        adata.uns["spatial"]["segmentation"] = nuc if segment == "nuclei" else cell

        result_dict[i] = adata
        
    with open(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}.pickle"), 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
if __name__ == "__main__":
    main()