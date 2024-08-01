import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import anndata as ad
import sys
sys.path.append("..")
from src import *

import argparse

import warnings
warnings.filterwarnings("ignore")

def get_config_value(config, key):
    try:
        val = config[key]
    except:
        print("please specify", key, "in your config file")
        exit()
    return val
    
    
def main():
    # this file creates the segmentation files and expression data from the MELC images
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Config path')
    args = parser.parse_args()
    config_path = args.config_path
    
    with open(config_path) as f:
        config = json.load(f)
    print(config)
    data_path = get_config_value(config, "melanoma_data")
    seg_results_path = get_config_value(config, "segmentation_results") 
    antibody_mapping_path = get_config_value(config, "antibody_gene_mapping")
    segment = get_config_value(config, "segment")
    assert segment in ["nucleus", "cell"], "please choose \"cell\" or \"nuclues\" as segment"
    
    
    with open(antibody_mapping_path) as f:
        antibody_gene_symbols = json.load(f)

    seg = MELC_Segmentation(data_path, membrane_markers=["CD45"]) 
    os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)
    result_dict = dict()

    EA = ExpressionAnalyzer(data_path=data_path, segmentation_results_dir_path=seg_results_path, membrane_markers=["CD45"], markers_of_interest=list(antibody_gene_symbols.keys()))
    EA.run(segment=segment, profile=None)

    expression_data = EA.expression_data.sort_index()

    for i, fov in enumerate(tqdm(seg.fields_of_view)):
        if os.path.exists(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}_{fov}.pickle")):
            continue

        seg.field_of_view = fov
        
        with open(os.path.join(seg_results_path, f"{fov}_nucleus.pickle"), "rb") as handle:
            where_nuc = pickle.load(handle)
        with open(os.path.join(seg_results_path, f"{fov}_cell.pickle"), "rb") as handle:
            where_cell = pickle.load(handle)

        where_dict = where_nuc if segment == "nucleus" else where_cell   
        where_dict = dict(sorted(where_dict.items()))   
        
        exp_fov = expression_data.loc[fov].copy()
        exp_fov = exp_fov.drop(["Sample", "Group"], axis=1)
    
        adata = ad.AnnData(exp_fov)
        adata.var = pd.DataFrame(np.array(exp_fov.columns), columns=["gene_symbol"])
        
        #adata.obsm["cellLabelInImage"] = np.array([int(a) for a in list(exp_fov.index)])

        #adata.varm["antibody"] = pd.DataFrame(exp_fov.columns, columns=["antibody"])
        #adata.obsm["cellSize"] = np.array([len(where_dict[k][0]) for k in where_dict])      
        #adata.obsm["Group"] = np.array([group] * len(adata.obsm["cellSize"]))
                
        #adata.uns["patient_id"] = pat_id

        #adata.obsm["patient_label"] = np.array([pat_id] * len(adata.obsm["cellSize"]))
        adata.obsm["field_of_view"] = np.array([fov] * exp_fov.shape[0]) 
        #adata.uns["cell_coordinates"] = where_dict
        #adata.uns["spatial"]["segmentation"] = nuc if segment == "nuclei" else cell

        result_dict[i] = adata
        
    with open(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}.pickle"), 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
if __name__ == "__main__":
    main()