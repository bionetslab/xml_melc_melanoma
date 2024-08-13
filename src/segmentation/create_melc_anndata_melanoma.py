import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import anndata as ad
import sys
sys.path.append("../")
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
    hpa_data_path = get_config_value(config, "hpa_data")
    segment = "cell"
    
    with open(antibody_mapping_path) as f:
        antibody_gene_symbols = json.load(f)
        
    print("getting reference data")
    reference = get_hpa_reference("skin", hpa_data_path)
    print("done")
    os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)
    result_dict = dict()

    EA = ExpressionAnalyzer(data_path=data_path, segmentation_results_dir_path=seg_results_path, membrane_markers=["CD45"], markers_of_interest=list(antibody_gene_symbols.keys()))
    EA.run(segment=segment, profile=None)

    expression_data = EA.expression_data.sort_index()
    #data = get_data_csv(groups=["Melanoma"], high_quality_only=False, pfs=False, config_path=config_path)
    data = get_data_csv(groups=["Melanoma"], high_quality_only=True, pfs=True, config_path=config_path)
    #data = data.set_index("Sample")
    dfs = dict()
    markers_used_for_assignment = None
    for i, sample in tqdm(enumerate(data["Sample"])):
        exp_fov = expression_data.loc[sample].copy()
        exp_fov = exp_fov.drop(["Sample", "Group", "Index", "Field of View", "Bcl-2"], axis=1)
        exp_fov = exp_fov.dropna(axis="columns", how="any")
        
        df = pd.DataFrame()
        for c in exp_fov.columns:
            if c in ["CD45RA", "CD45RO", "PPB", 'CD66abce']:
                continue
            symbol = antibody_gene_symbols[c]
            if isinstance(symbol, list):
                for s in symbol:
                    df[s] = exp_fov[c]
            else:
                df[symbol] = exp_fov[c]
        dfs[sample] = df

        if markers_used_for_assignment is None:
            markers_used_for_assignment = df.columns
        else:
            markers_used_for_assignment = list(set(markers_used_for_assignment) & set(list(df.columns)))

    
    for i, sample in tqdm(enumerate(dfs.keys())):
        df = dfs[sample][markers_used_for_assignment]
        segmented = os.path.join(seg_results_path, f'{sample}_cell.npy')
        with open(segmented, "rb") as openfile:
            seg_file = np.load(openfile)
        
        adata = ad.AnnData(df)
        adata.var = pd.DataFrame(np.array(df.columns), columns=["gene_symbol"])#
        adata.var_names = df.columns
        adata.uns["segmentation"] = seg_file
        adata.obsm["field_of_view"] = np.array([sample] * df.shape[0]) 
        
        tree = identify_cell_types(adata, reference[markers_used_for_assignment].copy(), min_fold_change=2, z_score_cutoff=1.96/4)
        adata.uns["cell_type_assignment_tree"] = tree
        result_dict[sample] = adata
        
    print(len(result_dict))
    ad_dir = os.path.join(seg_results_path, "anndata_files")
    os.makedirs(ad_dir, exist_ok=True)
    with open(os.path.join(ad_dir, f"adata_{segment}.pickle"), 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote", os.path.join(ad_dir, f"adata_{segment}.pickle"))
    
if __name__ == "__main__":
    main()