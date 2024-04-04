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
sys.path.append("/data/bionets/je30bery/ALS_MELC_Data_Analysis/segmentation/")
sys.path.append("/data/bionets/je30bery/ALS_MELC_Data_Analysis/marker_expression/")
from melc_segmentation import MELC_Segmentation
from initial_analysis import ExpressionAnalyzer
import anndata as ad
import warnings
warnings.filterwarnings("ignore")


data = "melanoma"

f = open("config.json")
config = json.load(f)
data_path = config[data]
seg_results_path = config["segmentation_results"]
# TODO os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)
seg = MELC_Segmentation(data_path, membrane_markers=None) 
# membrane_marker: str/None 
# radius: multiple of cell radius
comorbidity_info = False

antibody_gene_symbols = {
    'ADAM10': 'ADAM10',
    'Bcl-2': 'BCL2',
    'CD10': 'MME',
    'CD107a': 'LAMP1',
    'CD13': 'ANPEP',
    'CD138': 'SDC1',
    'CD14': 'CD14',
    'CD1a': 'CD1A',
    'CD2': 'CD2',
    'CD25': 'IL2RA',
    'CD271': 'NGFR',
    'CD3': ['CD3D', 'CD3E', 'CD3G'],
    'CD36': 'CD36',
    'CD4': 'CD4',
    'CD44': 'CD44',
    'CD45': 'PTPRC',
    'CD45RA': 'PTPRC',
    'CD45RO': 'PTPRC',
    'CD5': 'CD5',
    'CD56': 'NCAM1',
    'CD6': 'CD6',
    'CD63': 'CD63',
    'CD66abce': ['CD66A', 'CD66B', 'CD66C', 'CD66E'],
    'CD7': 'CD7',
    'CD71': 'TFRC',
    'CD8': ['CD8A', 'CD8B'],
    'CD9': 'CD9',
    'CD95': 'FAS',
    'Collagen IV': ['COL4A1', 'COL4A2'],
    'Cytokeratin-14': 'KRT14',
    'EBF-P': 'EBF1',
    'EGFR': 'EGFR',
    'EGFR-AF488': 'EGFR',
    'HLA-ABC': ['HLA-A', 'HLA-B', 'HLA-C'],
    'HLA-DR': ['HLA-DRA', 'HLA-DRB1', 'HLA-DRB3', 'HLA-DRB4', 'HLA-DRB5'],
    'KIP1': 'CDKN1B',
    'Ki67': 'MKI67',
    'L302': 'NCR3LG1',
    'MCSP': 'CSPG4',
    'Melan-A': 'MLANA',
    'Nestin-AF488': 'NES',
    'Notch-1': 'NOTCH1',
    'Notch-3': 'NOTCH3',
    'PPARgamma': 'PPARG',
    'PPB': 'TP63',
    'RIM3': 'RIMS3',
    'TAP73': 'TP73',
    'Vimentin': 'VIM',
    'p63': 'TP63',
    'phospho-Connexin': 'GJA1'
}

# TODO os.makedirs(os.path.join(seg_results_path, "anndata_files"), exist_ok=True)

segment = "cell"
result_dict = dict()

if comorbidity_info:
    comorbidities = pd.read_csv("/data_slow/je30bery/data/ALS/ALS_comorbidities.txt", delimiter=";")
    comorbidities = comorbidities.set_index("pat_id")

EA = ExpressionAnalyzer(data_path=data_path, segmentation_results_dir_path=seg_results_path, membrane_markers=None, markers_of_interest=list(antibody_gene_symbols.keys()))
EA.run(segment=segment, profile=None)

expression_data = EA.expression_data.sort_index()
print(expression_data.shape)
#expression_data = expression_data.fillna(0)
#expression_data = expression_data.drop_duplicates()

for i, fov in enumerate(tqdm(seg.fields_of_view)):
    if os.path.exists(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}_{fov}.pickle")):
        continue
    if "ipynb" in fov:
        continue

    seg.field_of_view = fov
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


    if comorbidity_info:
        """
        for c in comorbidities.columns:
            if "ALS" in sample:
                adata.obsm[str(c)] = np.array([str(comorbidities.loc[sample, c])]* exp_fov.shape[0])
                adata.uns[str(c)] = str(comorbidities.loc[sample, c])
            else:
                adata.obsm[str(c)] = np.array(["unknown"] * exp_fov.shape[0])
                adata.uns[str(c)] = "unknown"
        """
        pass

    adata.obsm["field_of_view"] = np.array([fov] * exp_fov.shape[0]) 
    #adata.uns["field_of_view"] = fov
    

    if "spatial" not in adata.uns:
        adata.uns["spatial"] = {}  # Create the "spatial" key if it doesn't exist

    adata.obsm["control_mean_expression"] = np.array([EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].mean(axis=0).values] * exp_fov.shape[0])
    adata.obsm["control_std_expression"] = np.array([EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].std(axis=0).values] * exp_fov.shape[0])       
    adata.uns["control_mean_expression"] = EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].mean(axis=0).values
    adata.uns["control_std_expression"] = EA.expression_data[EA.expression_data["Group"] == "Healthy"].iloc[:, :-2].std(axis=0).values


    adata.uns["cell_coordinates"] = where_dict
    adata.uns["spatial"]["segmentation"] = nuc if segment == "nuclei" else cell

    result_dict[i] = adata

print(result_dict.keys())
print(result_dict[0])

with open(os.path.join(seg_results_path, "anndata_files", f"adata_{segment}.pickle"), 'wb') as handle:
    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)