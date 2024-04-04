import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from utils import NeighborEnricher
import pickle
import anndata as ad
from scipy.stats import norm
import os
from tqdm import tqdm

sys.path.append("../../")
import MAGICAL as mg

base = "/data/bionets/"
path = os.path.join(base, "datasets/melc/melc_clinical_data.csv")
dataset = group = "Melanoma"
data = mg.get_data_csv(path=path, dataset=dataset, group=group)

quality_dict = np.load(os.path.join(base, "je30bery/melanoma_data/qualtiy_assessment/quality.npy"), allow_pickle=True).item()
k = np.array(list(quality_dict.keys()))
v = np.array(list(quality_dict.values()))
high_quality = k[np.where(v == "2")]

with open("/data/bionets/je30bery/melanoma_data/cell_type_analysis/THEORETIC/cell_coordinates.pickle", 'rb') as handle:
    cell_coordinates = pickle.load(handle)

data = data[data["file_path"].isin(high_quality)]
data = data[data["file_path"].isin(list(cell_coordinates.keys()))]

data = data.reset_index()
float_ts = {'T1a':1/8, 'T1b':2/8, 'T2a':3/8, 'T2b':4/8, 'T3a':5/8,'T3b':6/8, 'T4a':7/8, 'T4b':8/8, 'T4b N1b':8/8}
data["Float tumor stage"] = data["Tumor stage"].apply(lambda x: float_ts[x])

ne = NeighborEnricher(cell_coordinates, base)
tumor_stages = [['T1a', 'T1b'], ['T2a', 'T2b'], ['T3a', 'T3b'], ['T4b']]
z_scores = list()
for tumor_stage in tqdm(tumor_stages):
    data_subset = data[data["Tumor stage"].isin(tumor_stage)]
    count_dfs = list()
    for fov in tqdm(np.unique(data_subset["file_path"]), leave=False):
        idx_neighbor_graph = ne.build_idx_neighbor_graph(fov, k=20)
        roi_cells = ne.get_roi_cells(fov, layer="2")
        cell_types = ne.get_cell_types(fov)
        cell_type_neighbor_graph = ne.get_cell_type_neighbor_graph(idx_neighbor_graph, cell_types)
        count_df = ne.get_neighbor_counts(cell_type_neighbor_graph, fov, k=20)
        count_dfs.append(ne.get_neighbor_counts_with_roi(count_df, roi_cells=roi_cells))
    conc_count_df = pd.concat(count_dfs, axis=0)
    roi, rni, diff = ne.get_neighbour_count_difference(conc_count_df)
    z_scores = ne.get_zscores_for_neighbors(count_df, diff, 1000)
    ts = "_".join(tumor_stage)
    z_scores.to_csv(f"{ts}_z_scores_k={k}.csv")

