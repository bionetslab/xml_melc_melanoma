import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import anndata as ad
from scipy.stats import norm
import os
from tqdm import tqdm
from neighbor_graphs import *
sys.path.append("../data_utils")
from data_utils import *

base = "/data/bionets/"
dataset = group = "Melanoma"
data = get_data_csv(base=base, dataset=dataset, group=group)


with open(os.path.join(base, "je30bery/melanoma_data/cell_type_analysis/THEORETIC/cell_coordinates.pickle"), 'rb') as handle:
    cell_coordinates = pickle.load(handle)

data = data[data["file_path"].isin(list(cell_coordinates.keys()))]

data = data.reset_index()

ne = NeighborEnricher(cell_coordinates, base)
tumor_stages = [['T1a', 'T1b'], ['T2a', 'T2b'], ['T3a', 'T3b'], ['T4b']]
layer = "combined_0-3"

os.makedirs(os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/z_scores/{layer}/"), exist_ok=True)
os.makedirs(os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/neighbor_graphs/{layer}"), exist_ok=True)


for k in [20, 11, 6, 2]:
    for tumor_stage in tqdm(tumor_stages):
        data_subset = data[data["Tumor stage"].isin(tumor_stage)]
        count_dfs = list()
        for fov in tqdm(np.unique(data_subset["file_path"]), leave=False):
            p = os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/neighbor_graphs/{layer}/{fov}_cell_type_neighbor_graph_layer={layer}_k=20.csv")
            try:
                roi_cells = ne.get_roi_cells(fov, layer=layer)
                if k == 20:
                    
                    if os.path.exists(p):
                        cell_type_neighbor_graph = pd.read_csv(p).rename({"Unnamed: 0": "cell_type"}, axis=1).set_index("cell_type")
                    else:
                        idx_neighbor_graph = ne.build_idx_neighbor_graph(fov, k=k)
                        cell_types = ne.get_cell_types(fov)
                        cell_type_neighbor_graph = ne.get_cell_type_neighbor_graph(idx_neighbor_graph, cell_types)
                        cell_type_neighbor_graph.to_csv(p)
                else:
                    cell_type_neighbor_graph = pd.read_csv(p).rename({"Unnamed: 0": "cell_type"}, axis=1).set_index("cell_type")
                count_df = ne.get_neighbor_counts(cell_type_neighbor_graph, fov, k=k)
                count_dfs.append(ne.get_neighbor_counts_with_roi(count_df, roi_cells=roi_cells))
            except Exception as e:
                print(e)
                print(fov)
        conc_count_df = pd.concat(count_dfs, axis=0)
        roi, rni, diff = ne.get_neighbour_count_difference(conc_count_df)
        ts = "_".join(tumor_stage)
        roi.to_csv(os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/z_scores/{layer}/{ts}_roi_k={k}.csv"))
        rni.to_csv(os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/z_scores/{layer}/{ts}_rni_k={k}.csv"))
        means, stds, z_scores = ne.get_zscores_for_neighbors(count_df, diff, 1000)
        
        z_scores.to_csv(os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/z_scores/{layer}/{ts}_z_scores_k={k}.csv"))
        means.to_csv(os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/z_scores/{layer}/{ts}_means_k={k}.csv"))
        stds.to_csv(os.path.join(base, f"je30bery/melanoma_data/MAGICAL/data/z_scores/{layer}/{ts}_stds_k={k}.csv"))