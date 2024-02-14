import numpy as np
import pandas as pd
import anndata as ad
import cv2 
import sys
import os
import scipy.spatial as spatial
from tqdm import tqdm
from scipy.stats import norm
import pickle

class NeighborEnricher:
    def __init__(self, cell_coordinates, base_path) -> None:
        self.roi_imgs = dict()
        self.rni_imgs = dict()
        self.cell_coordinates = cell_coordinates
        self.base_path = base_path
        self.count_dfs = dict()

    
    def build_idx_neighbor_graph(self, fov, k):
        coordinates = self.cell_coordinates[fov]
        centers = dict()
        for id in coordinates.keys():
            where = coordinates[id]
            centers[id] = np.mean(where, axis=1)
        
        cols = [f"{i}th neighbor" for i in range(k)]
        idx_neighbor_graph = pd.DataFrame(index=coordinates.keys(), columns=cols)
        
        point_tree = spatial.cKDTree(np.array(list(centers.values())))
        
        for id in coordinates.keys():
            neighbor_idxs = point_tree.query(x=centers[id], k=k, distance_upper_bound=50, workers=-1)
            neighbor = neighbor_idxs[1]
            idx_neighbor_graph.loc[id, cols] = neighbor
        return idx_neighbor_graph


    def get_roi_cells(self, fov, layer="2", model="classification_model"):
        roi_path = os.path.join(self.base_path, f"je30bery/melanoma_data/MAGICAL/data/ROIs/{model}", layer)
        with open(os.path.join(roi_path, fov + "_idxs.pkl"), "rb") as fp:   
            roi_segments = pickle.load(fp)
        return roi_segments


    def get_cell_types(self, fov):
        roi_path = os.path.join(self.base_path, "je30bery/melanoma_data/MAGICAL/data/cell_types/")
        with open(os.path.join(roi_path, fov + "_cell_types.pkl"), "rb") as fp:   
            cell_types = pickle.load(fp)
        return cell_types


    def get_cell_type_neighbor_graph(self, idx_neighbor_graph, cell_types):
        cell_types.append(np.nan)
        vf = np.vectorize(lambda x: cell_types[x])
        cell_type_neighbor_graph = pd.DataFrame(vf(idx_neighbor_graph.values), index=idx_neighbor_graph.index, columns=idx_neighbor_graph.columns)
        return cell_type_neighbor_graph
        

    def get_neighbor_counts(self, cell_type_neighbor_graph, fov, k=20):

        if fov in self.count_dfs.keys():
            count_df = self.count_dfs[fov]
        
        count_df = pd.DataFrame(columns=list(np.unique(cell_type_neighbor_graph["0th neighbor"])) + ["nan"], index=cell_type_neighbor_graph.index)
        
        cols = [f"{i}th neighbor" for i in range(1, k)]
        for i in tqdm(range(len(cell_type_neighbor_graph)), leave=False):
            v, c = np.unique(cell_type_neighbor_graph.iloc[i][cols].astype(str), return_counts=True)
            count_df.iloc[i].loc[v] = c
        count_df["cell_type"] = cell_type_neighbor_graph["0th neighbor"]
        count_df.set_index("cell_type", inplace=True)
        
        self.count_dfs[fov] = count_df
        return count_df


    def get_neighbor_counts_with_roi(self, count_df, roi_cells=None, p=-1):
        if p != -1:
            count_df["ROI"] = np.random.choice([1, 0], len(count_df), p=[p, 1-p])
        elif roi_cells is not None:
            roi_info = np.zeros(len(count_df))
            assert roi_cells[0] == 0
            roi_info[roi_cells[1:] - 1] = 1
            count_df["ROI"] = roi_info
        else:
            raise ValueError("either provide p or roi_cells")
        return count_df

    
    def get_neighbour_count_difference(self, count_df):
        roi_counts = pd.DataFrame(columns=list(np.unique(count_df.index.values)) + ["nan"])
        rni_counts = pd.DataFrame(columns=list(np.unique(count_df.index.values)) + ["nan"])
        
        for center_cell in np.unique(count_df.index):
            for i in [0, 1]: # i = 0 -> not in ROI, i = 1 -> in ROI
                if center_cell in count_df[count_df["ROI"] == i].index.values:
                    try:
                        cell_count_df = count_df[count_df["ROI"] == i].loc[center_cell].drop(["ROI"], axis=1)
                        counts = cell_count_df.sum(axis=0) / len(cell_count_df)
                        
                    except ValueError:
                        # if there is only one cell of type center cell, cell_count_df becomes a Series instead of a dataframe and the axis parameter needs to be removed to drop columns, the length needs to be calculated differently
                        counts = count_df[count_df["ROI"] == i].loc[center_cell].drop(["ROI"]) #ToDO
                    
                    if i == 0:
                        rni_counts.loc[center_cell] = counts
                    else:
                        roi_counts.loc[center_cell] = counts
        roi_counts.replace({np.nan:0}, inplace=True)
        rni_counts.replace({np.nan:0}, inplace=True)
        true_diff = roi_counts - rni_counts
        return roi_counts, rni_counts, true_diff
  
    
    def get_zscores_for_neighbors(self, count_df, true_diff, n):
        p = np.sum(count_df["ROI"] == 1) / len(count_df)
        diffs = np.zeros((n, true_diff.shape[0], true_diff.shape[1]))
        for i in tqdm(range(n), leave=False):
            count_df = self.get_neighbor_counts_with_roi(count_df, roi_cells=None, p=p)
            _, _, rand_diff = self.get_neighbour_count_difference(count_df)
            diffs[i] = rand_diff
        means = pd.DataFrame(np.nanmean(diffs, axis=0), index=true_diff.index, columns=true_diff.columns)
        stds = pd.DataFrame(np.nanstd(diffs, axis=0), index=true_diff.index, columns=true_diff.columns)
        z_scores = (true_diff - means) / stds
        return means, stds, z_scores
        
    
    
    
    
    
    
    
        
        