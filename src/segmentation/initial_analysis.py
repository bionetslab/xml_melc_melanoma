import os
import pickle
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import sys

sys.path.append("..")
from src import *
from tqdm import tqdm

to_delete_if_redundant = {
    "ADAM10": ["ADAM10-FITC"],
    "CD138": ["CD138-FITC",],
    "CD14": ["CD14-FITC"],
    "CD4": ["CD4-PE"],
    "CD83": ["CD83-PE", "CD83-AF488"]}

class ExpressionAnalyzer:
    def __init__(self, data_path, segmentation_results_dir_path, radii_ratio=1.3, membrane_markers=None, save_plots=False, markers_of_interest=None):
        """
        Initialize the ExpressionAnalyzer class.

        Args:
            data_path (str): The path to the data directory.
            segmentation_results_dir_path (str): The path to the segmentation results directory.
            membrane_markers (list, optional): A list of membrane marker names.
            save_plots (bool, optional): Whether to save generated plots.

        """
        self.data_path = data_path
        self.seg = MELC_Segmentation(data_path, membrane_markers=membrane_markers)
        self.radii_ratio = radii_ratio
        self.expression_data = None
        self.segmentation_results_dir = segmentation_results_dir_path
        self.save_plots = save_plots
        self.markers = dict()
        self.markers_of_interest = markers_of_interest

                 

    def run(self, segment="nucleus", profile=None):
        """
        Run the analysis for expression data.

        Args:
            segment (str, optional): The segment type to analyze (e.g., "nuclei").
            profile (dict, optional): A dictionary defining the expression profile.
        """
        self.segment_all()
        self.get_expression_of_all_samples(segment)
        if profile is not None:
            self.binarize_and_normalize_expression()
            plot_df = self.count_condition_cells(profile)
            self.plot_condition_df(plot_df, self.title_from_dict(profile), segment)
        
        
    def segment_all(self):
        """
        Segment nuclei and cells for all fields of view.

        """
        for fov in tqdm(self.seg.fields_of_view, desc="Segmenting"):
            nuclei_path = os.path.join(self.segmentation_results_dir, f"{fov}_nucleus.npy")
            if not os.path.exists(nuclei_path):
                try:
                    self.seg.field_of_view = fov
                    nuc, mem, _, _ = self.seg.run(fov, self.radii_ratio)
                    np.save(nuclei_path, nuc.astype(int))
                    np.save(os.path.join(self.segmentation_results_dir, f"{fov}_cell.npy"), mem.astype(int))
                except Exception as e:
                    print(fov, e)
                    continue
                
                
            nuclei_pickle_path = os.path.join(self.segmentation_results_dir, f"{fov}_nucleus.pickle")
            if not os.path.exists(nuclei_pickle_path):
                with open(nuclei_pickle_path, 'wb') as handle:
                    pickle.dump(self.seg.nucleus_label_where[fov], handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(self.segmentation_results_dir, f"{fov}_cell.pickle"), 'wb') as handle:
                    pickle.dump(self.seg.membrane_label_where[fov], handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    def get_expression_per_marker_and_sample(self, adaptive, where_dict):
        """
        Calculate expression for markers and samples.

        Args:
            adaptive (numpy.ndarray): The adaptive thresholded image.
            where_dict (dict): Dictionary mapping labels to coordinates.

        Returns:
            dict: Dictionary of expression values.

        """
        expression_dict = dict()
        for n in where_dict:
            if n == 0:
                continue

            segment = where_dict[n]
            exp = np.sum(adaptive[segment[0], segment[1]]) / len(segment[0])
            expression_dict[n] = exp / 255
        return expression_dict

    
    def get_expression_of_all_samples(self, segment):
        """
        Retrieve expression data for all samples.

        Args:
            segment (str): The segment type to analyze (e.g., "nuclei").

        """
        result_dfs = dict()
        for fov in tqdm(self.seg.fields_of_view, desc="Calculating expression"):
            os.makedirs(os.path.join(self.segmentation_results_dir, f"marker_expression_{segment}_results/"), exist_ok=True)
            expression_result_path = os.path.join(self.segmentation_results_dir, f"marker_expression_{segment}_results/{fov}.pkl")
            segmentation_result_path = os.path.join(self.segmentation_results_dir, f"{fov}_{segment}.pickle")
            
            self.seg.field_of_view = fov

            markers = {
                m.split("_")[0]: os.path.join(self.seg.get_fov_dir(), m)
                for m in sorted(os.listdir(self.seg.get_fov_dir()))
                if m.endswith(".tif") and "phase" not in m
            }
                        
            keys = list(markers.keys()).copy()
            cols = list()
            
            try:
                del markers['Propidium iodide']          
                del markers['PBS']  
                del markers['TcR alpha']
            except:
                pass
            for m in keys:
                col = "-".join(m.split("-")[:-1])
                
                if self.markers_of_interest:
                    if col not in self.markers_of_interest:
                        try:
                            del markers[m]
                        except:
                            pass
                    else:
                        cols.append(col)
                else:
                    cols.append(col)
                    
            cols, counts = np.unique(cols, return_counts=True)
            for redundant_col in cols[np.where(counts > 1)]:
                if len(redundant_col) > 0:
                    assert redundant_col in list(to_delete_if_redundant.keys()), f"{redundant_col} was not in previously known redundant columns {cols[np.where(counts > 1)]}"
                    for m in to_delete_if_redundant[redundant_col]:
                        del markers[m]
            
            cols = np.array([c for c in cols if len(c) > 0])
            self.markers[fov] = markers
            
            if not os.path.exists(expression_result_path):
                try:
                    with open(segmentation_result_path, 'rb') as handle:
                        where_dict = pickle.load(handle)
                except:
                    print(fov, "did not have segmentation file")
                    continue
                
                rows = list(where_dict.keys())
                df = pd.DataFrame(index=rows, columns=markers)
                
                for m in markers:                    
                    m_img = cv2.imread(markers[m], cv2.IMREAD_GRAYSCALE)
                    tile_std = np.std(m_img)
                    adaptive = cv2.adaptiveThreshold(m_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, -tile_std)

                    expression_dict = self.get_expression_per_marker_and_sample(adaptive, where_dict)
                    assert list(where_dict.keys()) == list(expression_dict.keys())
                    df[m] = list(expression_dict.values())
                
                df.to_pickle(expression_result_path)

            else:
                df = pd.read_pickle(expression_result_path)
                    
            
            cols = ["-".join(m.split("-")[:-1]) for m in markers.keys()]
            df.columns = cols            
            df["Field of View"] = fov
            df["Sample"] = "_".join(fov.split("_")[0:2])
            df["Group"] = fov.split("_")[0]
            df["Index"] = df.index
            v, c = np.unique(df.columns, return_counts=True)
            result_dfs[fov] = df
            
        self.expression_data = pd.concat(result_dfs)



    