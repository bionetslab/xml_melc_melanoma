import numpy as np
import pandas as pd
import os
import random
import pickle

def get_expression_anndata(anndata_file_path):
    x = pickle.load(open(anndata_file_path, 'rb'))
    return x

def get_expression_matrix_as_df(anndata, antibody_gene_symbols):
    df = pd.DataFrame()
    raw_df = pd.DataFrame(anndata.X, columns=anndata.var["gene_symbol"])
    for c in raw_df.columns:
        if c in ["CD45RA", "CD45RO", "PPB", 'CD66abce']:
            continue
        symbol = antibody_gene_symbols[c]
        if isinstance(symbol, list):
            for s in symbol:
                df[s] = raw_df[c]
        else:
            df[symbol] = raw_df[c]
    return df


def get_roi_cells(roi_path, fov):
    with open(os.path.join(roi_path, fov + "_idxs.pkl"), "rb") as fp:   
        roi_segments = pickle.load(fp)
    return roi_segments


def get_cell_types(cell_types_path, fov):
    with open(os.path.join(cell_types_path, fov + "_cell_types.pkl"), "rb") as fp:   
        cell_types = pickle.load(fp)
    return cell_types


def get_blocks(segmentation_shape=(2018, 2018), grid_shape=(16, 16)):
    nx, ny = grid_shape 
    bx, by = segmentation_shape[0] // grid_shape[0], segmentation_shape[1] // grid_shape[1]
    block_labels = np.arange(nx * ny).reshape(nx, ny)
    expanded_blocks = np.repeat(np.repeat(block_labels, bx, axis=0), by, axis=1)
    
    padding = (segmentation_shape[0] - expanded_blocks.shape[0]) // 2, (segmentation_shape[1] - expanded_blocks.shape[1]) // 2
    blocks = np.pad(expanded_blocks, padding, mode="reflect") 
    return blocks


def assign_blocks(segmentation, blocks):
    zeros = np.zeros((int(np.max(segmentation) + 1), len(np.unique(blocks))))
    block_df = pd.DataFrame(zeros, columns=np.unique(blocks))
    for b in np.unique(blocks):
        cells = np.unique(segmentation[np.where(blocks == b)])
        block_df.loc[cells, b] = 1
    block_df.drop(0, axis=0, inplace=True)
    block_df = block_df[block_df.sum(axis=1) != 0]
    return block_df.reset_index()
            

def shuffle_mapping(mapping):
    ks = np.array(list(mapping.keys()))
    vs = list(mapping.values())
    random.shuffle(vs)    
    return ks[np.where(np.array(vs) == 1)]
