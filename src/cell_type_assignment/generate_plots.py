
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import glob
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from multiprocessing import Pool
import time
import math
from collections import Counter
import scanpy as sc
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import random
from scipy.stats import mannwhitneyu
import os
import warnings
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import gseapy as gp
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import mygene
import seaborn as sns
from gseapy import barplot, dotplot
import random
import matplotlib.pyplot as plt
import numpy as np
import decoupler as dc
from numpy.random import default_rng
from copy import deepcopy
import os
from skimage import io
from skimage.util import img_as_ubyte
import imageio as iio
from skimage import img_as_ubyte
import re
from matplotlib.pyplot import figure
import os
from sklearn.mixture import GaussianMixture
import statistics
from bigtree import Node
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from pathlib import Path

def plot_cell_assignment_tree(split_tuples):
    clustering_tree={}
    for i in split_tuples:
        clustering_tree[i.__dict__['split_var']]=i.__dict__['cell_type']
    # --------------------
    plotting_tree={}
    last_key='HPA-based celltype annotation (tree):'
    # plotting_tree_list=[last_key]
    plotting_tree_list=[]
    # ---
    plotting_tree_count=-1
    big_tree_count=0
    exec('Node'+str(big_tree_count)+'='+'Node("' + last_key + '")')
    for i in clustering_tree:
        plotting_tree_count+=1
        # if plotting_tree_count==0:
        #     asdf
        # final_plotting_tree_list=[]
        plot_tree_dict_path='plotting_tree'
        if plotting_tree_count>0:
            for j in plotting_tree_list:
                plot_tree_dict_path=plot_tree_dict_path+'[' + '"' + j + '"' + ']'
                # final_plotting_tree_list.append(j)
        else:
            pass
        # ---
        last_key=i
        last_key_pos=i+'+'
        last_key_neg=i+'-'
        # ---
        exec(plot_tree_dict_path+ '=' + '{}'  )
        exec(plot_tree_dict_path+'[' + '"' + last_key_pos + '"' +']'+'='+ '"' + clustering_tree[i] + '"'  )
        big_tree_count+=1
        exec('Node'+str(big_tree_count)+'='+'Node("' + last_key_pos + ':' + clustering_tree[i] + '"' + ',' +  'parent=' + 'Node'+str(big_tree_count-1)  + ')')
        # ---
        plotting_tree_list.append(last_key_neg)
        big_tree_count+=1
        exec('Node'+str(big_tree_count)+'='+'Node("' + last_key_neg + '"' + ',' +  'parent=' + 'Node'+str(big_tree_count-2)  + ')')
    # ---
    exec('Node0.show()')
    # ---
    print('\nHPA-based celltype annotation (dict):')
    print(plotting_tree)
    # --------------------
    return clustering_tree, plotting_tree



def plot_En_heatmap(cols):
    root_dir = Path(__file__).resolve().parent
    _path_=root_dir / 'hpa_data' / 'cell_type_nTPM_max_norm.csv'
    expr_per_cell_type_max_norm=pd.read_csv(_path_, sep=',')
    expr_per_cell_type_max_norm=expr_per_cell_type_max_norm.rename(columns={'Unnamed: 0': "Celltypes"})
    expr_per_cell_type_max_norm=expr_per_cell_type_max_norm.set_index('Celltypes')
    plot_heatmap(expr_per_cell_type_max_norm, cols, outfile='heatmap.pdf', title='Gene-wise max-normalized nTPM per cell type')
    return expr_per_cell_type_max_norm


def plot_heatmap(data, genes, figsize=(9,6), outfile=None, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data=data[genes],ax=ax)
    ax.set(ylabel='')
    if not title is None:
        ax.set_title(title)
    fig.tight_layout()
    if not outfile is None:
        fig.savefig(outfile)
    return('Heatmap plotted!')