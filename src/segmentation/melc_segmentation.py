import numpy as np 
import cv2
from tqdm import tqdm 

from stardist.models import StarDist2D
import os
import cv2
import time
import numpy as np
from scipy.spatial import KDTree
from csbdeep.utils import normalize
import os

class MELC_Segmentation:
    def __init__(self, data_path, membrane_markers) -> None:
        """
        Initialize the MELC_Segmentation class.

        Args:
            data_path (str): The path to the data directory.
            membrane_markers (list or str): A list of membrane marker names or a single membrane marker name.

        """
        if membrane_markers is not None:
            if not isinstance(membrane_markers, list):
                membrane_markers = [membrane_markers]
                
        self.fields_of_view = [f for f in sorted(os.listdir(data_path)) if ("sample" in f)]
        
        self._field_of_view = None
        self._membrane_markers = membrane_markers
        self._model = None
        self._kdforest = dict()        
        self._data_path = data_path
        self._NN_dict = dict()
        
        self.nucleus_label_where = dict()
        self.membrane_label_where = dict()
        
   
    def run(self, field_of_view=None, radii_ratio=None):
        """
        Run the segmentation process.

        Args:
            field_of_view (str, optional): The field of view to process. If None, the current field of view is used.
            radii_ratio (float, optional): The radius ratio hyperparameter for estimating membranes.

        Returns:
            tuple: A tuple containing:
                - nuclei_labels (numpy.ndarray): Labels for nuclei segmentation.
                - combined_membranes (numpy.ndarray): Combined membrane labels.
                - nucleus_label_where (dict): Dictionary mapping nucleus labels to coordinates.
                - membrane_label_where (dict): Dictionary mapping membrane labels to coordinates.

        """
            
        # set field of view and assert it has been set:
        if field_of_view is not None:
            self.field_of_view = field_of_view
        else:
            print("Please specify field of view!")
        
        # get nuclei segmentation:
        prop_iodide = self.get_marker("bleach")
        if prop_iodide is None:
            print(field_of_view, "does not have Propidium Iodide channel")
            return
        nuclei_labels, nuclei_centers = self.segment(prop_iodide)
        
        membrane_marker_available = False
        if self._membrane_markers is not None:
            membrane_marker_available = True
            combined_membrane_labels = None
            
            # iteratively collect segments if there are more than one markers
            for marker in self._membrane_markers:
                membrane_marker = self.get_marker(marker)
                if membrane_marker is None:
                    membrane_marker_available = False
                    continue
                else:
                    membrane_marker_available = True
                membrane_labels, _ = self.segment(membrane_marker)
                if combined_membrane_labels is None:
                    combined_membrane_labels = membrane_labels
                else:
                    combined_membrane_labels = np.where(combined_membrane_labels == 0, membrane_labels, combined_membrane_labels)
            if membrane_marker_available:
                # reconstruct and assign the same labels as for nuclei
                reconstructed_membranes, nuclei_centers_without_membranes, new_radii_ratio, nucleus_radii_to_circle = self.existing_membranes_as_nuclei_NN(combined_membrane_labels, nuclei_labels, nuclei_centers)
                if new_radii_ratio is not None:
                    radii_ratio = new_radii_ratio
                    
        if (self._membrane_markers is None) or (not membrane_marker_available): 
            if radii_ratio is None:
                print("the desired radius needs to be specified as a hyperparameter.")
                return
            nuclei_centers_without_membranes = {i: point for i, point in enumerate(nuclei_centers)}
            nucleus_radii_to_circle = None
        
        # estimate missing membranes as circles and ensure they belong to nearest neighbor
        estimated_membranes = self.estimate_membranes_as_nuclei_NN_in_radius(nuclei_labels, nuclei_centers_without_membranes, radii_ratio, nucleus_radii_to_circle)
        
        # combine membranes
        if (self._membrane_markers is not None) and membrane_marker_available:
            combined_membranes = reconstructed_membranes + estimated_membranes
        else:
            combined_membranes = estimated_membranes
        return nuclei_labels, combined_membranes, self.nucleus_label_where, self.membrane_label_where

         
    
    @property
    def field_of_view(self):
        return self._field_of_view
    
    
    @field_of_view.setter
    def field_of_view(self, field_of_view):
        if isinstance(field_of_view, str) and field_of_view in self.fields_of_view:
            self._field_of_view = field_of_view    
        else:
            print(field_of_view, "is not valid")
        

    def get_marker(self, marker):
        fov_dir = self.get_fov_dir()
        if marker == "bleach":
            membrane_marker_path = self._get_channel_path(fov_dir, f"{marker}")
        else:
            membrane_marker_path = self._get_channel_path(fov_dir, f"{marker}-")

        if membrane_marker_path == -1:
            return None
        else:
            return cv2.imread(membrane_marker_path, cv2.IMREAD_GRAYSCALE)


    def get_fov_dir(self):
        fov_dir = os.path.join(self._data_path, self._field_of_view)
        assert os.path.isdir(fov_dir), f"Field of view {self._field_of_view} does not exist!"
        return fov_dir
    

#    def _get_bleach_dir(self, fov_dir):
#        bleach_dir = os.path.join(fov_dir, "bleach")
#        assert os.path.isdir(bleach_dir), f"Field of view {fov_dir} does not contain a bleach directory!"
#        return bleach_dir
    
    
    def _get_channel_path(self, img_dir, channel):
        channels = [c for c in os.listdir(img_dir) if channel.lower() in c.lower()]
        if len(channels) == 1:
            return os.path.join(img_dir, channels[0])
        else:
            print(f"Path for field of view {img_dir} does not contain exactly one image of the desired channel {channel}!")        
            return -1

    def segment(self, img):
        """
        Segment nuclei or membranes from an input image.

        Args:
            img (numpy.ndarray): The input grayscale image for segmentation.

        Returns:
            tuple: A tuple containing:
                - labels (numpy.ndarray): Labels for resulting segmentation.
                - cluster_centers (numpy.ndarray): Coordinates of the segment centers.

        """
        if self._model is None:
            self._model = StarDist2D.from_pretrained('2D_versatile_fluo')     
        axis_norm = (0, 1)
        norm_img = normalize(img, 1,99.8, axis=axis_norm)
        labels, details = self._model.predict_instances(norm_img)
        cluster_centers = details['points']
        return labels, cluster_centers
    
    
    def radius_from_max_dist_within_segment(self, segment):
        """
        Calculate the radius from the maximum distance within a segment.

        Args:
            segment (numpy.ndarray): Coordinates of points within the segment.

        Returns:
            float: The calculated radius.

        """
        x_min = np.min(segment[0])
        x_max = np.max(segment[0])
        x_diff = x_max - x_min
        y_min = np.min(segment[1])
        y_max = np.max(segment[1])
        y_diff = y_max - y_min
        diff = np.max([x_diff, y_diff])
        radius = diff / 2
        return radius
    
    
    def new_where_nucleus(self, nucleus, nucleus_label):
        if self.field_of_view in self.nucleus_label_where:
            self.nucleus_label_where[self.field_of_view].update({nucleus_label: np.array(nucleus)})
        else:
            self.nucleus_label_where[self.field_of_view] = {nucleus_label: np.array(nucleus)}
            
            
    def new_where_membrane(self, membrane, membrane_label):
        if self.field_of_view in self.membrane_label_where:
            self.membrane_label_where[self.field_of_view].update({membrane_label: np.array(membrane)})
        else:
            self.membrane_label_where[self.field_of_view] = {membrane_label: np.array(membrane)}


    def existing_membranes_as_nuclei_NN(self, membrane_labels, nuclei_labels, nuclei_centers):
        """
        Assign existing membranes to nuclei and ensure that they are assigned to their nearest neighbor nucleus.

        Args:
            membrane_labels (numpy.ndarray): Labels for membrane segmentation.
            nuclei_labels (numpy.ndarray): Labels for nuclei segmentation.
            nuclei_centers (dict): Dictionary of nucleus labels to their coordinates.

        Returns:
            tuple: A tuple containing:
                - reconstructed_membranes (numpy.ndarray): Membrane labels assigned to nuclei.
                - nuclei_centers_without_membrane (dict): Nucleus labels without assigned membranes.
                - radii_ratio (float): Median radius ratio.
                - nucleus_radii_to_circle (dict): Dictionary of nucleus labels to radius for circles.

        """
        if self.field_of_view not in self._kdforest:
            self._kdforest[self.field_of_view] = KDTree(nuclei_centers, leafsize=100)
        kdtree = self._kdforest[self.field_of_view]

        reconstructed_membranes = np.zeros_like(membrane_labels, dtype=int)
        nuclei_centers_without_membrane = dict()

        nucleus_radii = list()
        membrane_radii = list()
        nucleus_radii_to_circle = dict()
        for idx, point in tqdm(enumerate(nuclei_centers)):
            membrane_label = membrane_labels[int(point[0]), int(point[1])]
            nucleus_label = nuclei_labels[int(point[0]), int(point[1])]

            if membrane_label > 0: # there is an actual label
                # add the signal to the reconstruction 
                membrane = np.where(membrane_labels == membrane_label)
                self.new_where_membrane(membrane, membrane_label)
                segment = np.array(membrane).T
            else: 
                # there is no label
                # remember coordinates
                nuclei_centers_without_membrane[idx] = point
                nucleus = np.where(nuclei_labels == nucleus_label)
                self.new_where_nucleus(nucleus, nucleus_label)
                # remember nucleus radius
                nucleus_radii_to_circle[idx] = self.radius_from_max_dist_within_segment(nucleus)
                continue
            
            # assert that each signal belongs to nearest neighbor
            dist, idxs = kdtree.query(segment)
            segment = segment[np.where(idxs == idx)].T
                
            nucleus = np.where(nuclei_labels == nucleus_label)
            self.new_where_nucleus(nucleus, nucleus_label)
            
            # check if membrane is bigger than nucleus:
            if segment.shape[1] < len(nucleus[0]):
                # if not, discard membrane signal
                nuclei_centers_without_membrane[idx] = point
                nucleus_radii_to_circle[idx] = self.radius_from_max_dist_within_segment(nucleus)
                continue
            membrane_radii.append(self.radius_from_max_dist_within_segment(segment))
            nucleus_radii.append(self.radius_from_max_dist_within_segment(nucleus))

            reconstructed_membranes[segment[0], segment[1]] = nucleus_label
            self.new_where_membrane(segment, nucleus_label)

        if len(membrane_radii) > 0:
            ratio = np.array(membrane_radii) / np.array(nucleus_radii)
            radii_ratio = np.median(ratio)
        else: 
            radii_ratio = None
        return reconstructed_membranes, nuclei_centers_without_membrane, radii_ratio, nucleus_radii_to_circle

    

    def nearest_neighbors(self, nuclei_centers, nuclei_labels_shape):
        """
        Compute nearest neighbors for nuclei centers.

        Args:
            nuclei_centers (list): List of nucleus coordinates.
            nuclei_labels_shape (tuple): Shape of the nuclei labels array.

        Returns:
            numpy.ndarray: Nearest neighbor indices for each pixel.

        """
        if self.field_of_view not in self._NN_dict:
            kdtree = KDTree(nuclei_centers, leafsize=100)
            xaxis = np.linspace(0, nuclei_labels_shape[0] - 1, nuclei_labels_shape[0])
            yaxis = np.linspace(0, nuclei_labels_shape[1] - 1, nuclei_labels_shape[1])
            xv, yv = np.meshgrid(xaxis, yaxis)
            _, NN_idxs = kdtree.query(np.c_[xv.ravel(), yv.ravel()])
            NN_idxs = NN_idxs.reshape((nuclei_labels_shape[0], nuclei_labels_shape[1])).T 
            self._NN_dict[self.field_of_view] = NN_idxs
        return self._NN_dict[self.field_of_view]
    
    def in_circle(self, x, y, x_c, y_c, radius):
        return ((x - x_c) ** 2 + (y - y_c) ** 2) <= radius ** 2 
    
    def estimate_membranes_as_nuclei_NN_in_radius(self, nuclei_labels, nuclei_centers_without_membrane, radius, nucleus_radii_to_circle):
        """
        Estimate membranes from nuclei using nearest neighbors within a given radius.

        Args:
            nuclei_labels (numpy.ndarray): Labels for nuclei segmentation.
            nuclei_centers_without_membrane (dict): Dictionary of nucleus labels to coordinates.
            radius (float): Radius parameter for estimation.
            nucleus_radii_to_circle (dict): Dictionary of nucleus labels to radius for circles.

        Returns:
            numpy.ndarray: Membrane labels estimated from nuclei.

        """


        if self.field_of_view not in self._kdforest:
            
            nuclei_centers = list(nuclei_centers_without_membrane.values())
            if len(nuclei_centers) == 0:
                print(self.field_of_view, "could not be segmented")
            self._kdforest[self.field_of_view] = KDTree(nuclei_centers, leafsize=100)
        kdtree = self._kdforest[self.field_of_view]
        
        reconstructed_membranes = np.zeros_like(nuclei_labels, dtype=int)
        xaxis = np.linspace(0, nuclei_labels.shape[0] - 1, nuclei_labels.shape[0])
        yaxis = np.linspace(0, nuclei_labels.shape[1] - 1, nuclei_labels.shape[1])

        for idx in tqdm(nuclei_centers_without_membrane):
            point = nuclei_centers_without_membrane[idx]
            # get nucleus label
            nucleus_label = nuclei_labels[int(point[0]), int(point[1])]       

            if nucleus_radii_to_circle is not None:
                if idx in nucleus_radii_to_circle:
                        # if radius has already been calculated, use that value
                        nucleus_radius = nucleus_radii_to_circle[idx]
                else:
                    # else calculate radius
                    nucleus = np.where(nuclei_labels == nucleus_label)
                    self.new_where_nucleus(nucleus, nucleus_label)
                    nucleus_radius = self.radius_from_max_dist_within_segment(nucleus)
                
            else:
                # else calculate radius
                nucleus = np.where(nuclei_labels == nucleus_label)
                self.new_where_nucleus(nucleus, nucleus_label)
                nucleus_radius = self.radius_from_max_dist_within_segment(nucleus)
            
            # calculate membrane radius and draw circle
            absolut_radius = radius * nucleus_radius 
            segment = np.array(np.where(self.in_circle(xaxis[:,None], yaxis[None,:], int(point[0]), int(point[1]), np.ceil(absolut_radius) + 1))).T
            dist, idxs = kdtree.query(segment)
            segment = segment[np.where(idxs == idx)].T
            reconstructed_membranes[segment[0], segment[1]] = nucleus_label
            self.new_where_membrane(segment, nucleus_label)
        return reconstructed_membranes 