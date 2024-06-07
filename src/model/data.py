from torch.utils.data import Dataset
import numpy as np
import torchvision as tv
import numpy as np
import cv2
import random
from scipy.ndimage import rotate
import os
import torch as t
import matplotlib.pyplot as plt
import json


class AddGaussianNoiseToRandomChannel(object):
    """
    Class for custom image augmentation
    Args:
        mean (float): mean of random noise.
        std (float): standard deviation of random noise.
    """
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """
        Args:
            tensor (torch.tensor), MELC image to which the noise should be added     
        Returns:
            tensor (torch.tensor), MELC image with additive Gaussian noise
        """
        c = int(random.random() * tensor.shape[0])
        rand = t.randn(tensor[c].size()) *  self.std + self.mean
        tensor[c] +=  rand
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        

class MelanomaData(Dataset):
    """
    Dataset class for loading melanoma MELC data.

    Args:
        markers (list): List of marker names.
        pretrain (bool): True for Melanoma vs Nevi, False for Coarse tumor stage as label
        data (DataFrame): DataFrame containing the dataset, as for example produced by src/data_utils/get_data_csv()
        mode (str, optional): The mode of operation, one of "train", "val", or "segment". Defaults to "train".
        in "train" mode, the data is augmented, whereas it is only resized and normalized in "val" mode. "segment" mode just returns the original images.
        size (int, optional): Size of the resized images. Defaults to 512.

    Attributes:
        _pretrain (bool): Flag indicating the task.
        _config (dict): The global config file
        _means (dict): Dictionary containing means of marker values.
        _stds (dict): Dictionary containing standard deviations of marker values.
        _data (DataFrame): DataFrame containing the dataset.
        _mode (str): The mode of operation.
        _markers (list): List of marker names.
        _resize_and_normalize (torchvision.transforms.Compose): Composed torchvision transforms for resizing and normalization.
        _transforms (torchvision.transforms.Compose): Composed torchvision transforms for data augmentation.
    """
    def __init__(self, markers, pretrain, data, mode="train", size=512, config_path="/data_nfs/je30bery/melanoma_data/config.json"):
        assert mode in ["train", "val", "segment"]
        self._pretrain = pretrain
        
        with open(config_path, 'r') as f:
            self._config = json.load(f)
            
        with open(os.path.join(self._config["dataset_statistics"], 'melanoma_means.json'), 'r') as fp:
            self._means = json.load(fp)
        with open(os.path.join(self._config["dataset_statistics"], 'melanoma_stds.json'), 'r') as fp:
            self._stds = json.load(fp)
        
        self._size = size
        self._data = data
        self._mode = mode
        self._markers = markers
        self._resize_and_normalize = tv.transforms.Compose([
            tv.transforms.Resize((size, size), interpolation=tv.transforms.InterpolationMode.BILINEAR, antialias=True),
            tv.transforms.Normalize(mean=[self._means[m] for m in markers], std=[self._stds[m] for m in markers])
            ])
                                                     
        self._transforms = tv.transforms.Compose([
            tv.transforms.RandomRotation(degrees=30, interpolation=tv.transforms.InterpolationMode.NEAREST, expand=False),
            tv.transforms.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), antialias=True),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.RandomVerticalFlip(p=0.5),
            AddGaussianNoiseToRandomChannel(mean=0, std=0.1)
            ])


    def __len__(self):
        '''
        Return number of samples
        '''
        return len(self._data)
    
    
    def _get_channel(self, sample, channel):
        """
        Returns the image data for a specific channel.

        Args:
            sample (str): Path to the sample directory.
            channel (str): Name of the channel.

        Returns:
            numpy.ndarray: Image data for the specified channel.
        """
        try:
            try:
                file = [os.path.join(sample, m) for m in os.listdir(sample) if channel in m and not os.path.isdir(os.path.join(sample, m))][0]
            except TypeError:
                file = [os.path.join(sample, m.decode('utf-8')) for m in os.listdir(sample) if channel in m.decode('utf-8') and not os.path.isdir(os.path.join(sample, m.decode('utf-8')))][0]
                #print(file, "was byte encoded")
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = img.astype("float64") / 255.
        except IndexError:
                img = np.full((2018, 2018), self._means[channel])
        return img
        
    
    def __getitem__(self, index):
        """
        Retrieves the augmented patch for a given sample index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: A tuple containing the tensor image data and its label in train and val mode. In segment mode, only the image is returned.
        """
        sample = os.path.join(self._config["melanoma_data"], self._data.iloc[index]["file_path"])

        if self._pretrain:
            label = float(self._data.iloc[index]["Group"] == "Melanoma")
            label = np.array(float(label))
        else:
            label = float(self._data.iloc[index]["PFS label"])
            label = np.array(float(label))
  
        label = t.from_numpy(label).float().unsqueeze(0)
        img_files = list()
        for ch in self._markers:
            img_files.append(self._get_channel(sample, ch))
        
        img_files = np.array(img_files)
        if self._mode == "segment":
            return img_files
            
        tens = t.from_numpy(img_files)
        tens = self._resize_and_normalize(tens)
            
        if self._mode == "train":
            tens = self._transforms(tens)

        tens = tens.float()
        return tens, label
        

