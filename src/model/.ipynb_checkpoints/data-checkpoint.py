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
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        c = int(random.random() * tensor.shape[0])
        rand = t.randn(tensor[c].size()) *  self.std + self.mean
        tensor[c] +=  rand
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        

class MelanomaData(Dataset):
    def __init__(self, markers, classify, data, mode="train", size=512):
        assert mode in ["train", "val", "segment"]
        self._classify = classify
        with open(f'/data_nfs/je30bery/melanoma_data/MAGICAL/data/dataset_statistics/melanoma_means.json', 'r') as fp:
            self._means = json.load(fp)
        with open(f'/data_nfs/je30bery/melanoma_data/MAGICAL/data/dataset_statistics/melanoma_stds.json', 'r') as fp:
            self._stds = json.load(fp)
        
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
            #tv.transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.01, 0.5)),
            AddGaussianNoiseToRandomChannel(mean=0, std=0.1)
            ])


    def __len__(self):
        '''
        Return number of samples
        '''
        return len(self._data)
    
    
    def _get_channel(self, sample, channel):
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
        '''
        Given sample index, return the augmented patch
        '''
        sample = os.path.join(f"/data_nfs/datasets/melc/melanoma/processed", self._data.iloc[index]["file_path"])

        if self._classify:
            label = float(self._data.iloc[index]["Group"] == "Melanoma")
            label = np.array(float(label))
        else:
            label = float(self._data.iloc[index]["Coarse tumor stage"])
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
        

