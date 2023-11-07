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
    def __init__(self, markers, data, mode="train", size=512):
        assert mode in ["train", "val"]
        self._means = {'Propidium': 0.08709432088084537,
 'phase': 0.20165798089522852,
 'CD95': 0.15458872610503402,
 'CD274': 0.011001287112713207,
 'Melan-A': 0.20414190063443236,
 'Bcl-2': 0.19218153937964813,
 'ADAM10-PE': 0.14058495656368772, 
 'CD63': 0.08018959974667886,
  'HLA-DR': 0.10238566351889483,
 'CD3': 0.07303562418427084,
 'CD8': 0.06087764743323087,}

        self._stds = {'Propidium': 0.14430829142838894,
 'phase': 0.17220733942502406,
 'CD95': 0.17349428432452133,
 'CD274': 0.07518154696891112,
 'Melan-A': 0.18290552133520185,
 'Bcl-2': 0.17599850706607423,
 'ADAM10-PE': 0.1505377234936453, 
 'CD63': 0.12011534021218785,
 'HLA-DR': 0.13826473686677376,
 'CD3': 0.1299362741733283,
 'CD8': 0.11157233764201473,}

        
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
            #AddGaussianNoiseToRandomChannel(mean=0, std=0.1)
            ])


    def __len__(self):
        '''
        Return number of samples
        '''
        return len(self._data)
    
    
    def _get_channel(self, sample, channel):
        try:
            file = [m for m in os.listdir(sample) if channel in m and not os.path.isdir(os.path.join(sample, m))][0]
            img = cv2.imread(os.path.join(sample, file), cv2.IMREAD_GRAYSCALE)
            img = img.astype("float64") / 255.
        except:
            img = np.full((2018, 2018), self._means[channel])
            # raise Exception(f"{sample} lacks {channel}")
        return img
        
    
    def __getitem__(self, index):
        '''
        Given sample index, return the augmented patch
        '''
        sample = self._data[index]
        
        label = np.array([0., 0.])
        label[os.path.basename(self._data[index]).startswith("Nevi")] = 1. # Melanoma: [1, 0], Nevi: [0, 1]
        label = t.from_numpy(label)
         
        img_files = list()
        for ch in self._markers:
            img_files.append(self._get_channel(sample, ch))
        
        img_files = np.array(img_files)
        tens = t.from_numpy(img_files)
        tens = self._resize_and_normalize(tens)
            
        if self._mode == "train":
            tens = self._transforms(tens)

        tens = tens.float()
        return tens, label

