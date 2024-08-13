import sys
import os
import torch as t
import numpy as np
from tqdm import tqdm
import cv2
from .individual_channel_gradcam import GradCAM

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_binary(smoothgrad, seg_file, output_size=512, cutoff=0.9):
    """
    Generate a binary mask based on SmoothGrad and segmentation file.

    Args:
    - smoothgrad (numpy.ndarray): SmoothGrad heatmap.
    - seg_file (numpy.ndarray): Segmentation file.
    - output_size (int, optional): Output size of the binary mask. Defaults to 512.
    - cutoff (float, optional): Cutoff quantile for thresholding. Defaults to 0.9.

    Returns:
    - tuple: Tuple containing the binary mask and the unique segments identified in the segmentation file.

    """
    smoothgrad = smoothgrad / (np.max(smoothgrad) + 1e-5) * 255
    pos = smoothgrad[np.where(smoothgrad > 0)]
    
    if len(pos) > 0:
        thresh = np.quantile(pos, cutoff)
        bin = (smoothgrad > thresh)
    else:
        bin = np.zeros_like(seg_file)
    
    resized = cv2.resize(bin.astype(np.uint8), seg_file.shape, interpolation=cv2.INTER_AREA)
    
    roi_segments = np.unique((resized * seg_file))
    roi_seg_file = (np.where(np.isin(seg_file, roi_segments), seg_file, 0) > 0) * 255
    roi_seg_file = cv2.resize(roi_seg_file.astype(np.uint8), (output_size, output_size))
    return roi_seg_file, roi_segments


def get_smooth_grad(dataloader, model, n_smooth=1, cuda=False, noise_std=1):
    """
    Generate a binary mask based on SmoothGrad and segmentation file.

    Args:
    - smoothgrad (numpy.ndarray): SmoothGrad heatmap.
    - seg_file (numpy.ndarray): Segmentation file.
    - output_size (int, optional): Output size of the binary mask. Defaults to 512.
    - cutoff (float, optional): Cutoff value for thresholding. Defaults to 0.9.

    Returns:
    - tuple: Tuple containing the binary mask and the unique segments identified in the segmentation file.

    """
    #dl = t.utils.data.DataLoader(MelanomaData(markers, classify=False, data=data, mode="val"), batch_size=1, shuffle=False)
    if cuda:
        model = model.cuda()
    model.eval()    
    gradcam = list()
    
    GradC = GradCAM(model=model)

    if dataloader:
        it = iter(dataloader)

    d = 0
    while True:
        try:
            if dataloader:
                input_t, label, _ = next(it)
            else:
                if d == n_smooth:
                    break
                input_t = t.zeros([1, 51, 512, 512])
                label = (-1)^d
                
        except StopIteration:
            break
        d += 1
        #input_t.requires_grad = True
        if cuda:
            input_t = input_t.cuda()
        gc = GradC.cam(input_t, n_smooth=n_smooth, label=label, noise_std=noise_std)  
        gc = np.mean(gc, axis=(0,1))
        gradcam.append(gc)
    
    gradcam = np.stack(gradcam, axis=2)
    if d > 1:
        mean = np.mean(gradcam, axis=2)
        gradcam = gradcam - np.stack([mean] * d, axis=2)
    gradcam *= gradcam > 0
    return gradcam


