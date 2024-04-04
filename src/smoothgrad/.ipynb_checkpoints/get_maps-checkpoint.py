import sys
import os
import torch as t
import numpy as np
from tqdm import tqdm
import cv2
#base = "/data/bionets" if "ramses" in os.uname()[1] else "/data_nfs/"
#sys.path.append(os.path.join(base, "je30bery/melanoma_data/MAGICAL/model/"))
#from model import EfficientnetWithFinetuning, VGGWithFinetuning, ResNetWithFinetuning
#sys.path.append("../../MAGICAL/smoothgrad")
from .individual_channel_gradcam import GradCAM

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_binary(smoothgrad, seg_file, output_size=512, cutoff=0.9):
    smoothgrad = smoothgrad / (np.max(smoothgrad) + 1e-5) * 255
    resized = cv2.resize(smoothgrad.astype(np.uint8), seg_file.shape)

    pos = resized[np.where(resized > 0)]
    thresh = np.quantile(pos, cutoff)
    resized = (resized > thresh)
    
    roi_segments = np.unique((resized * seg_file))
    roi_seg_file = (np.where(np.isin(seg_file, roi_segments), seg_file, 0) > 0) * 255
    roi_seg_file = cv2.resize(roi_seg_file.astype(np.uint8), (output_size, output_size))
    return roi_seg_file / 255., roi_segments


def get_smooth_grad(dataloader, model, n_smooth=10, cuda=False):
    #dl = t.utils.data.DataLoader(MelanomaData(markers, classify=False, data=data, mode="val"), batch_size=1, shuffle=False)
    if cuda:
        model = model.cuda()
    model.eval()    
    gradcam = list()
    
    GradC = GradCAM(model=model)
    
    it = iter(dataloader)

    d = 0
    while True:
        try:
            input_t, label = next(it)
        except StopIteration:
            break
        d += 1
        #input_t.requires_grad = True
        if cuda:
            input_t = input_t.cuda()
        gc = GradC.cam(input_t, n_smooth=n_smooth, label=label)  
        gc = np.mean(gc, axis=(0,1))
        gradcam.append(gc)
    
    gradcam = np.stack(gradcam, axis=2)
    if d > 1:
        mean = np.mean(gradcam, axis=2)
        gradcam = gradcam - np.stack([mean] * d, axis=2)
    gradcam *= gradcam > 0
    return gradcam


