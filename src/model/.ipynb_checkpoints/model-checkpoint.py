import torchvision as tv
import torch as t
import numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b7, EfficientNet_B7_Weights, efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights


resnet50(weights=ResNet50_Weights.DEFAULT)
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


# this avoids some error in the torchvision version I am using
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict



class EfficientnetWithFinetuning(t.nn.Module):
    def __init__(self, indim, cam=False):        
        super(EfficientnetWithFinetuning, self).__init__()

        self.cam = cam
        if self.cam:
            self.in_activation = dict()
            self.out_activation = dict()

        # re-use layers of efficientnet_b4 (good ratio of #parameters and performance on ImageNet)
        eff = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1) 

        self.features = eff.features

        # replace first layer to suit the input dimensions
        if indim != 3:
            self.features[0][0] = t.nn.Conv2d(indim, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.avgpool = eff.avgpool
        self.flatten = t.nn.Flatten()
        self.classifier = t.nn.Sequential(t.nn.Dropout(p=0.1, inplace=True), t.nn.Linear(1792, 1, bias=True), t.nn.Sigmoid())
        
        for layer in self.classifier:
            if isinstance(layer, t.nn.Linear):
                t.nn.init.xavier_uniform_(layer.weight)
                t.nn.init.constant_(layer.bias, 0.1)  # You can adjust the bias initialization as needed




    def forward(self, x):
        #for p in self.features[1:7].parameters(): 
        #    p.requires_grad = False
 
        x = self.features(x)
        if self.cam:
            self.activation = x.clone()
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)         
        return x

    def get_last_conv_activation(self):
        # Retrieve the stored activation for the specified layer
        return self.activation
    


class EfficientnetWithFinetuningWithVGGClassifier(t.nn.Module):
    def __init__(self, indim, cam=False):        
        super(EfficientnetWithFinetuningWithVGGClassifier, self).__init__()

        self.cam = cam
        if self.cam:
            self.in_activation = dict()
            self.out_activation = dict()

        # re-use layers of efficientnet_b4 (good ratio of #parameters and performance on ImageNet)
        eff = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1) 

        self.features = eff.features

        # replace first layer to suit the input dimensions
        if indim != 3:
            self.features[0][0] = t.nn.Conv2d(indim, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.avgpool = eff.avgpool
        self.flatten = t.nn.Flatten()
        
        self.vggclassifier = t.nn.Sequential(
                t.nn.Linear(in_features=1792, out_features=1792, bias=True),
                t.nn.ReLU(inplace=False),
                t.nn.Dropout(p=0.5, inplace=False),
                t.nn.Linear(in_features=1792, out_features=1792, bias=True),
                t.nn.ReLU(inplace=False),
                t.nn.Dropout(p=0.5, inplace=False),
                t.nn.Linear(in_features=1792, out_features=1, bias=True),
                t.nn.Sigmoid()
            )
        
        for layer in self.vggclassifier:
            if isinstance(layer, t.nn.Linear):
                t.nn.init.xavier_uniform_(layer.weight)
                t.nn.init.constant_(layer.bias, 0.1)  # You can adjust the bias initialization as needed




    def forward(self, x):
        for p in self.features.parameters(recurse=True): 
            p.requires_grad = False
 
        x = self.features(x)
        if self.cam:
            self.activation = x.clone()
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.vggclassifier(x)         
        return x

    def get_last_conv_activation(self):
        # Retrieve the stored activation for the specified layer
        return self.activation



class ResNetWithFinetuning(t.nn.Module):
    def __init__(self, indim, cam=False):        
        super(ResNetWithFinetuning, self).__init__()
        self.cam = cam
        # re-use layers of resnet as baseline 
        res = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if indim != 3:
            self.conv1 = t.nn.Conv2d(indim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.bn1 = res.bn1 
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        
        self.avgpool = res.avgpool
        self.flatten = t.nn.Flatten()
        self.classifier = t.nn.Sequential(t.nn.Dropout(p=0.1, inplace=True), t.nn.Linear(2048, 1, bias=True), t.nn.Sigmoid())



    def forward(self, x):
        # freeze all layers except the first one to keep the pre-trained parameters
        for l in [self.layer2, self.layer3]:
            for s in l:
                s.parameters().requires_grad = False
        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.relu(x) 
        x  = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.cam:
            self.activation = x.clone()
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)         
        return x

    def get_last_conv_activation(self):
        # Retrieve the stored activation for the specified layer
        return self.activation

    

class VGGWithFinetuning(t.nn.Module):
    def __init__(self, indim, cam=False):        
        super(VGGWithFinetuning, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg.features
        self.cam = cam
        if indim != 3:
            self.features[0] = t.nn.Conv2d(indim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features[23] = t.nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False) # improve granularity of last layer for feature importance extraction

        
        self.avgpool = vgg.avgpool
        self.flatten = t.nn.Flatten()
        self.classifier = t.nn.Sequential(
                t.nn.Linear(in_features=25088, out_features=4096, bias=True),
                t.nn.ReLU(inplace=False),
                t.nn.Dropout(p=0.5, inplace=False),
                t.nn.Linear(in_features=4096, out_features=4096, bias=True),
                t.nn.ReLU(inplace=False),
                t.nn.Dropout(p=0.5, inplace=False),
                t.nn.Linear(in_features=4096, out_features=1, bias=True),
                t.nn.Sigmoid()
            )


    def forward(self, x):
        # freeze all layers except the first one to keep the pre-trained parameters
        #for l in self.features[10:22]:
        #    l.requires_grad = False
        x = self.features(x)
        if self.cam:
            self.activation = x.clone()
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)         
        return x
    
    
    def get_last_conv_activation(self):
        # Retrieve the stored activation for the specified layer
        return self.activation

