import torchvision as tv
import torch as t

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b7, EfficientNet_B7_Weights, efficientnet_b4, EfficientNet_B4_Weights
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

        # two classifiers to enable (unsupervised) clustering on 50-dim. latent space, could be replaced by one if latent space not required
        latent_features = 50 
        out_features = 1
        
        self.classifier1 = t.nn.Sequential(t.nn.Dropout(p=0.1, inplace=True), t.nn.Linear(1792, latent_features, bias=True))
        #self.classifier2 = t.nn.Sequential(t.nn.Linear(latent_features, out_features, bias=True), t.nn.Sigmoid())
        # for regression (no Sigmoid Layer):
        self.classifier2 = t.nn.Linear(latent_features, out_features, bias=True)

        self.flatten = t.nn.Flatten()


    def forward(self, x):
        # freeze all layers except the first one to keep the pre-trained parameters
        x_copy = x.clone()
        #for l in self.features[1:]:
        #    l.requires_grad = False
        if self.cam:
            for l in self.features:
                self.in_activation[l] = x_copy
                x_copy = l(x_copy)
                self.out_activation[l] = x_copy
                
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier1(x)     
        x = self.classifier2(x)    
        return x

    def get_activation(self, l, mode):
        # Retrieve the stored activation for the specified layer
        assert mode in ["in", "out"], "Do you want the input or output activations?"
        if mode == "in":
            return self.in_activation[l]
        return self.out_activation[l]



if __name__ == "__main__":
    m = EfficientnetWithFinetuning(5)
    print(m)
