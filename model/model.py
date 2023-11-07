import torchvision as tv
import torch as t

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b7, EfficientNet_B7_Weights, efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict



class EfficientnetWithFinetuning(t.nn.Module):
    def __init__(self, indim):
        super(EfficientnetWithFinetuning, self).__init__()
        eff = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.features = eff.features
        if indim != 3:
            #self.features[0][0] = t.nn.Conv2d(indim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.features[0][0] = t.nn.Conv2d(indim, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.avgpool = eff.avgpool
        out_features = 2
        #self.classifier = t.nn.Sequential(t.nn.Dropout(p=0.2, inplace=True), t.nn.Linear(1280, out_features, bias=True), t.nn.Sigmoid())
        self.classifier = t.nn.Sequential(t.nn.Dropout(p=0.2, inplace=True), t.nn.Linear(1792, out_features, bias=True), t.nn.Sigmoid())
        self.flatten = t.nn.Flatten()


    def forward(self, x):
        #freeze the specified layers
        for l in self.features[1:]:
            l.requires_grad = False
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)    
        return x 
    


if __name__ == "__main__":
    m = EfficientnetWithFinetuning(5)
    print(m)
