import torch as t
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights, resnet50, ResNet50_Weights, vgg16, VGG16_Weights, resnet18
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


# this avoids some error in the torchvision version I am using
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

#https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent


class ResNet18_smoothgrad(t.nn.Module):    
    def __init__(self, 
                 indim, 
                 cam=False, 
                 checkpoint_path="/data/bionets/je30bery/melanoma_data/pre_trained_weights/tenpercent_resnet18.ckpt"):       
        
        super(ResNet18_smoothgrad, self).__init__()
        self._cam = cam
        self.activation = None
        self.res = resnet18(weights=None)
        
        checkpoint = t.load(checkpoint_path, map_location="cpu")
        new_state_dict = {}

        for key, value in checkpoint['state_dict'].items():
            new_key = key.replace("model.resnet.", "")
            new_state_dict[new_key] = value
        self.res.load_state_dict(new_state_dict, strict=False)
        
        if indim != 3:
            self.res.conv1 = t.nn.Conv2d(indim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)        
        self.res.fc = t.nn.Sequential(t.nn.Linear(in_features=512, out_features=512, bias=True),
                                      t.nn.ReLU(inplace=False),
                                      t.nn.Dropout(p=0.2, inplace=False),
                                      t.nn.Linear(in_features=512, out_features=512, bias=True),
                                      t.nn.ReLU(inplace=False),
                                      t.nn.Dropout(p=0.2, inplace=False),
                                      t.nn.Linear(in_features=512, out_features=1, bias=True),
                                      t.nn.Sigmoid())
        self.res.flatten = t.nn.Flatten()
        # Xavier initialization
        for layer in self.res.fc:
            if isinstance(layer, t.nn.Linear):
                t.nn.init.xavier_uniform_(layer.weight)
                t.nn.init.constant_(layer.bias, 0.1) 
                

    def forward(self, x, epoch=-1):
        x = self.res.conv1(x)
        x = self.res.bn1(x)
        x = self.res.relu(x)
        x = self.res.maxpool(x)
        x = self.res.layer1(x)
        x = self.res.layer2(x)
        x = self.res.layer3(x)
        x = self.res.layer4(x)
        if self._cam:
            self.activation = x.clone()
        x = self.res.avgpool(x)
        x = self.res.flatten(x)
        x = self.res.fc(x)
        return x


    def get_last_conv_activation(self):
        return self.activation


class ResNet18_pretrained_for_finetuning(t.nn.Module):    
    def __init__(self, indim, cam=False, checkpoint_path="/data/bionets/je30bery/melanoma_data/pre_trained_weights/tenpercent_resnet18.ckpt"):        
        super(ResNet18_pretrained_for_finetuning, self).__init__()

        # re-use layers of efficientnet_b4 (good ratio of #parameters and performance on ImageNet)
        self.res = resnet18(weights=None)
        checkpoint = t.load(checkpoint_path, map_location="cpu")
        new_state_dict = {}

        for key, value in checkpoint['state_dict'].items():
            new_key = key.replace("model.resnet.", "")
            new_state_dict[new_key] = value

        # Load the modified state_dict into the model
        self.res.load_state_dict(new_state_dict, strict=False)

        if indim != 3:
            self.res.conv1 = t.nn.Conv2d(indim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # binary classifier
        
        self.res.fc = t.nn.Sequential(t.nn.Linear(in_features=512, out_features=512, bias=True),
                                      t.nn.ReLU(inplace=False),
                                      t.nn.Dropout(p=0.2, inplace=False),
                                      t.nn.Linear(in_features=512, out_features=512, bias=True),
                                      t.nn.ReLU(inplace=False),
                                      t.nn.Dropout(p=0.2, inplace=False),
                                      t.nn.Linear(in_features=512, out_features=1, bias=True),
                                      t.nn.Sigmoid())
        
        # Xavier initialization
        for layer in self.res.fc:
            if isinstance(layer, t.nn.Linear):
                t.nn.init.xavier_uniform_(layer.weight)
                t.nn.init.constant_(layer.bias, 0.1) 
                

    def forward(self, x, epoch=-1):
        for p in self.res.conv1.parameters(): 
            p.requires_grad = False
        for p in self.res.bn1.parameters(): 
            p.requires_grad = False
        for p in self.res.layer1.parameters(): 
            p.requires_grad = False
        for p in self.res.layer2.parameters(): 
            p.requires_grad = False
        for p in self.res.layer3.parameters(): 
            p.requires_grad = False
        for p in self.res.layer4.parameters(): 
            p.requires_grad = False
        x = self.res(x)      
        return x

    def get_last_conv_activation(self):
        return self.activation



class ResNet18_pretrained(t.nn.Module):    
    def __init__(self, indim, cam=False, checkpoint_path="/data/bionets/je30bery/melanoma_data/pre_trained_weights/tenpercent_resnet18.ckpt"):        
        super(ResNet18_pretrained, self).__init__()

        self._cam = cam
        if self._cam:
            self.activation = 0
        # re-use layers of efficientnet_b4 (good ratio of #parameters and performance on ImageNet)
        self.res = resnet18(weights=None)
        #print(res)
        checkpoint = t.load(checkpoint_path, map_location="cpu")
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            new_key = key.replace("model.resnet.", "")
            new_state_dict[new_key] = value

        # Load the modified state_dict into the model
        self.res.load_state_dict(new_state_dict, strict=False)

        if indim != 3:
            self.res.conv1 = t.nn.Conv2d(indim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # binary classifier
        self.res.flatten = t.nn.Flatten()

        self.res.fc = t.nn.Sequential(t.nn.Dropout(p=0.1, inplace=True), t.nn.Linear(512, 1, bias=True), t.nn.Sigmoid())
        
        # Xavier initialization
        for layer in self.res.fc:
            if isinstance(layer, t.nn.Linear):
                t.nn.init.xavier_uniform_(layer.weight)
                t.nn.init.constant_(layer.bias, 0.1) 
                

    def forward(self, x, epoch=-1):
        if epoch < 15:
            for p in self.res.layer1.parameters(): 
                p.requires_grad = False
            for p in self.res.layer2.parameters(): 
                p.requires_grad = False
            for p in self.res.layer3.parameters(): 
                p.requires_grad = False
            for p in self.res.layer4.parameters(): 
                p.requires_grad = False
        x = self.res(x)  
        return x    

    def get_last_conv_activation(self):
        return self.activation
