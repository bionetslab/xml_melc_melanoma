from torchvision import models, transforms
import cv2
import numpy as np
import torch.nn.functional as F
import torch as t

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.cam = True
        self.gradients = None
        # Hook to get the gradients of the target layer
        self.hook = self.register_hooks()

    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        if self.model.__class__.__name__ == "ResNetWithFinetuning":
            target_layer = self.model.layer4[-1]
        elif self.model.__class__.__name__ == "EfficientnetWithFinetuning":
            target_layer = self.model.features[-1]
        else:
            target_layer = self.model.features[-1]

        
        hook = target_layer.register_full_backward_hook(backward_hook)
        return hook

    
    def remove_hooks(self):
        self.hook.remove()

    def forward(self, x):
        return self.model(x)

    def backward(self, output):
        self.model.zero_grad()
        output.backward(gradient=t.ones_like(output), retain_graph=True)


    def cam(self, input_image, n_smooth, label):
        heatmaps = list()
        for _ in range(n_smooth):
            if n_smooth > 1:
                noise = t.randn(input_image.size())
                output = self.forward(input_image.clone() + noise.to(input_image.device))
            else:                
                output = self.forward(input_image)
                
            self.backward(output)
            weights = self.gradients
            activations = self.model.get_last_conv_activation()

            heatmap = weights * activations

            if label == 0:
                heatmap *= -1
            
            heatmap = F.relu(heatmap).detach().cpu().numpy()
            assert np.min(heatmap) >= 0
            if n_smooth > 1:
                heatmaps.append(heatmap)
            else:
                heatmap = heatmap / (np.max(heatmap) + 1e-5) # for numerical stability
                return heatmap
        heatmaps = np.stack(heatmaps, axis=0)
        heatmap = np.mean(heatmaps, axis=0)
        heatmap /= (np.max(heatmap) + 1e-5) # for numerical stability
        return heatmap
     