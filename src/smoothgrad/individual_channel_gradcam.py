from torchvision import models, transforms
import cv2
import numpy as np
import torch.nn.functional as F
import torch as t

class GradCAM:
    """
    Class for computing GradCAM/SmoothGrad visualizations.

    Args:
        model (torch.nn.Module): The PyTorch model for which GradCAM will be computed.

    Attributes:
        model (torch.nn.Module): The PyTorch model for which GradCAM will be computed.
        gradients (torch.Tensor): Gradients of the target layer with respect to the output.
        hook (handle): Handle for the registered hook.
    """

    def __init__(self, model):
        self.model = model
        self.model.cam = True
        self.gradients = None
        self.hook = self.register_hooks()

    def register_hooks(self):
        """
        Register backward hook on the target layer.

        Returns:
            handle: Handle for the registered hook.
        """
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
        # backward output, but set gradient to ones to ensure that the gradients computed are uniformly spread across the activations of the target layer
        # technically, the value of output (the model prediction) does not really matter in this case
        # needed, so that smoothgrad operates correctly within the computational graph and propagates gradients through the network in the expected manner
        output.backward(gradient=t.ones_like(output), retain_graph=True)


    def cam(self, input_image, n_smooth, label):
        """
        Compute GradCAM (n_smooth=1) or Smoothgrad (n_smooth>1) for the input image.

        Args:
            input_image (torch.Tensor): Input image tensor.
            n_smooth (int): Number of smooth gradcams to compute.
            label (int): Label index for the GradCAM computation.

        Returns:
            numpy.ndarray: GradCAM heatmap for the input image.
        """
        heatmaps = list()
        for _ in range(n_smooth): # generate n noisy images 
            if n_smooth > 1:
                noise = t.randn(input_image.size())
                output = self.forward(input_image.clone() + noise.to(input_image.device))
            else:                
                output = self.forward(input_image)
                
            self.backward(output)
            # get activations and weights of the last conv layer
            weights = self.gradients
            activations = self.model.get_last_conv_activation()

            # multiply
            heatmap = weights * activations

            # negate gradients for counter factual example
            if label == 0:
                heatmap *= -1
            heatmap = F.relu(heatmap).detach().cpu().numpy()

            if n_smooth > 1:
                heatmaps.append(heatmap)
            else:
                heatmap = heatmap / (np.max(heatmap) + 1e-5) # for numerical stability
                return heatmap
            
        heatmaps = np.stack(heatmaps, axis=0) # stack list
        heatmap = np.mean(heatmaps, axis=0) # calculate mean over "deep" dimension
        heatmap /= (np.max(heatmap) + 1e-5) # for numerical stability
        return heatmap
     