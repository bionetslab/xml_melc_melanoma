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
            
        target_layer = self.model.res.layer4[1].conv2
        hook = target_layer.register_full_backward_hook(backward_hook)
        return hook

    
    def remove_hooks(self):
        self.hook.remove()

    def forward(self, x):
        return self.model(x)

    def backward(self, output):
        self.model.zero_grad()
        output.backward(gradient=t.ones_like(output), retain_graph=True)


    def cam(self, input_image, n_smooth, label, noise_std=1):
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
                    noise = t.randn(input_image.size()) * noise_std
                    noise = noise.to(input_image.device)
                    noisy_image = input_image.clone() + noise
                    output = self.forward(noisy_image)
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
            heatmap = heatmap.detach().cpu().numpy() #= F.relu(heatmap).detach().cpu().numpy()
            if n_smooth > 1:
                heatmaps.append(heatmap)
            else:
                heatmap = heatmap / (np.max(heatmap) + 1e-5) # for numerical stability
                return heatmap
            
        heatmaps = np.stack(heatmaps, axis=0) # stack list
        heatmap = np.mean(heatmaps, axis=0) # calculate mean over "deep" dimension
        heatmap /= (np.max(heatmap) + 1e-5) # for numerical stability
        return heatmap
     