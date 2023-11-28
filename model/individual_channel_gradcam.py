import torch
from torchvision import models, transforms
import cv2
import numpy as np
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook to get the gradients of the target layer
        self.hook = self.register_hooks()

    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = self.model.features[self.target_layer]
        hook = target_layer.register_backward_hook(backward_hook)
        return hook

    def remove_hooks(self):
        self.hook.remove()

    def forward(self, x):
        return self.model(x)

    def backward(self, output):
        self.model.zero_grad()
        output.backward(gradient=torch.ones_like(output), retain_graph=True)


    def generate_heatmap(self, activations, gradients):
        # Global average pooling
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        #heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
        heatmap = weights * activations
        # ReLU to keep only positive contributions
        heatmap = F.relu(heatmap)
        return heatmap

    def cam(self, input_image, save_path=None):
        # Forward pass
        output = self.forward(input_image)

        # Backward pass
        self.backward(output)

        # Get the target layer activations
        activations = self.model.get_activation()

        # Generate heatmap
        heatmap = self.generate_heatmap(activations, self.gradients)

        # Upsample the heatmap to the input size
        #heatmap = F.interpolate(heatmap, input_image.shape[2:], mode='bilinear', align_corners=False)

        # Normalize the heatmap
        heatmap = heatmap / torch.max(heatmap)
        return heatmap
        # Convert the heatmap to a numpy array
        #heatmap = heatmap.detach().numpy()[0, 0, :, :]

        # Resize input image to heatmap size
        #input_image_resized = cv2.resize(input_image[0].numpy().transpose(1, 2, 0), (heatmap.shape[1], heatmap.shape[0]))
        #return heatmap, input_image_resized
        # Superimpose heatmap on the input image
        #heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        