from torchvision import models, transforms
import cv2
import numpy as np
import torch.nn.functional as F
import torch as t

class GradCAM:
    def __init__(self, model, target_layer, mode="out"):
        model.cam = True
        model.in_activation = dict()
        model.out_activation = dict()
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.mode = mode

        # Hook to get the gradients of the target layer
        self.hook = self.register_hooks()

    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            if self.mode == "in":
                self.gradients = grad_input[0]
            else:
                self.gradients = grad_output[0]
        target_layer = self.model.features[self.target_layer]
        hook = target_layer.register_full_backward_hook(backward_hook)
        return hook

    def remove_hooks(self):
        self.hook.remove()

    def forward(self, x):
        return self.model(x)

    def backward(self, output):
        self.model.zero_grad()
        output.backward(gradient=t.ones_like(output), retain_graph=True)


    def cam(self, input_image, n_smooth=5):
        heatmaps = list()
        for _ in range(n_smooth):
            if n_smooth > 1:
                noise = t.randn(input_image.size())
                output = self.forward(input_image.clone() + noise.cuda())
            else:                
                output = self.forward(input_image)
                
            self.backward(output)
            weights = self.gradients
            activations = self.model.get_activation(self.model.features[self.target_layer], self.mode)
            heatmap = weights * activations
            heatmap = F.relu(heatmap).detach().cpu().numpy()
            assert np.min(heatmap) >= 0
            heatmap = heatmap / np.max(heatmap)
            if n_smooth > 1:
                heatmaps.append(heatmap)
            else:
                return heatmap
        heatmaps = np.stack(heatmaps, axis=0)
        heatmap = np.mean(heatmaps, axis=0)
        heatmap /= np.max(heatmap)
        return heatmap
        # Convert the heatmap to a numpy array
        #heatmap = heatmap.detach().numpy()[0, 0, :, :]

        # Resize input image to heatmap size
        #input_image_resized = cv2.resize(input_image[0].numpy().transpose(1, 2, 0), (heatmap.shape[1], heatmap.shape[0]))
        #return heatmap, input_image_resized
        # Superimpose heatmap on the input image
        #heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        