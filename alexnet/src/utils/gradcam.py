import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping for AlexNet.
    Generates class activation heatmaps overlaying input images.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()
        
        # If target layer is not specified, find the last Conv2d in model.features
        if target_layer is None:
            for module in reversed(list(model.features.children())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
                    
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for a given input tensor and target class.
        
        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_class: Target class index. If None, uses class with highest score.
            
        Returns:
            Normalized 2D numpy array heatmap in range [0, 1] with shape (H, W).
        """
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Compute weights: spatial average of gradients
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = torch.sum(weights * activations, dim=0)  # (H, W)

        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize heatmap to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to match input image resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (w, h))
        return heatmap

    @staticmethod
    def overlay_heatmap(original_pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """
        Overlays 2D heatmap onto PIL image.
        """
        img_np = np.array(original_pil_image.convert("RGB"))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlayed = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
        return Image.fromarray(overlayed)
