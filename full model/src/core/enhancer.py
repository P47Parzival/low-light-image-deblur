import torch
import numpy as np
import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.zero_dce import enhance_net_nopool

class LowLightEnhancer:
    def __init__(self, weights_path=None, device='cpu'):
        self.device = torch.device(device)
        # Initialize the model with scale_factor=1 as per typical usage or default
        self.model = enhance_net_nopool(scale_factor=1).to(self.device)
        
        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print(f"Warning: Weights file not found at {weights_path}. Using random weights.")
        
        self.model.eval()

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhances a single frame using Zero-DCE.
        Args:
            frame: Input image (numpy array, BGR).
        Returns:
            Enhanced image (numpy array, BGR).
        """
        # Preprocess
        img = frame.astype(np.float32) / 255.0
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1) # HWC to CHW
        img = img.unsqueeze(0) # Add batch dimension
        img = img.to(self.device)

        # Inference
        with torch.no_grad():
            # The new model returns (enhance_image, x_r)
            enhanced_img, _ = self.model(img)

        # Postprocess
        enhanced_img = enhanced_img.squeeze(0).permute(1, 2, 0) # CHW to HWC
        enhanced_img = enhanced_img.cpu().numpy()
        enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced_img
