import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --------------------------------------------------------------------------------
# SCI Architecture
# --------------------------------------------------------------------------------
class EnhanceNetwork(nn.Module):
    def __init__(self, layers=3, channels=3):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.in_conv(input)
        x = self.relu(x)
        x = self.conv(x)

        illumination = input + x
        illumination = torch.clamp(illumination, 1e-4, 1)
        
        output = input / illumination
        output = torch.clamp(output, 0, 1)

        return output

class LowLightEnhancer:
    def __init__(self, weights_path=None, device='cpu'):
        self.device = torch.device(device)
        self.model = EnhanceNetwork(layers=3, channels=3).to(self.device)
        
        if weights_path is None:
             base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../full model/SCI'))
             weights_path = os.path.join(base_dir, 'difficult.pt')

        if weights_path and os.path.exists(weights_path):
            print(f"[LowLightEnhancer] Loading SCI weights from {weights_path}")
            # Use strict=False to be safe against minor version mismatches
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
        else:
            print(f"[LowLightEnhancer] Warning: Weights file not found at {weights_path}")
        
        self.model.eval()

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return None

        # -----------------------------------------------
        # 1. PRE-PROCESS
        # -----------------------------------------------
        # Convert BGR (OpenCV) to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0.0 - 1.0
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Tensorize
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # -----------------------------------------------
        # 2. INFERENCE
        # -----------------------------------------------
        with torch.no_grad():
            enhanced_tensor = self.model(img_tensor)

        # -----------------------------------------------
        # 3. POST-PROCESS
        # -----------------------------------------------
        # Back to Numpy
        enhanced_img = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Clip to safe range before converting
        enhanced_img = np.clip(enhanced_img, 0, 1)
        
        # Convert back to uint8
        enhanced_img_uint8 = (enhanced_img * 255.0).astype(np.uint8)
        
        # Back to BGR
        final_bgr = cv2.cvtColor(enhanced_img_uint8, cv2.COLOR_RGB2BGR)

        # -----------------------------------------------
        # 4. MANUAL DARKENING (OpenCV Way)
        # -----------------------------------------------
        # Create a spatial mask for the center
        H, W = final_bgr.shape[:2]
        
        # CENTER MASK Generation
        # We want the center to be darker (e.g. multiplied by 0.6)
        # and edges to be normal (multiplied by 1.0)
        
        # Use an elliptical mask for better fit than pure radial
        mask = np.ones((H, W), dtype=np.float32)
        
        center_x, center_y = W // 2, H // 2
        # Radius of the "dark spot"
        radius_x, radius_y = W // 2, H // 2
        
        # Draw a gradient ellipse? 
        # Simpler: Create a radial gradient using distance transform
        # Create a black image with a white dot in center
        temp_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(temp_mask, (center_x, center_y), int(min(H, W) * 0.4), 255, -1) # 40% radius solid center
        
        # Blur strictly to create smooth gradient
        temp_mask = cv2.GaussianBlur(temp_mask, (301, 301), 0)
        
        # Normalize mask to 0-1
        alpha_mask = temp_mask.astype(np.float32) / 255.0
        
        # Where alpha is 1 (Center), we want factor 0.7
        # Where alpha is 0 (Edge), we want factor 1.0
        # Factor = 1.0 - (alpha * 0.3)
        darkening_factor = 1.0 - (alpha_mask * 0.3)
        
        # Expand for 3 channels
        darkening_factor_3ch = cv2.merge([darkening_factor, darkening_factor, darkening_factor])
        
        # Apply
        final_output = final_bgr.astype(np.float32) * darkening_factor_3ch
        
        return final_output.astype(np.uint8)