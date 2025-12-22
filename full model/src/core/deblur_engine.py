import torch
import numpy as np
import cv2
import os
from src.core.nafnet_arch import NAFNet

class DeblurGANEngine:
    def __init__(self, weights_path):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] NAFNet: Running on {self.device}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"NAFNet weights not found at: {weights_path}")
            
        print(f"[INFO] NAFNet: Loading weights from {weights_path}...")
        
        # Determine config from filename (simple heuristic)
        # NAFNet-GoPro-width32.pth -> width=32
        width = 32
        if "width64" in weights_path:
            width = 64
            
        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]
        
        # Initialize Model
        self.model = NAFNet(img_channel=3, width=width, middle_blk_num=middle_blk_num, 
                          enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        
        # Load Weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle 'params' key if present (common in basicsr checkpoints)
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] NAFNet: Model loaded successfully.")

    def deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Takes a BGR image (numpy), deblurs it, and returns BGR image.
        """
        if self.model is None:
            return image
        
        h, w = image.shape[:2]
        
        # Preprocess: BGR -> RGB, Normalize 0 to 1, Batch Dimension, Tensor
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output_tensor = self.model(img_tensor)
        
        # Postprocess: Tensor -> Numpy, 0..1 -> 0..255, RGB -> BGR
        output = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output_bgr
