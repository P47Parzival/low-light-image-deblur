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
        Includes Padding and Test-Time Augmentation (TTA) for better quality.
        """
        if self.model is None:
            return image
        
        h, w = image.shape[:2]
        
        # 1. Pad image to be a multiple of 32
        # This prevents edge artifacts and ensures the network architecture aligns correctly
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        img_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        def run_inference(img_in):
            # Preprocess: BGR -> RGB, Normalize 0 to 1, Batch Dimension, Tensor
            img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output_tensor = self.model(img_tensor)
            
            # Postprocess: Tensor -> Numpy (Float 0..1)
            output = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            return output

        # 2. Run inference (Original)
        out_1 = run_inference(img_padded)

        # 3. Run inference (Flipped - Test Time Augmentation)
        # This helps recover details by processing the image from a different orientation
        img_flipped = cv2.flip(img_padded, 1)
        out_2 = run_inference(img_flipped)
        out_2 = cv2.flip(out_2, 1)

        # 4. Average the results for higher quality
        final_output = (out_1 + out_2) / 2.0
        
        # 5. Convert back to BGR and uint8
        final_output = np.clip(final_output * 255.0, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)
        
        # 6. Crop padding to return original size
        return output_bgr[:h, :w, :]
