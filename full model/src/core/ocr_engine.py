# from paddleocr import PaddleOCR
# import cv2
# import numpy as np
# import logging

# # Suppress PaddleOCR logs
# logging.getLogger("ppocr").setLevel(logging.ERROR)

# class WagonOCR:
#     def __init__(self):
#         # use_angle_cls=True: Rotates text if the wagon is tilted (very common)
#         # lang='en': We only need English numbers/letters
#         # show_log=False: Keeps your terminal clean
#         print("Initializing PaddleOCR...")
#         self.ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv3')

#     def process_wagon(self, wagon_crop_image):
#         """
#         Input: The cropped image of ONE wagon (from YOLO).
#         Output: The detected Wagon Number string.
#         """
#         if wagon_crop_image is None or wagon_crop_image.size == 0:
#             return None

#         # --- INSERT DEBLURRING HERE LATER ---
#         # wagon_crop_image = deblur_model(wagon_crop_image)
#         # ------------------------------------

#         # Run OCR
#         # cls=True runs the angle classifier
#         try:
#             result = self.ocr.ocr(wagon_crop_image)
#         except Exception as e:
#             print(f"OCR Error: {e}")
#             return None

#         # Paddle returns a complex list. Let's parse the best result.
#         detected_text = []
#         if result and result[0]:
#             for line in result[0]:
#                 text = None
#                 confidence = 1.0  # default if not provided

#                 if isinstance(line[1], (list, tuple)):
#                     text = line[1][0]
#                     if len(line[1]) > 1 and isinstance(line[1][1], (float, int)):
#                         confidence = float(line[1][1])
#                 elif isinstance(line[1], str):
#                     text = line[1]
                
#                 # Filter noise: Wagon numbers usually have 5+ digits
#                 # Relaxed: > 0.4 conf, > 0 chars (Show anything)
#                 print(f"      [Paddle Raw] Text: '{text}' | Conf: {confidence:.2f}")
#                 if confidence > 0.4 and len(text) >= 1:
#                     detected_text.append(text)

#         if not detected_text:
#              print("      [OCR] No valid text found after filtering.")
#              return None

#         # Return the most likely text (or join them if split across lines)
#         full_text = " ".join(detected_text)
#         print(f"      [Final Selection] {full_text}")
#         return full_text


# pip install easyocr

import easyocr
import torch

class WagonOCR:
    def __init__(self):
        print("Initializing EasyOCR...")
        self.reader = self._init_reader()

    def _init_reader(self):
        """Prefer GPU when available; fall back to CPU otherwise."""
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            try:
                return easyocr.Reader(['en'], gpu=True)
            except Exception as err:
                # Log and fall back if CUDA runtime is unavailable or misconfigured
                print(f"EasyOCR GPU init failed ({err}); switching to CPU.")
        else:
            print("EasyOCR GPU not detected; using CPU.")

        return easyocr.Reader(['en'], gpu=False)

    def process_wagon(self, wagon_crop_image):
        if wagon_crop_image is None or wagon_crop_image.size == 0:
            return None

        try:
            results = self.reader.readtext(wagon_crop_image)
        except Exception as e:
            print(f"OCR Error: {e}")
            return None

        detected_text = []
        for (bbox, text, confidence) in results:
            print(f"      [EasyOCR Raw] Text: '{text}' | Conf: {confidence:.2f}")
            if confidence > 0.3 and len(text) >= 1:
                detected_text.append(text)

        if not detected_text:
            print("      [OCR] No valid text found after filtering.")
            return None

        full_text = " ".join(detected_text)
        print(f"      [Final Selection] {full_text}")
        return full_text