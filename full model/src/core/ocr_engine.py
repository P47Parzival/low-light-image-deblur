from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging

# Suppress PaddleOCR logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

class WagonOCR:
    def __init__(self):
        # use_angle_cls=True: Rotates text if the wagon is tilted (very common)
        # lang='en': We only need English numbers/letters
        # show_log=False: Keeps your terminal clean
        print("Initializing PaddleOCR...")
        # Force CPU environment variable is set at the top of the file
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv3')

    def process_wagon(self, wagon_crop_image):
        """
        Input: The cropped image of ONE wagon (from YOLO).
        Output: The detected Wagon Number string.
        """
        if wagon_crop_image is None or wagon_crop_image.size == 0:
            return None

        # --- INSERT DEBLURRING HERE LATER ---
        # wagon_crop_image = deblur_model(wagon_crop_image)
        # ------------------------------------

        # Run OCR
        # cls=True runs the angle classifier
        try:
            result = self.ocr.ocr(wagon_crop_image)
        except Exception as e:
            print(f"OCR Error: {e}")
            return None

        # Paddle returns a complex list. Let's parse the best result.
        detected_text = []
        if result and result[0]:
            for line in result[0]:
                text = None
                confidence = 1.0  # default if not provided

                if isinstance(line[1], (list, tuple)):
                    text = line[1][0]
                    if len(line[1]) > 1 and isinstance(line[1][1], (float, int)):
                        confidence = float(line[1][1])
                elif isinstance(line[1], str):
                    text = line[1]
                
                # Filter noise: Wagon numbers usually have 5+ digits
                # Adjusting logic based on Indian Wagon numbering (11 digits) or general robustness
                if confidence > 0.6 and len(text) > 4:
                    detected_text.append(text)

        # Return the most likely text (or join them if split across lines)
        return " ".join(detected_text)
