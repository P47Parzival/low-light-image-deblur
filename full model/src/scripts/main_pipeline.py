import cv2
from ultralytics import YOLO
import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.ocr_engine import WagonOCR
from src.core.enhancer import LowLightEnhancer

def main_pipeline(video_path, weights_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # 1. Load Models
    print(f"Loading YOLO model from {weights_path}...")
    try:
        yolo_model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    try:
        ocr_engine = WagonOCR()
    except Exception as e:
        print(f"Error loading OCR engine: {e}")
        return

    # Check for enhancer weights (optional integration)
    enhancer = None
    # Uncomment to enable enhancer if weights exist
    # enhancer = LowLightEnhancer(weights_path='src/core/Epoch99.pth') 

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Starting pipeline. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break

        # --- STEP 2: LOW LIGHT ENHANCEMENT ---
        if enhancer:
            frame = enhancer.enhance_frame(frame)

        # --- STEP 3: WAGON DETECTION ---
        results = yolo_model(frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure coordinates are within frame bounds
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Crop the wagon
                wagon_crop = frame[y1:y2, x1:x2]

                # --- STEP 4 & 5: OCR ---
                # Only run OCR if the wagon is large enough (closer to camera)
                # This saves processing power!
                if (x2 - x1) > 200: 
                    wagon_number = ocr_engine.process_wagon(wagon_crop)
                    
                    if wagon_number:
                        # Draw the number on the main frame
                        cv2.putText(frame, f"ID: {wagon_number}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        print(f"Detected ID: {wagon_number}")

                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Wagon Inspection AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Wagon Inspection Pipeline.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video.")
    parser.add_argument("--weights_path", type=str, default="railway_hackathon/wagon_counter_v1/weights/best.pt", help="Path to YOLO weights.")
    
    args = parser.parse_args()
    
    main_pipeline(args.video_path, args.weights_path)
