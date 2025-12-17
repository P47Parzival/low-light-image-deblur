import cv2
from ultralytics import YOLO
import sys
import os
import argparse
import multiprocessing as mp
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.ocr_engine import WagonOCR
from src.core.enhancer import LowLightEnhancer


# -----------------------------
# OCR Worker Process
# -----------------------------

def ocr_worker(input_queue, output_queue):
    """
    Runs OCR in a separate process.
    """
    ocr_engine = WagonOCR()

    while True:
        item = input_queue.get()
        if item is None:
            break

        wagon_id, crop = item
        text = ocr_engine.process_wagon(crop)

        if text:
            output_queue.put((wagon_id, text))


# -----------------------------
# Main Pipeline
# -----------------------------

def main_pipeline(video_path, weights_path):

    if not os.path.exists(video_path):
        print("Video not found")
        return

    print("[INFO] Loading YOLO + ByteTrack")
    model = YOLO(weights_path)

    enhancer = None
    # enhancer = LowLightEnhancer(weights_path='src/core/Epoch99.pth')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return

    # -----------------------------
    # OCR multiprocessing setup
    # -----------------------------
    ocr_input_queue = mp.Queue(maxsize=10)
    ocr_output_queue = mp.Queue()

    ocr_process = mp.Process(
        target=ocr_worker,
        args=(ocr_input_queue, ocr_output_queue),
        daemon=True
    )
    ocr_process.start()

    # -----------------------------
    # Runtime memory
    # -----------------------------
    wagon_text = {}      # track_id -> OCR text
    ocr_requested = set()

    input_size = 640
    frame_count = 0
    skip_frames = 3      # GPU later → 1 or 2

    print("[INFO] Starting pipeline (press Q to quit)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        if enhancer:
            frame = enhancer.enhance_frame(frame)

        resized = cv2.resize(frame, (input_size, input_size))

        # -----------------------------
        # YOLO + ByteTrack
        # -----------------------------
        if frame_count % skip_frames == 0:
            results = model.track(
                resized,
                persist=True,
                tracker="trackers/byte_track.yaml",
                verbose=False
            )
        else:
            results = model.predict(resized, verbose=False)

        # -----------------------------
        # Read OCR results (non-blocking)
        # -----------------------------
        while not ocr_output_queue.empty():
            tid, text = ocr_output_queue.get()
            wagon_text[tid] = text
            print(f"[OCR] Track {tid}: {text}")

        # -----------------------------
        # Draw results
        # -----------------------------
        for r in results:
            if r.boxes.id is None:
                continue

            for box, track_id in zip(r.boxes.xyxy, r.boxes.id):
                track_id = int(track_id)

                x1, y1, x2, y2 = box
                x1 = int(x1 * w / input_size)
                x2 = int(x2 * w / input_size)
                y1 = int(y1 * h / input_size)
                y2 = int(y2 * h / input_size)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Request OCR ONCE per track
                if (
                    track_id not in ocr_requested
                    and (x2 - x1) > 200
                    and frame_count % skip_frames == 0
                ):
                    crop = frame[y1:y2, x1:x2]
                    try:
                        ocr_input_queue.put_nowait((track_id, crop))
                        ocr_requested.add(track_id)
                    except:
                        pass

                label = f"ID {track_id}"
                if track_id in wagon_text:
                    label += f" | {wagon_text[track_id]}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Wagon Inspection AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -----------------------------
    # Cleanup
    # -----------------------------
    ocr_input_queue.put(None)
    ocr_process.join()

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument(
        "--weights_path",
        default="railway_hackathon_take2/merged_model_v1/weights/best.pt"
    )

    args = parser.parse_args()
    main_pipeline(args.video_path, args.weights_path)
