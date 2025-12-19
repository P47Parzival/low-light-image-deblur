import cv2
from ultralytics import YOLO
import sys
import os
import argparse
import multiprocessing as mp
import time
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.indian_railways import IndianWagonParser
from src.scripts.pipeline_viz import draw_stats, draw_track
from src.core.enhancer import LowLightEnhancer

# -----------------------------
# OCR Worker
# -----------------------------
def ocr_worker(input_queue, output_queue):
    # Force CPU for this process ONLY, before importing Paddle
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Local import to prevent main process from seeing this Env Var change or loading Paddle
    from src.core.ocr_engine import WagonOCR
    
    ocr_engine = WagonOCR()
    while True:
        try:
            item = input_queue.get()
            if item is None: break
            wagon_id, crop, req_time = item
            
            # Raw text
            raw_text = ocr_engine.process_wagon(crop)
            
            # Try Parsing
            parsed_data = None
            if raw_text:
                parsed_data = IndianWagonParser.parse(raw_text)
                
            if raw_text:
                output_queue.put((wagon_id, raw_text, parsed_data, req_time))
        except Exception as e:
            print(f"OCR WORKER ERROR: {e}")

# -----------------------------
# Main Loop
# -----------------------------
def main_pipeline(video_path, weights_path):
    if not os.path.exists(video_path): return
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)

    # Multiprocessing
    ocr_in_q = mp.Queue(maxsize=10)
    ocr_out_q = mp.Queue()
    ocr_p = mp.Process(target=ocr_worker, args=(ocr_in_q, ocr_out_q), daemon=True)
    ocr_p.start()

    # State
    wagon_data = {} # id -> {raw, parsed}
    ocr_requested = set()
    
    # Profiler
    metrics = {
        'fps': deque(maxlen=50),
        'det': deque(maxlen=50),
        'ocr': deque(maxlen=50)
    }
    prev_time = time.time()
    frame_cnt = 0

    print("[INFO] Pipeline Started. Press 'Q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_cnt += 1
        h, w = frame.shape[:2]

        # 1. Detection
        t0 = time.time()
        results = model.track(frame, persist=True, tracker="trackers/byte_track.yaml", verbose=False)
        metrics['det'].append((time.time()-t0)*1000)

        # 2. Check OCR Results
        while not ocr_out_q.empty():
            tid, raw, parsed, req_t = ocr_out_q.get()
            metrics['ocr'].append((time.time()-req_t)*1000)
            
            wagon_data[tid] = {'raw': raw, 'parsed': parsed}
            
            if parsed:
                print(f"[MATCH] ID {tid}: {parsed['formatted']} ({parsed['type']})")

        # 3. Process Tracks
        active_tracks = 0
        if results and results[0].boxes.id is not None:
            active_tracks = len(results[0].boxes.id)
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Request OCR condition
                if track_id not in ocr_requested and (x2-x1) > 150 and frame_cnt % 3 == 0:
                     crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                     try: ocr_in_q.put_nowait((track_id, crop, time.time()))
                     except: pass
                     ocr_requested.add(track_id)

                # Info Text Construction
                info = None
                if track_id in wagon_data:
                    d = wagon_data[track_id]
                    if d['parsed']:
                        p = d['parsed']
                        info = f"{p['formatted']}\nType: {p['type']}\nRly: {p['railway']}\nYr: {p['year']}"
                    else:
                        info = f"Raw: {d['raw']}"

                draw_track(frame, (x1,y1,x2,y2), track_id, info)

        # 4. Stats & Display
        curr_time = time.time()
        metrics['fps'].append(1 / (curr_time - prev_time) if curr_time > prev_time else 0)
        prev_time = curr_time
        
        stats = [
            f"FPS: {sum(metrics['fps'])/len(metrics['fps']):.1f}" if metrics['fps'] else "FPS: 0",
            f"Detection: {sum(metrics['det'])/len(metrics['det']):.0f} ms" if metrics['det'] else "Det: 0",
            f"OCR Latency: {sum(metrics['ocr'])/len(metrics['ocr']):.0f} ms" if metrics['ocr'] else "OCR: 0",
            f"Active Tracks: {active_tracks}"
        ]
        draw_stats(frame, stats)

        cv2.imshow("Wagon Pipeline", frame)
        if cv2.waitKey(1) == ord('q'): break

    # Cleanup
    ocr_in_q.put(None)
    ocr_p.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--weights_path", default="railway_hackathon_take3/merged_model_v2/weights/best.pt")
    args = parser.parse_args()
    main_pipeline(args.video_path, args.weights_path)
