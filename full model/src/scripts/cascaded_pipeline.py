import cv2
from ultralytics import YOLO
import sys
import os
import argparse
import multiprocessing as mp
import time
from collections import deque
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.ocr_engine import WagonOCR
from src.core.indian_railways import IndianWagonParser
from src.scripts.pipeline_viz import draw_stats, draw_track

# -----------------------------
# OCR Processing (CPU)
# -----------------------------
def ocr_worker(input_queue, output_queue):
    ocr = WagonOCR()
    while True:
        item = input_queue.get()
        if item is None: break
        
        wagon_id, crop, req_time = item
        
        # In a real scenario, DeblurGAN would run here before OCR
        
        raw_text = ocr.process_wagon(crop)
        
        if raw_text:
            parsed = IndianWagonParser.parse(raw_text)
            output_queue.put((wagon_id, raw_text, parsed, req_time))

# -----------------------------
# Cascaded Pipeline
# -----------------------------
def cascaded_pipeline(video_path, model_a_path, model_b_path):
    if not os.path.exists(video_path): return
    
    print(f"[INFO] Loading Model A (Wagon): {model_a_path}")
    model_a = YOLO(model_a_path)
    
    print(f"[INFO] Loading Model B (Number): {model_b_path}")
    # Check if model B exists, if not warn user
    if not os.path.exists(model_b_path):
        print(f"[WARNING] Model B not found at {model_b_path}. Number detection will fail.")
        model_b = None
    else:
        model_b = YOLO(model_b_path)

    cap = cv2.VideoCapture(video_path)
    
    # OCR Setup
    ocr_in_q = mp.Queue(maxsize=10)
    ocr_out_q = mp.Queue()
    ocr_p = mp.Process(target=ocr_worker, args=(ocr_in_q, ocr_out_q), daemon=True)
    ocr_p.start()
    
    # Logging Setup
    import datetime
    output_dir = os.path.join(os.path.dirname(video_path), '../../full model/detection')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(output_dir, f"{timestamp}.txt")
    
    with open(log_file_path, 'w') as f:
        f.write(f"Detection Log - {timestamp}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Video: {video_path}\n")
        f.write("--------------------------------------------------\n")
        f.write("Wagon ID | Raw Text | Parsed Data\n")
        f.write("--------------------------------------------------\n")

    print(f"[INFO] Logging results to: {log_file_path}")

    unique_wagons = set()
    
    # Restoring Initialization
    frame_cnt = 0
    prev_time = time.time()
    metrics = {'fps': deque(maxlen=50), 'det': deque(maxlen=50), 'ocr': deque(maxlen=50)}
    wagon_data = {}
    ocr_requested = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_cnt += 1
        t0 = time.time()
        
        # -----------------------------
        # STEP 1: Model A (Full Frame) - Detect Wagons
        # -----------------------------
        results_a = model_a.track(frame, persist=True, tracker="trackers/byte_track.yaml", verbose=False)
        
        active_wagons_list = []
        if results_a and results_a[0].boxes.id is not None:
            boxes = results_a[0].boxes.xyxy.cpu().numpy()
            ids = results_a[0].boxes.id.cpu().numpy()
            clss = results_a[0].boxes.cls.cpu().numpy()
            
            for box, track_id, cls in zip(boxes, ids, clss):
                track_id = int(track_id)
                if int(cls) == 0: 
                    active_wagons_list.append((track_id, box))
                    unique_wagons.add(track_id)

        # -----------------------------
        # STEP 2: Model B (Crops) - Detect Numbers
        # -----------------------------
        if model_b:
            for wagon_id, box in active_wagons_list:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                
                # Validation
                if x2<=x1 or y2<=y1: continue
                
                # Crop Wagon
                wagon_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                
                # Run Model B on Crop
                if frame_cnt % 3 == 0 and wagon_crop.size > 0:
                    results_b = model_b.predict(wagon_crop, verbose=False, conf=0.25)
                    
                    # If Number Found (Class 0 in Model B)
                    for r in results_b:
                        for nbox in r.boxes.xyxy:
                            nx1, ny1, nx2, ny2 = map(int, nbox)
                            
                            # Only trigger OCR once per wagon for now
                            if wagon_id not in ocr_requested:
                                number_img = wagon_crop[ny1:ny2, nx1:nx2]
                                if number_img.size > 0:
                                    ocr_in_q.put((wagon_id, number_img, time.time()))
                                    ocr_requested.add(wagon_id)
                                    
                            # Visualization
                            gx1, gy1 = x1 + nx1, y1 + ny1
                            gx2, gy2 = x1 + nx2, y1 + ny2
                            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)


        metrics['det'].append((time.time()-t0)*1000)

        # -----------------------------
        # STEP 3: Check OCR & Write Log
        # -----------------------------
        while not ocr_out_q.empty():
            tid, raw, parsed, req_t = ocr_out_q.get()
            metrics['ocr'].append((time.time()-req_t)*1000)
            wagon_data[tid] = {'raw': raw, 'parsed': parsed}
            
            # Write to log
            with open(log_file_path, 'a') as f:
                parsed_str = parsed['formatted'] if parsed else "N/A"
                f.write(f"{tid:<9} | {raw:<20} | {parsed_str}\n")
                if parsed:
                    f.write(f"          | Type: {parsed.get('type','')} | Rly: {parsed.get('railway','')} | Yr: {parsed.get('year','')}\n")

        # -----------------------------
        # STEP 4: Visualization
        # -----------------------------
        for wagon_id, box in active_wagons_list:
            x1, y1, x2, y2 = map(int, box)
            
            info = None
            if wagon_id in wagon_data:
                d = wagon_data[wagon_id]
                info = d['parsed']['formatted'] if d['parsed'] else d['raw']
            
            draw_track(frame, (x1,y1,x2,y2), wagon_id, info, color=(255, 0, 0))

        # Stats
        curr_time = time.time()
        metrics['fps'].append(1/(curr_time-prev_time) if curr_time>prev_time else 0)
        prev_time = curr_time
        
        avg_fps = sum(metrics['fps'])/len(metrics['fps']) if metrics['fps'] else 0
        stats = [f"FPS: {avg_fps:.1f}", 
                 f"Det Time: {sum(metrics['det'])/len(metrics['det']):.0f}ms",
                 f"Count: {len(unique_wagons)}"]
        draw_stats(frame, stats)
        
        cv2.imshow("Cascaded Pipeline", frame)
        if cv2.waitKey(1) == ord('q'): break

    ocr_in_q.put(None)
    ocr_p.join()
    cap.release()
    cv2.destroyAllWindows()
    
    print("-" * 50)
    print(f"[SUMMARY] Total Wagons Counted: {len(unique_wagons)}")
    print(f"[SUMMARY] Log saved to: {log_file_path}")
    print("-" * 50)
    
    # Write summary to log file
    with open(log_file_path, 'a') as f:
        f.write("\n--------------------------------------------------\n")
        f.write(f"Total Wagons Counted: {len(unique_wagons)}\n")
        f.write("--------------------------------------------------\n")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    # Default Paths
    parser.add_argument("--model_a", default="railway_hackathon_take4/merged_model_v3/weights/best.pt")
    # Placeholder for Model B until user trains it
    parser.add_argument("--model_b", default="railway_hackathon_numbers/number_detector_v1/weights/best.pt")
    
    args = parser.parse_args()
    cascaded_pipeline(args.video_path, args.model_a, args.model_b)
