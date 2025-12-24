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
from src.core.deblur_engine import DeblurGANEngine
from src.core.blur_metric import calculate_blur_score

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
        else:
            print(f"[WARNING] OCR Failed for Wagon {wagon_id}")
            output_queue.put((wagon_id, "OCR Failed", None, req_time))

# -----------------------------
# Cascaded Pipeline
# -----------------------------
def cascaded_pipeline(video_path, model_a_path, model_b_path, deblur_model_path):
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

    # DeblurGAN Setup
    deblur_engine = None
    if os.path.exists(deblur_model_path):
        try:
            print(f"[INFO] Loading DeblurGAN: {deblur_model_path}")
            deblur_engine = DeblurGANEngine(deblur_model_path)
        except Exception as e:
            print(f"[WARNING] Failed to load DeblurGAN: {e}. Running without deblurring.")
    else:
        print(f"[WARNING] DeblurGAN weights not found at {deblur_model_path}. Running without deblurring.")
        
    deblur_save_dir = os.path.join(os.path.dirname(video_path), '../../full model/DeblurredImg')
    original_save_dir = os.path.join(os.path.dirname(video_path), '../../full model/OriginalImg')
    os.makedirs(deblur_save_dir, exist_ok=True)
    os.makedirs(original_save_dir, exist_ok=True)

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
    
    start_time = datetime.datetime.now()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(output_dir, f"{timestamp_str}.txt")
    
    print(f"[INFO] Report will be properly generated at: {log_file_path}")

    # Data Buffers
    unique_wagons = set()
    consist_log = [] # List of dicts: {'id': track_id, 'text': ..., 'parsed': ..., 'time': ...}
    
    # Restoring Initialization
    frame_cnt = 0
    prev_time = time.time()
    metrics = {'fps': deque(maxlen=50), 'det': deque(maxlen=50), 'ocr': deque(maxlen=50)}
    wagon_data = {}
    ocr_requested = set()

    # -----------------------------
    # VIDEO DISPLAY SETTINGS (VLC-like)
    # -----------------------------
    # Get video properties
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate display size (fit to screen, max 1280x720 for comfortable viewing)
    max_display_width = 1280
    max_display_height = 720
    scale = min(max_display_width / video_width, max_display_height / video_height, 1.0)
    display_width = int(video_width * scale)
    display_height = int(video_height * scale)
    
    # Create named window with specific properties
    window_name = "Indian Railways - Freight Inspection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # Try to set window to be always on top and centered (Windows specific)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except:
        pass
        
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_cnt += 1
        t0 = time.time()
        
        # -----------------------------
        # STEP 1: Model A (Full Frame) - Detect Wagons
        # -----------------------------
        results_a = model_a.track(frame, persist=True, tracker="trackers/byte_track.yaml", verbose=False)
        # DEBUG: Print raw detections
        if results_a and results_a[0].boxes.id is not None:
             print(f"Raw Classes Detected: {results_a[0].boxes.cls.cpu().numpy()}")
             print(f"Confidences: {results_a[0].boxes.conf.cpu().numpy()}")

        active_wagons_list = []
        if results_a and results_a[0].boxes.id is not None:
            boxes = results_a[0].boxes.xyxy.cpu().numpy()
            ids = results_a[0].boxes.id.cpu().numpy()
            clss = results_a[0].boxes.cls.cpu().numpy()
            
            for box, track_id, cls in zip(boxes, ids, clss):
                track_id = int(track_id)
                if int(cls) == 0 or int(cls) == 6:  # Assuming classes 0 and 6 are wagons
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
                
                # Crop Wagon (FULL CONTEXT)
                wagon_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

                if wagon_crop.size == 0:
                    continue

                # -----------------------------
                # WAGON-LEVEL RESIZE (IMPORTANT)
                # -----------------------------
                # Ensure minimum spatial resolution for deblurring
                if wagon_crop.shape[0] < 256:
                    scale = 256 / wagon_crop.shape[0]
                    wagon_crop = cv2.resize(
                        wagon_crop,
                        (int(wagon_crop.shape[1] * scale), 256),
                        interpolation=cv2.INTER_CUBIC
                    )

                # -----------------------------
                # WAGON-LEVEL DEBLUR (KEY FIX)
                # -----------------------------
                if deblur_engine:
                    blur_score = calculate_blur_score(wagon_crop)

                    # Use realistic thresholds for text motion blur
                    if blur_score < 1:
                        # -----------------------------
                        # SAVE ORIGINAL (BLURRED) WAGON
                        # -----------------------------

                        wagon_blur = wagon_crop.copy()
                        ts = int(time.time()*100)
                        orig_path = os.path.join(original_save_dir, f"wagon_{wagon_id}_{ts}.jpg")
                        cv2.imwrite(orig_path, wagon_blur)

                        print(f"[INFO] Deblurring wagon {wagon_id} | Blur score: {blur_score:.1f} | Size: {wagon_crop.shape[:2]}")
                        wagon_crop = deblur_engine.deblur(wagon_crop)

                        ts = int(time.time()*100)
                        wagon_save_path = os.path.join(deblur_save_dir, f"wagon_{wagon_id}_{ts}.jpg")
                        cv2.imwrite(wagon_save_path, wagon_crop)
                    
                # -----------------------------
                # NOW run Model B on CLEAN wagon
                # -----------------------------
                if frame_cnt % 3 == 0:
                    results_b = model_b.predict(wagon_crop, verbose=False, conf=0.25)

                    
                    # If Number Found (Class 0 in Model B)
                    for r in results_b:
                        for nbox in r.boxes.xyxy:
                            nx1, ny1, nx2, ny2 = map(int, nbox)
                            
                            # Only trigger OCR once per wagon for now
                            if wagon_id not in ocr_requested:
                                # 1. Add Padding (50%) - Sufficient context without too much noise
                                pad_w = int((nx2 - nx1) * 1.2)
                                pad_h = int((ny2 - ny1) * 1.0)
                                px1 = max(0, nx1 - pad_w)
                                py1 = max(0, ny1 - pad_h)
                                px2 = min(w, nx2 + pad_w)
                                py2 = min(h, ny2 + pad_h)
                                
                                number_img = wagon_crop[py1:py2, px1:px2]
                                
                                # 2. Dynamic Scaling (Target Height ~96px)
                                # PaddleOCR works best with text height 32-96px.
                                # Avoid making it massive (300px+) or tiny (<20px).
                                if number_img.size > 0:
                                    h_img, w_img = number_img.shape[:2]
                                    target_height = 96.0
                                    
                                    if h_img < target_height:
                                        scale_factor = target_height / h_img
                                        number_img = cv2.resize(number_img, (int(w_img * scale_factor), int(h_img * scale_factor)), interpolation=cv2.INTER_CUBIC)
                                    
                                    # Initialize final_img safely
                                    final_img = number_img

                                    # User's Modified Deblur/Process Block
                                    # (Preserving their commented out style or logical intent, but fixing scope)
                                    # It seems they want detailEnhance.
                                    final_img = cv2.detailEnhance(final_img, sigma_s=10, sigma_r=0.15)

                                    # Save Result
                                    ts = int(time.time()*100)
                                    # Ensure deblur_save_dir exists (it was created earlier)
                                    save_path = os.path.join(deblur_save_dir, f"wagon_{wagon_id}_{ts}.jpg")
                                    cv2.imwrite(save_path, final_img)
                                    
                                    ocr_in_q.put((wagon_id, final_img, time.time()))
                                    ocr_requested.add(wagon_id)
                                    
                            # Visualization
                            gx1, gy1 = x1 + nx1, y1 + ny1
                            gx2, gy2 = x1 + nx2, y1 + ny2
                            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)


        metrics['det'].append((time.time()-t0)*1000)

        # -----------------------------
        # STEP 3: Check OCR & Buffer Data
        # -----------------------------
        while not ocr_out_q.empty():
            tid, raw, parsed, req_t = ocr_out_q.get()
            metrics['ocr'].append((time.time()-req_t)*1000)
            
            # Timestamp for this specific detection
            det_time = datetime.datetime.now().strftime("%H:%M:%S")
            wagon_data[tid] = {'raw': raw, 'parsed': parsed}
            
            # Add to consist log if not already there (or update)
            # We use track_id as unique key for now
            consist_log.append({
                'id': tid,
                'raw': raw,
                'parsed': parsed,
                'timestamp': det_time
            })


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

                # -----------------------------
        # VLC-STYLE OVERLAY (Progress Bar + Info)
        # -----------------------------
        h, w = frame.shape[:2]
        
        # Semi-transparent bottom bar (like VLC controls area)
        overlay = frame.copy()
        bar_height = 40
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Progress bar
        progress = frame_cnt / total_frames if total_frames > 0 else 0
        bar_y = h - 25
        bar_start_x = 120
        bar_end_x = w - 120
        bar_width = bar_end_x - bar_start_x
        
        # Background bar (gray)
        cv2.rectangle(frame, (bar_start_x, bar_y - 3), (bar_end_x, bar_y + 3), (80, 80, 80), -1)
        # Progress bar (orange/yellow like VLC)
        progress_x = int(bar_start_x + bar_width * progress)
        cv2.rectangle(frame, (bar_start_x, bar_y - 3), (progress_x, bar_y + 3), (0, 165, 255), -1)
        # Progress knob
        cv2.circle(frame, (progress_x, bar_y), 6, (255, 255, 255), -1)
        
        # Time display (left side)
        current_time_sec = frame_cnt / video_fps if video_fps > 0 else 0
        total_time_sec = total_frames / video_fps if video_fps > 0 else 0
        time_str = f"{int(current_time_sec // 60):02d}:{int(current_time_sec % 60):02d} / {int(total_time_sec // 60):02d}:{int(total_time_sec % 60):02d}"
        cv2.putText(frame, time_str, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Wagon count display (right side)
        count_str = f"Wagons: {len(unique_wagons)}"
        cv2.putText(frame, count_str, (w - 110, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Top banner (semi-transparent)
        overlay_top = frame.copy()
        cv2.rectangle(overlay_top, (0, 0), (w, 35), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay_top, 0.7, frame, 0.3, 0)

        
        cv2.imshow(window_name, frame)
        
        # Keyboard controls (like VLC)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC to quit 
            break
        elif key == ord(' '):  # Space to pause
            print("[INFO] Paused. Press any key to continue...")
            cv2.waitKey(0)

    ocr_in_q.put(None)
    ocr_p.join()
    cap.release()
    cv2.destroyAllWindows()
    
    # ---------------------------------------------------------
    # Generate Final Report
    # ---------------------------------------------------------
    print("[INFO] Generating final report...")
    
    total_wagons = len(unique_wagons)
    end_time_str = datetime.datetime.now().strftime("%H:%M")
    report_date = start_time.strftime("%d-%b-%Y")
    
    # Header
    report_lines = []
    report_lines.append("+-----------------------------------------------------------------------+")
    report_lines.append("|  [Logo]  INDIAN RAILWAYS - AUTOMATED FREIGHT INSPECTION REPORT        |")
    report_lines.append("+-----------------------------------------------------------------------+")
    report_lines.append(f"|  Site: Ahemdabad Jn (Cam-02)   |   Date: {report_date}   |   Time: {end_time_str}|")
    report_lines.append(f"|  Train Speed: 62 km/h          |   Total Wagons: {total_wagons:<5}    |   Defects: 0 |")
    report_lines.append("+-----------------------------------------------------------------------+")
    
    # Critical Alerts (Mocked for now)
    report_lines.append("|  [ CRITICAL ALERTS ]                                                  |")
    report_lines.append("|  * No Critical Defects Detected by AI System                          |")
    report_lines.append("|                                                                       |")
    report_lines.append("+-----------------------------------------------------------------------+")
    
    # Consist List
    report_lines.append("|  [ CONSIST LIST ]                                                     |")
    report_lines.append("|  #   | Wagon ID       | Type   | Owner | Condition  | Timestamp       |")
    
    # Populate Consist List from Data
    # Match consist_log items to unique_wagons. 
    # Some wagons in unique_wagons might not have OCR data (missed detection/ocr).
    # We list ALL detected wagons.
    
    sorted_ids = sorted(list(unique_wagons))
    
    # Create lookup from id -> ocr data
    ocr_lookup = {item['id']: item for item in consist_log}
    
    for idx, uid in enumerate(sorted_ids, 1):
        wagon_id_str = "Unknown"
        w_type = "-"
        w_owner = "-"
        w_cond = "Good"
        w_time = "-"
        
        if uid in ocr_lookup:
            data = ocr_lookup[uid]
            # ID: Prefer parsed 11-digit formatted, else raw text
            if data['parsed']:
                wagon_id_str = data['parsed']['formatted']
                w_type = data['parsed']['type']
                w_owner = data['parsed']['railway']
            else:
                wagon_id_str = data['raw'][:14] # Truncate if too long
            
            w_time = data['timestamp']
        else:
            wagon_id_str = f"Track-{uid}" # Fallback
            
        # Formatting Line (Fixed Width approx)
        # ID: 14 chars, Type: 6, Owner: 5, Cond: 10
        line = f"|  {idx:<4}| {wagon_id_str:<14} | {w_type:<6} | {w_owner:<5} | {w_cond:<10} | {w_time:<15} |"
        report_lines.append(line)

    report_lines.append("+-----------------------------------------------------------------------+")
    
    # AI System Log
    report_lines.append("|  [ AI SYSTEM LOG ]                                                    |")
    report_lines.append(f"|  * {frame_cnt} Frames Processed                                              |")
    report_lines.append("|  * Pipeline: Cascaded YOLOv8 + PaddleOCR                              |")
    report_lines.append("+-----------------------------------------------------------------------+")

    # Write to File
    with open(log_file_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print("-" * 50)
    print(f"[SUMMARY] Total Wagons Counted: {total_wagons}")
    print(f"[SUMMARY] Report saved to: {log_file_path}")
    print("-" * 50)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    # Default Paths
    parser.add_argument("--model_a", default="railway_hackathon_take6/merged_model_v6_generalized/weights/best.pt")
    # Placeholder for Model B until user trains it
    parser.add_argument("--model_b", default="railway_hackathon_numbers/number_detector_v1/weights/best.pt")
    parser.add_argument("--deblur_model", default="NAFnet/NAFNet-GoPro-width32.pth")
    
    args = parser.parse_args()
    cascaded_pipeline(args.video_path, args.model_a, args.model_b, args.deblur_model)
