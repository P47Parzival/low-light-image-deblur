from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import yt_dlp
import uvicorn
import asyncio
import random
import time
import os
import sys
from datetime import datetime

# Import Database Module
sys.path.append(os.path.join(os.path.dirname(__file__), '../core'))
import database
import report_generator
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import Response

# Import Pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from cascaded_pipeline import cascaded_pipeline

app = FastAPI()

# Initialize DB (Run Migrations)
database.init_db()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve Static Files (Images)
# The pipeline saves images to:
# - .../full model/detection
# - .../full model/DeblurredImg
# - .../full model/OriginalImg
# - .../full model/OCRimage
# We mount the 'full model' parent directory (project root) so we can access all of them.
# Fix: Use '../../' to go to root. Do NOT append 'full model' again.
full_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.makedirs(full_model_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=full_model_dir), name="static")

# Global Mock Data (Still used for live stats for now)
mock_stats = {
    "total_wagons": 0,
    "last_wagon_id": "N/A",
    "defects_found": 0,
    "status": "Idle"
}

# ... (YouTube functions remain same, skipping for brevity in this replace block if possible, but replace_file_content replaces chunks)
# I will keep the existing imports and setup, just adding the new routes.

@app.get("/history")
async def get_history():
    """Get list of all past inspections."""
    return database.get_all_inspections()

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a video and automatically trigger processing."""
    try:
        # Define Paths
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../full model'))
        video_dir = os.path.join(base_dir, 'Video')
        os.makedirs(video_dir, exist_ok=True)
        
        file_path = os.path.join(video_dir, file.filename)
        
        # Save File
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        print(f"[API] Video saved to: {file_path}")
        
        # Define Model Paths
        model_a = os.path.join(base_dir, "railway_hackathon_take6/merged_model_v6_generalized/weights/best.pt")
        model_b = os.path.join(base_dir, "railway_hackathon_numbers/number_detector_v1/weights/best.pt") 
        deblur_model = os.path.join(base_dir, "NAFnet/NAFNet-GoPro-width64.pth")
        
        # Create Inspection Record BEFORE processing (so frontend has an ID)
        inspection_id = database.create_inspection(file.filename)
        
        # Trigger Pipeline in Background
        background_tasks.add_task(
            cascaded_pipeline, 
            video_path=file_path,
            model_a_path=model_a,
            model_b_path=model_b,
            deblur_model_path=deblur_model,
            headless=True,
            inspection_id=inspection_id
        )
        
        return {
            "message": "Upload successful. Processing started in background.", 
            "filename": file.filename,
            "inspection_id": inspection_id,
            "status": "PROCESSING"
        }
        
    except Exception as e:
        print(f"Error during upload: {e}")
        return Response(content=f"Upload failed: {str(e)}", status_code=500)

@app.get("/inspections/{inspection_id}/status")
async def get_inspection_status(inspection_id: int):
    """Check the status of a specific inspection."""
    insp = database.get_inspection_by_id(inspection_id)
    if not insp:
        return Response(content="Inspection not found", status_code=404)
    
    # If status column missing (migration edge case), assume completed if wagons exist?
    # Or just default to COMPLETED if not PROCESSING?
    status = insp.get('status', 'COMPLETED') 
    return {"status": status}


@app.get("/history/{inspection_id}/report")
async def generate_report_pdf(inspection_id: int):
    """Generate and download PDF report for an inspection."""
    inspection = database.get_inspection_by_id(inspection_id)
    if not inspection:
        return Response(content="Inspection not found", status_code=404)
        
    wagons = database.get_wagons_for_inspection(inspection_id)
    
    # Generate PDF
    pdf = report_generator.generate_report(inspection, wagons)
    
    # Output to bytes
    # The .output() method returns the document as bytes, which is ready for the response.
    pdf_byte_array = pdf.output()
    pdf_bytes = bytes(pdf_byte_array) # Ensure it's the immutable `bytes` type for the Response
    
    headers = {
        'Content-Disposition': f'attachment; filename="report_{inspection_id}.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

@app.get("/history/{inspection_id}")
async def get_inspection_details(inspection_id: int):
    """Get all wagons for a specific inspection."""
    wagons = database.get_wagons_for_inspection(inspection_id)
    
    clean_wagons = []
    for w in wagons:
        w_dict = dict(w)
        # Convert absolute path to static URL
        # Logic: find 'full model' in path and take everything after it
        for key in ['original_image_path', 'deblurred_image_path', 'cropped_number_path']:
             # Note: API might return keys slightly differently depending on DB row factory
             # But let's assume keys match schema
            val = w_dict.get(key)
            if val and isinstance(val, str) and 'full model' in val:
                # abs_path: C:\Users\dhruv\...\full model\DeblurredImg\wagon_1_123.jpg
                # rel_path: DeblurredImg/wagon_1_123.jpg
                
                # Split by 'full model' (ignoring case if possible, but usually FS matches)
                # We use simple split assuming standard installation
                parts = val.split('full model')
                if len(parts) > 1:
                    rel_path = parts[-1].replace('\\', '/').lstrip('/')
                    w_dict[key] = f"http://localhost:8000/static/{rel_path}"
        
        clean_wagons.append(w_dict)
        
    return clean_wagons

@app.get("/stats")
async def get_stats():
    return mock_stats

def get_youtube_stream_url(youtube_url: str) -> str:
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


async def generate_frames(url_key: int, request: Request):
    """Generator function to stream video frames from a YouTube URL."""
    urls = {
        1: "https://www.youtube.com/watch?v=7xdHH9KMSVk",
        2: "https://www.youtube.com/watch?v=nO81bQFql7M",
        3: "https://www.youtube.com/watch?v=23tmCNeFh7A"
    }
    youtube_url = urls.get(url_key, urls[1])
    cap = None
    try:
        stream_url = get_youtube_stream_url(youtube_url)
        cap = cv2.VideoCapture(stream_url)
        loop = asyncio.get_running_loop()
        
        while True:
            # Check if the client has disconnected
            if await request.is_disconnected():
                print(f"Client disconnected for stream {url_key}. Stopping.")
                break

            # Run blocking I/O (cap.read) in a separate thread to avoid blocking the event loop
            success, frame = await loop.run_in_executor(None, cap.read)

            if not success:
                print(f"Stream {url_key} ended or failed. Breaking loop.")
                break

            # Also run the potentially blocking encoding in an executor
            ret, buffer = await loop.run_in_executor(None, cv2.imencode, '.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(0.04)  # Use asyncio.sleep for non-blocking delay
    except Exception as e:
        # This will catch errors like client disconnection during the yield
        print(f"An error occurred in generate_frames for stream {url_key} (client likely disconnected): {e}")
    finally:
        if cap:
            print(f"Releasing video capture for stream {url_key}.")
            cap.release()

@app.get("/video_feed/{stream_id}")
async def video_feed(stream_id: int, request: Request):
    return StreamingResponse(generate_frames(stream_id, request), media_type="multipart/x-mixed-replace; boundary=frame")


# -----------------------------
# LIVE PROCESSING CONTROL
# -----------------------------
live_process = None
live_inspection_id = None

import multiprocessing

def run_live_pipeline(stream_url, inspection_id):
    """Wrapper to run the pipeline in a separate process."""
    # Define Model Paths (Same as upload)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../full model'))
    model_a = os.path.join(base_dir, "railway_hackathon_take6/merged_model_v6_generalized/weights/best.pt")
    model_b = os.path.join(base_dir, "railway_hackathon_numbers/number_detector_v1/weights/best.pt") 
    deblur_model = os.path.join(base_dir, "NAFnet/NAFNet-GoPro-width64.pth")
    
    # Run Pipeline
    # Note: cascaded_pipeline handles DB connections internally
    cascaded_pipeline(
        video_path=stream_url,
        model_a_path=model_a,
        model_b_path=model_b,
        deblur_model_path=deblur_model,
        headless=True,
        inspection_id=inspection_id
    )

@app.post("/live/start")
async def start_live_processing(stream_id: int = 1):
    """Start the AI pipeline on the live stream."""
    global live_process, live_inspection_id
    
    if live_process and live_process.is_alive():
        return {"status": "error", "message": "Live processing is already running."}
    
    # Get Stream URL
    urls = {
        1: "https://www.youtube.com/watch?v=7xdHH9KMSVk",
        2: "https://www.youtube.com/watch?v=nO81bQFql7M",
        3: "https://www.youtube.com/watch?v=23tmCNeFh7A"
    }
    youtube_url = urls.get(stream_id, urls[1])
    try:
        stream_url = get_youtube_stream_url(youtube_url)
    except Exception as e:
        return {"status": "error", "message": f"Failed to get stream URL: {e}"}
        
    # Create Inspection Record
    video_name = f"Live Stream {stream_id} - {datetime.now().strftime('%H:%M')}"
    live_inspection_id = database.create_inspection(video_name)
    database.update_inspection_times(live_inspection_id, start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Start Process
    live_process = multiprocessing.Process(
        target=run_live_pipeline, 
        args=(stream_url, live_inspection_id)
    )
    live_process.start()
    
    print(f"[API] Live processing started. PID: {live_process.pid} | Inspection ID: {live_inspection_id}")
    
    return {
        "status": "started", 
        "inspection_id": live_inspection_id, 
        "message": "AI Pipeline attached to live feed."
    }

@app.post("/live/stop")
async def stop_live_processing():
    """Stop the AI pipeline."""
    global live_process, live_inspection_id
    
    if live_process and live_process.is_alive():
        print(f"[API] Stopping live processing process {live_process.pid}...")
        live_process.terminate()
        live_process.join()  # Wait for it to finish
        live_process = None
        
        # Update DB
        if live_inspection_id:
            database.update_inspection_times(live_inspection_id, end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            database.update_inspection_status(live_inspection_id, "COMPLETED") 
            # Note: Might leave metrics empty if pipeline didn't finish gracefully, 
            # but that's acceptable for forced stop.
        
        return {"status": "stopped", "message": "Live processing stopped."}
    
    return {"status": "idle", "message": "No active processing to stop."}

@app.get("/live/status")
async def get_live_status():
    """Check if live processing is active."""
    global live_process, live_inspection_id
    is_running = live_process is not None and live_process.is_alive()
    return {
        "is_running": is_running,
        "inspection_id": live_inspection_id if is_running else None
    }

@app.get("/stats")
async def get_stats():
    return mock_stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
