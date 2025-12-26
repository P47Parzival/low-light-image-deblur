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

# Import Database Module
sys.path.append(os.path.join(os.path.dirname(__file__), '../core'))
import database
import report_generator
from fastapi.responses import Response

app = FastAPI()

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
    # output(dest='S') returns the document as a string (latin-1 encoding).
    # We need to encode it to bytes.
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    
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


def generate_frames(url_key):
    urls = {
        1: "https://www.youtube.com/watch?v=7xdHH9KMSVk",
        2: "https://www.youtube.com/watch?v=nO81bQFql7M",
        3: "https://www.youtube.com/watch?v=23tmCNeFh7A"
    }
    
    youtube_url = urls.get(url_key, urls[1])
    try:
        stream_url = get_youtube_stream_url(youtube_url)
    except Exception as e:
        print(f"Error getting YouTube URL for stream {url_key}: {e}")
        return

    cap = cv2.VideoCapture(stream_url)
    
    # Simulate processing loop
    while True:
        success, frame = cap.read()
        # print("Frame read:", success)
        if not success:
            # If video ends, try to reconnect
            print(f"Stream {url_key} ended, restarting...")
            cap.release()
            try:
                stream_url = get_youtube_stream_url(youtube_url)
                cap = cv2.VideoCapture(stream_url)
                continue
            except:
                break

        # Encode header
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit FPS
        time.sleep(0.04) 

@app.get("/video_feed/{stream_id}")
async def video_feed(stream_id: int):
    return StreamingResponse(generate_frames(stream_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
async def get_stats():
    return mock_stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
