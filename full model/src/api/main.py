from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import yt_dlp
import uvicorn
import asyncio
import random
import time

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Mock Data
mock_stats = {
    "total_wagons": 0,
    "last_wagon_id": "N/A",
    "defects_found": 0,
    "status": "Idle"
}

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
        print("Frame read:", success)
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
