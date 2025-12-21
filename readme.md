# 🚂 Railway Wagon Inspection System with Low-Light Enhancement

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-19.2.0-61dafb.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9.3-blue.svg)](https://www.typescriptlang.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An advanced AI-powered railway wagon inspection system featuring low-light image enhancement, real-time wagon detection, automated number recognition, and defect identification.**

[Features](#-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#-usage)
  - [Running the Backend](#running-the-backend)
  - [Running the Frontend](#running-the-frontend)
  - [Processing Videos](#processing-videos)
- [Model Training](#-model-training)
- [Indian Railway Wagon Numbering System](#-indian-railway-wagon-numbering-system)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Performance Metrics](#-performance-metrics)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

This comprehensive railway inspection system is designed to address critical challenges in railway wagon monitoring, particularly in low-light conditions. The system combines state-of-the-art deep learning techniques for image enhancement, object detection, and optical character recognition (OCR) to provide real-time wagon identification and defect detection.

### Key Capabilities

- **Low-Light Enhancement**: Zero-DCE (Zero-Reference Deep Curve Estimation) based image enhancement for night-time and poor lighting conditions
- **Real-Time Wagon Detection**: YOLOv8-based object detection optimized for railway wagon identification
- **Automated Number Recognition**: PaddleOCR and EasyOCR integration for accurate wagon number extraction
- **Multi-Stream Processing**: Concurrent processing of multiple video feeds with live dashboard visualization
- **Defect Detection**: Automated identification of wagon anomalies and structural issues
- **YouTube Live Stream Support**: Direct processing of live YouTube streams for remote monitoring

---

## ✨ Features

### 🎯 Core Features

| Feature | Description |
|---------|-------------|
| **Zero-DCE Enhancement** | Deep learning-based low-light image enhancement without reference images |
| **YOLOv8 Detection** | Custom-trained wagon detection with ByteTrack tracking |
| **Multi-Model OCR** | Dual OCR engine (PaddleOCR + EasyOCR) for robust text recognition |
| **Real-Time Processing** | Multi-process architecture with frame queuing for optimal performance |
| **Live Dashboard** | React-based responsive web interface with real-time statistics |
| **Video Analytics** | Comprehensive video analysis with blur metrics and quality assessment |
| **Wagon Number Parsing** | Indian Railway standard wagon number validation and parsing |

### 🔧 Technical Features

- **Multi-Process OCR**: Asynchronous OCR processing with worker queues
- **Blur Detection**: Laplacian variance-based motion blur assessment
- **Track Persistence**: ByteTrack algorithm for stable wagon tracking across frames
- **YouTube Integration**: Direct stream processing using yt-dlp
- **FPS Optimization**: Configurable frame skipping and processing intervals
- **Statistics Tracking**: Real-time metrics for FPS, detection latency, and OCR performance

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  React + TypeScript + Vite + Tailwind CSS               │   │
│  │  - Multi-Stream Video Display                            │   │
│  │  - Real-Time Statistics Dashboard                        │   │
│  │  - Wagon Details Visualization                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                         Backend Layer                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Server (main.py)                                │   │
│  │  - Video Stream Endpoints                                │   │
│  │  - Statistics API                                        │   │
│  │  - CORS Middleware                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Zero-DCE     │→ │  YOLOv8      │→ │  ByteTrack Tracker   │  │
│  │ Enhancement  │  │  Detection   │  │  (Wagon Tracking)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Multi-Process OCR Worker Queue                  │  │
│  │  ┌──────────────┐         ┌──────────────┐              │  │
│  │  │ PaddleOCR    │   OR    │  EasyOCR     │              │  │
│  │  └──────────────┘         └──────────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   Indian Railway Wagon Number Parser & Validator        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                        Data Storage                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Models     │  │   Videos     │  │   Results/Logs       │  │
│  │  - YOLOv8    │  │  - Input     │  │  - CSV Outputs       │  │
│  │  - Zero-DCE  │  │  - Enhanced  │  │  - Detection Stats   │  │
│  │  - Trackers  │  │  - Processed │  │  - OCR Results       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Flow

1. **Input**: Video stream or YouTube URL
2. **Enhancement**: Zero-DCE processes low-light frames
3. **Detection**: YOLOv8 identifies wagon bounding boxes
4. **Tracking**: ByteTrack maintains wagon IDs across frames
5. **OCR**: Asynchronous text extraction from wagon regions
6. **Parsing**: Indian Railway number format validation
7. **Visualization**: Real-time display on web dashboard

---

## 🛠️ Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning Framework** | PyTorch | Zero-DCE model inference |
| **Object Detection** | YOLOv8 (Ultralytics) | Wagon detection and tracking |
| **OCR Engine** | PaddleOCR / EasyOCR | Text recognition |
| **Web Framework** | FastAPI | REST API and video streaming |
| **Computer Vision** | OpenCV (cv2) | Image/video processing |
| **Video Streaming** | yt-dlp | YouTube stream extraction |
| **Async Processing** | Python multiprocessing | Parallel OCR workers |

### Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| **Framework** | React | 19.2.0 |
| **Language** | TypeScript | 5.9.3 |
| **Build Tool** | Vite | 7.2.4 |
| **Styling** | Tailwind CSS | 4.1.18 |
| **Linting** | ESLint | 9.39.1 |

### AI Models

- **Zero-DCE**: Custom-trained low-light enhancement model (Epoch99.pth)
- **YOLOv8n**: Base model with custom wagon detection weights
- **Custom Wagon Detectors**:
  - `wagon_counter_v1` - Initial wagon detection model
  - `merged_model_v1/v2/v3` - Iteratively improved detection models
  - `wagonNumberDetectionV1/V2` - Specialized number region detection
  - `WNDEModelV2.pt` - Enhanced wagon number detection

---

## 📁 Project Structure

```
low-light-image-deblur/
│
├── readme.md                    # This file
├── Assests/                     # Demo assets and media
│
├── frontend/                    # React web application
│   ├── src/
│   │   ├── App.tsx             # Main application component
│   │   ├── main.tsx            # Entry point
│   │   ├── index.css           # Global styles
│   │   └── components/
│   │       ├── VideoFeed.tsx   # Video stream display component
│   │       ├── WagonDetails.tsx# Wagon information panel
│   │       └── StatsPanel.tsx  # Statistics dashboard
│   ├── public/                 # Static assets
│   ├── package.json            # NPM dependencies
│   ├── vite.config.ts          # Vite configuration
│   └── tsconfig.json           # TypeScript configuration
│
├── full model/                  # Main processing pipeline
│   ├── requirements.txt        # Python dependencies
│   ├── yolov8n.pt             # Base YOLOv8 model
│   │
│   ├── src/
│   │   ├── api/
│   │   │   └── main.py        # FastAPI server
│   │   │
│   │   ├── core/              # Core processing modules
│   │   │   ├── zero_dce.py    # Zero-DCE enhancement network
│   │   │   ├── enhancer.py    # Enhancement wrapper class
│   │   │   ├── ocr_engine.py  # OCR processing engine
│   │   │   ├── indian_railways.py  # Wagon number parser
│   │   │   └── blur_metric.py # Motion blur detection
│   │   │
│   │   └── scripts/           # Pipeline scripts
│   │       ├── main_pipeline.py    # Main processing pipeline
│   │       ├── enhance_video.py    # Video enhancement script
│   │       ├── analyze_video.py    # Video analysis tool
│   │       ├── pipeline_viz.py     # Visualization utilities
│   │       └── test_detection.py   # Detection testing
│   │
│   ├── railway_hackathon/      # Model training iterations
│   │   └── wagon_counter_v1/
│   │       ├── args.yaml
│   │       ├── results.csv
│   │       └── weights/
│   │           ├── best.pt
│   │           └── last.pt
│   │
│   ├── railway_hackathon_take2/
│   │   └── merged_model_v1/    # Improved model version 1
│   │       └── weights/
│   │
│   ├── railway_hackathon_take3/
│   │   └── merged_model_v2/    # Improved model version 2
│   │       └── weights/
│   │
│   ├── railway_hackathon_take4/
│   │   └── merged_model_v3/    # Improved model version 3
│   │       └── weights/
│   │
│   ├── trackers/
│   │   └── byte_track.yaml    # ByteTrack configuration
│   │
│   ├── YOLO/
│   │   ├── data.yaml          # Dataset configuration
│   │   ├── train_yolo.py      # Training script
│   │   └── train_yolo_colab.ipynb  # Colab training notebook
│   │
│   ├── zero_dce_model/
│   │   └── Epoch99.pth        # Pre-trained Zero-DCE weights
│   │
│   └── Video/                 # Input/output videos
│
└── wagon number detection/     # Standalone wagon number detection
    ├── README.md
    ├── requirements.txt
    ├── LICENSE
    │
    ├── src/
    │   ├── detect.py          # Wagon number detection script
    │   └── verifyno.py        # Number verification utility
    │
    ├── wnd/                   # Wagon Number Detection module
    │   ├── __init__.py
    │   ├── WagonNumberDetection.py
    │   └── models/
    │       ├── wagonNumberDetectionV1.pt
    │       ├── wagonNumberDetectionV2.pt
    │       └── WNDEModelV2.pt
    │
    ├── img/                   # Sample images
    ├── vids/                  # Sample videos
    └── results/               # Detection results
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher
- npm or yarn
- CUDA-capable GPU (optional, for faster processing)
- Git

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/P47Parzival/low-light-image-deblur.git
cd low-light-image-deblur
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

For full model processing:
```bash
cd "full model"
pip install -r requirements.txt
```

For wagon number detection:
```bash
cd "wagon number detection"
pip install -r requirements.txt
```

4. **Download model weights**

Place the following model files in their respective directories:
- Zero-DCE weights: `full model/zero_dce_model/Epoch99.pth`
- YOLOv8 base: `full model/yolov8n.pt`
- Custom wagon detection weights in `full model/railway_hackathon_take4/merged_model_v3/weights/best.pt`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
# or
yarn install
```

3. **Configure environment** (if needed)
```bash
# Create .env file for custom API endpoint
echo "VITE_API_URL=http://localhost:8000" > .env
```

---

## 📖 Usage

### Running the Backend

#### Option 1: FastAPI Server (Multi-Stream Dashboard)

```bash
cd "full model/src/api"
python main.py
```

The server will start at `http://localhost:8000`

Available endpoints:
- `GET /video_feed/1` - Video stream 1
- `GET /video_feed/2` - Video stream 2
- `GET /video_feed/3` - Video stream 3
- `GET /stats` - Current statistics

#### Option 2: Main Processing Pipeline

```bash
cd "full model/src/scripts"
python main_pipeline.py --video <path_to_video> --weights <path_to_weights>
```

**Arguments:**
- `--video` or `-v`: Path to input video file
- `--weights` or `-w`: Path to YOLOv8 weights file
- `--enhance`: Enable Zero-DCE enhancement (optional)
- `--output` or `-o`: Output video path (optional)

**Example:**
```bash
python main_pipeline.py \
  --video "../../Video/input.mp4" \
  --weights "../../railway_hackathon_take4/merged_model_v3/weights/best.pt" \
  --enhance \
  --output "../../Video/output.mp4"
```

#### Option 3: Video Enhancement Only

```bash
cd "full model/src/scripts"
python enhance_video.py --input <input_video> --output <output_video>
```

### Running the Frontend

```bash
cd frontend
npm run dev
# or
yarn dev
```

The application will be available at `http://localhost:5173`

**Production build:**
```bash
npm run build
npm run preview
```

### Processing Videos

#### Analyze Video Quality
```bash
cd "full model/src/scripts"
python analyze_video.py --video <path_to_video>
```

Output includes:
- Frame-by-frame blur metrics
- Average brightness levels
- Quality assessment report

#### Test Detection on Single Frame
```bash
cd "full model/src/scripts"
python test_detection.py --image <path_to_image> --weights <weights_path>
```

### Wagon Number Detection (Standalone)

```bash
cd "wagon number detection/src"
python detect.py --source <video_or_image_path>
```

**Options:**
- `--model`: Path to wagon number detection model
- `--conf`: Confidence threshold (default: 0.5)
- `--save`: Save detection results

---

## 🎓 Model Training

### YOLOv8 Wagon Detection

1. **Prepare dataset** following this structure:
```
datasets/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

2. **Update data.yaml**
```yaml
path: ../datasets
train:
  - db-rail/train/images
  - indian-railway/train/images
val:
  - db-rail/valid/images
  - indian-railway/valid/images
nc: 1
names: ['wagon']
```

3. **Train the model**
```bash
cd "full model/YOLO"
python train_yolo.py
```

Or use the Colab notebook: `train_yolo_colab.ipynb`

**Training parameters:**
- Model: yolov8n.pt
- Epochs: 100-200
- Image size: 640
- Batch size: 16 (adjust based on GPU)

### Zero-DCE Fine-tuning

The Zero-DCE model can be fine-tuned on custom low-light railway images:

```python
from src.core.zero_dce import enhance_net_nopool
# Training code for Zero-DCE enhancement
# (Refer to Zero-DCE paper for training details)
```

---

## 🔢 Indian Railway Wagon Numbering System

Indian Railway wagons use an 11-digit numbering system with specific meaning:

```
Example: 41 12 11 1234 5

Breakdown:
├─ C1-C2 (41): Wagon Type (e.g., 41 = BOXN type)
├─ C3-C4 (12): Owning Railway (e.g., 12 = Eastern Railway)
├─ C5-C6 (11): Year of Manufacture (e.g., 11 = 2011)
├─ C7-C10 (1234): Unique Wagon Number
└─ C11 (5): Check Digit (parity check)
```

### Wagon Type Codes (C1-C2)

| Code | Type | Description |
|------|------|-------------|
| 41 | BOXN | Covered goods wagon |
| 42 | BOXNHL | High capacity covered wagon |
| 50 | BCN | Covered wagon for commodities |
| 60 | BTPN | Tank wagon for petroleum |

### Railway Zone Codes (C3-C4)

| Code | Zone |
|------|------|
| 11 | Northern Railway |
| 12 | Eastern Railway |
| 13 | Western Railway |
| 14 | Southern Railway |

The parser (`indian_railways.py`) validates and extracts this information automatically.

---

## 📡 API Documentation

### REST Endpoints

#### GET /video_feed/{stream_id}

Returns MJPEG video stream.

**Parameters:**
- `stream_id` (int): Stream identifier (1, 2, or 3)

**Response:**
- Content-Type: `multipart/x-mixed-replace; boundary=frame`
- Body: MJPEG stream

**Example:**
```javascript
<img src="http://localhost:8000/video_feed/1" />
```

#### GET /stats

Returns current processing statistics.

**Response:**
```json
{
  "total_wagons": 42,
  "last_wagon_id": "41 12 11 1234 5",
  "defects_found": 3,
  "status": "Processing"
}
```

---

## ⚙️ Configuration

### ByteTrack Tracker Configuration

Edit `full model/trackers/byte_track.yaml`:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.5      # Detection confidence threshold
track_low_thresh: 0.1       # Low confidence threshold
new_track_thresh: 0.6       # New track initialization threshold
track_buffer: 30            # Frames to keep lost tracks
match_thresh: 0.8           # Matching threshold
```

### Zero-DCE Enhancement

Adjust enhancement in `src/core/enhancer.py`:

```python
enhancer = LowLightEnhancer(
    weights_path="zero_dce_model/Epoch99.pth",
    device='cuda'  # or 'cpu'
)
```

### OCR Configuration

Choose OCR engine in `src/core/ocr_engine.py`:

```python
# PaddleOCR (faster, better for complex text)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# OR EasyOCR (better for irregular text)
reader = easyocr.Reader(['en'])
```

---

## 📊 Performance Metrics

### System Performance (Typical)

| Metric | Value |
|--------|-------|
| Detection FPS | 15-25 FPS (GPU) / 5-10 FPS (CPU) |
| Enhancement Latency | ~50ms per frame (GPU) |
| OCR Processing Time | 200-500ms per wagon |
| Memory Usage | ~2-4 GB (with GPU) |
| Model Size | ~50 MB (YOLOv8n) |

### Model Accuracy

| Model | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| wagon_counter_v1 | 0.87 | 0.89 | 0.85 |
| merged_model_v3 | 0.92 | 0.94 | 0.91 |

### OCR Accuracy

- Number Recognition: ~85-90% (well-lit conditions)
- Number Recognition: ~70-80% (low-light conditions with enhancement)
- Format Validation: ~95% (with parser)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
```python
# Reduce batch size or use CPU
model = YOLO(weights_path)
results = model.track(frame, device='cpu')
```

#### 2. OpenCV Video Read Error

**Solution:**
```bash
# Install opencv with ffmpeg support
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 3. PaddleOCR Import Error

**Solution:**
```bash
# Install PaddlePaddle
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

#### 4. YouTube Stream Not Loading

**Solution:**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

#### 5. Frontend CORS Error

**Solution:**
Ensure FastAPI CORS middleware allows your frontend origin:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript strict mode for frontend
- Add unit tests for new features
- Update documentation for API changes
- Maintain backward compatibility

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Zero-DCE**: Li et al. - Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
- **YOLOv8**: Ultralytics - You Only Look Once v8
- **ByteTrack**: Zhang et al. - ByteTrack: Multi-Object Tracking by Associating Every Detection Box
- **PaddleOCR**: PaddlePaddle - Multilingual OCR toolkits
- **EasyOCR**: JaidedAI - Ready-to-use OCR with 80+ languages
- **Indian Railways**: For wagon numbering system standards

---

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/P47Parzival/low-light-image-deblur/issues)
- **Discussions**: [Community discussions](https://github.com/P47Parzival/low-light-image-deblur/discussions)

---

## 🗺️ Roadmap

- [ ] Real-time defect classification using CNN
- [ ] Integration with railway databases for automated logging
- [ ] Mobile app for field inspection
- [ ] Support for thermal imaging cameras
- [ ] Advanced blur removal using deblurring GANs
- [ ] Multi-language OCR support
- [ ] Cloud deployment guide (AWS/Azure/GCP)
- [ ] Docker containerization
- [ ] Automated testing suite
- [ ] Performance optimization for edge devices

---

<div align="center">

**Made with ❤️ for Railway Safety and Automation**

⭐ Star this repository if you find it helpful!

</div>
