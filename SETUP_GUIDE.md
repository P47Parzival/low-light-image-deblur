# GARUD Setup & Start Guide

## Quick Start (3 Steps)

### Step 1: Setup Backend
Open your terminal and run:
```bash
cd "c:\ADANI HACKATHON\low-light-image-deblur"
setup_backend.bat
```

This will:
- Create Python virtual environment
- Install all Python dependencies (PyTorch, TensorFlow, FastAPI, etc.)
- Initialize SQLite database
- Takes ~10-15 minutes

### Step 2: Setup Frontend
In your terminal, run:
```bash
cd "c:\ADANI HACKATHON\low-light-image-deblur"
setup_frontend.bat
```

This will:
- Install Node.js dependencies
- Verify build configuration
- Takes ~2-3 minutes

### Step 3: Start Application
In your terminal, run:
```bash
cd "c:\ADANI HACKATHON\low-light-image-deblur"
start.bat
```

This will:
- Start backend server on http://localhost:8000
- Start frontend server on http://localhost:5173
- Open browser automatically

## Manual Commands (If you prefer step-by-step)

### Backend Setup (Manual)
```bash
cd "c:\ADANI HACKATHON\low-light-image-deblur\full model\src\api"
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r ..\..\requirements.txt
python -c "import sys; sys.path.append('../core'); import database; database.init_db()"
```

### Frontend Setup (Manual)
```bash
cd "c:\ADANI HACKATHON\low-light-image-deblur\frontend"
npm install
```

### Start Backend (Manual)
```bash
cd "c:\ADANI HACKATHON\low-light-image-deblur\full model\src\api"
venv\Scripts\activate
python main.py
```

### Start Frontend (Manual)
```bash
cd "c:\ADANI HACKATHON\low-light-image-deblur\frontend"
npm run dev
```

## Troubleshooting

### Python not found
Install Python 3.8+ from https://www.python.org/downloads/

### Node.js not found
Install Node.js 16+ from https://nodejs.org/

### Port already in use
- Backend (8000): Change port in `full model/src/api/main.py` (line 425)
- Frontend (5173): Change port in `frontend/vite.config.ts`

### Missing models
The system will use the base YOLOv8 model (`yolov8n.pt`) if custom trained models are missing. This is normal for first-time setup.

## What to Expect

- **First run**: Downloads ~3GB of AI models, takes 10-15 minutes
- **Subsequent runs**: Starts in ~10 seconds
- **Backend API docs**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:5173
