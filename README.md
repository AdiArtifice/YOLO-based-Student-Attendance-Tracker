# Roboflow Local Inference + Workflow Runner

This project demonstrates how to spin up a local Roboflow Inference server (Docker-based) and run a workflow against single images, a folder, or a webcam for a YOLO-based student attendance system.

## 🔰 Initial Setup (One-Time)
Follow these steps sequentially on a fresh machine.

### 1. Clone the Repository
```powershell
git clone https://github.com/AdiArtifice/YOLO-based-Student-Attendance-Tracker.git
cd YOLO-based-Student-Attendance-Tracker
```

### 2. Create & Activate Python 3.11 Virtual Environment
```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -V   # expect 3.11.x
```
If you still see 3.13, explicitly force 3.11 (assuming installed):
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create `.env` (Never Commit Secrets)
```powershell
@'
ROBOFLOW_API_KEY=REPLACE_WITH_YOUR_KEY
ROBOFLOW_API_URL=http://localhost:9001
ROBOFLOW_WORKSPACE=tray-detection-cfllw
ROBOFLOW_WORKFLOW=detect-and-classify
@' | Out-File -FilePath .env -Encoding utf8 -NoNewline
```

### 5. Start Docker Desktop
```powershell
docker version
```
Ensure it returns both Client and Server sections.

### 6. Launch Local Inference Server
```powershell
inference server start --port 9001 --roboflow-api-key $env:ROBOFLOW_API_KEY
```
Expect log line: `Uvicorn running on http://0.0.0.0:9001`.


### 7. (Optional) Reachability Check
`/health` may return 404 but that still shows the server is responding.
```powershell
curl http://localhost:9001/health
```

### 8. Add Sample Images
Put test images in `images/`.

### 9. Run a Test Inference
```powershell
python test-roboflow.py --image .\images\sample.jpg --api-url auto --save-annotated --json
```
Outputs will appear under `outputs/`.

### 10. (Optional) Commit (Excluding `.env`)
```powershell
git add .
git commit -m "Initial inference setup"
git push origin main
```

## 🚀 Run the FastAPI Web Application

### 1. Start the API server
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Open the frontend
- Navigate to <http://localhost:8000> in your browser.
- Upload a classroom photo and wait for the response.

### 3. What happens under the hood
- The backend writes the uploaded file to a secure temporary path.
- It calls the Roboflow workflow via the local inference server.
- Predictions are parsed into an attendance list; optional annotated image is generated in-memory.
- All temporary files (input + any intermediate outputs) are deleted immediately after inference.
- The API returns JSON (`attendance`, `detections`, `annotated_image`) which the frontend renders directly.

> **Note:** Nothing is persisted on disk except while the request is running; outputs are sent back inline and temp files are wiped before responding.

## Prerequisites
- Windows 10/11 with WSL2 enabled (recommended for Docker performance)
- Docker Desktop installed and running (green whale icon)
- Python 3.11 (virtual environment recommended)

## 1. Install Python Dependencies
```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Provide Environment Variables
Create a `.env` file (never commit real keys):
```
ROBOFLOW_API_KEY=YOUR_KEY_HERE
ROBOFLOW_API_URL=http://localhost:9001
ROBOFLOW_WORKSPACE=tray-detection-cfllw
ROBOFLOW_WORKFLOW=detect-and-classify
```

## 3. Start Docker Desktop
Make sure it is running. Test with:
```powershell
docker version
```

## 4. Start Local Inference Server
```powershell
inference server start --port 9001 --roboflow-api-key $env:ROBOFLOW_API_KEY
```
If you need another port:
```powershell
inference server start --port 9100 --roboflow-api-key $env:ROBOFLOW_API_KEY
```

Health check:
```powershell
curl http://localhost:9001/health
```

## 5. Run Workflow Script
Single image:
```powershell
python test-roboflow.py --image .\image.jpg
```
Folder:
```powershell
python test-roboflow.py --folder .\images
```
Webcam:
```powershell
python test-roboflow.py --webcam 0
```

Fallback (hosted) if local not available (script auto-detects):
```powershell
python test-roboflow.py --image .\image.jpg --api-url auto
```

## 6. Outputs
CSV per image saved in `outputs/` directory. Webcam draws bounding boxes live.

Optional flags for richer outputs (still images / folders):

```powershell
python test-roboflow.py --image .\image.jpg --save-annotated --save-crops --json
```

Produces:
 - <name>_predictions.csv (tabular predictions)
 - <name>_annotated.jpg (image with boxes & labels) when --save-annotated
 - <name>_crops/ (individual detection crops) when --save-crops
 - <name>_raw.json (raw workflow response) when --json

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Error connecting to Docker daemon | Docker not running | Start Docker Desktop |
| No matching distribution for inference-cli | Wrong Python version (3.13) | Install Python 3.11 | 
| curl health fails | Server not running / port blocked | Re-start server, check firewall |
| OpenCV import error | Missing dependencies | `pip install opencv-python-headless` |

## Next Enhancements
- Video file processing
- Attendance logic (map class -> student ID + database)
- Logging + latency metrics

## License
Internal / educational use.
