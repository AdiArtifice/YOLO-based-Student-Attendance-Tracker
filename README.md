# Roboflow Local Inference + Workflow Runner

This project demonstrates how to spin up a local Roboflow Inference server (Docker-based) and run a workflow against single images, a folder, or a webcam for a YOLO-based student attendance system.

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
