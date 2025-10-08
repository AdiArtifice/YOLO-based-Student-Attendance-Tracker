from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from .attendance import (  # noqa: E402
    build_attendance,
    create_annotated_image_data,
    extract_predictions,
    run_workflow,
)

app = FastAPI(title="YOLO Attendance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>YOLO Attendance API</h1><p>Frontend not found.</p>"


@app.get("/api/health", response_class=JSONResponse)
async def healthcheck() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/upload", response_class=JSONResponse)
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type or not file.content_type.lower().startswith("image"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    suffix = Path(file.filename).suffix or ".jpg"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp.flush()
            temp_path = Path(tmp.name)

        try:
            raw_result = run_workflow(temp_path)
            predictions = extract_predictions(raw_result)
            attendance = build_attendance(predictions)
            annotated_image = create_annotated_image_data(temp_path, predictions)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    message = "Attendance processed" if attendance else "No students detected"
    detections = [pred.to_dict() for pred in predictions]
    response_payload = {
        "message": message,
        "attendance": attendance,
        "detections": detections,
        "annotated_image": annotated_image,
    }
    return JSONResponse(response_payload)


__all__ = ["app"]
