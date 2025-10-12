from __future__ import annotations

import tempfile
from pathlib import Path
from collections import Counter
import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

load_dotenv()

from .attendance import (  # noqa: E402
    build_attendance,
    create_annotated_image_data,
    extract_crops,
    extract_predictions,
    generate_manual_crops,
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
            attendance_entries = build_attendance(predictions)
            crop_map = extract_crops(raw_result)
            manual_crops = generate_manual_crops(temp_path, predictions)
            if manual_crops:
                stats = crop_map.get("stats")
                if not isinstance(stats, dict):
                    stats = {}
                    crop_map["stats"] = stats
                stats["manual_bbox_generated"] = stats.get("manual_bbox_generated", 0) + len(manual_crops)

                by_prediction_map = crop_map.get("by_prediction_id")
                if not isinstance(by_prediction_map, dict):
                    by_prediction_map = {}
                    crop_map["by_prediction_id"] = by_prediction_map

                by_label_map = crop_map.get("by_label")
                if not isinstance(by_label_map, dict):
                    by_label_map = {}
                    crop_map["by_label"] = by_label_map

                fallback_list = crop_map.get("fallback")
                if not isinstance(fallback_list, list):
                    fallback_list = []
                    crop_map["fallback"] = fallback_list

                for payload in manual_crops:
                    stored = False
                    pred_id = payload.get("prediction_id")
                    if pred_id:
                        key = str(pred_id).strip()
                        if key and key not in by_prediction_map:
                            by_prediction_map[key] = payload
                            stored = True
                    label = payload.get("label")
                    if isinstance(label, str):
                        key = label.strip()
                        if key and key not in by_label_map:
                            by_label_map[key] = payload
                            stored = True
                    if not stored:
                        fallback_list.append(payload)
            annotated_image = create_annotated_image_data(raw_result)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    crop_by_prediction = crop_map.get("by_prediction_id", {}) or {}
    crop_by_label = crop_map.get("by_label", {}) or {}
    crop_by_label_lower = {
        key.lower(): value
        for key, value in crop_by_label.items()
        if isinstance(key, str)
    }
    fallback_queue: List[Dict[str, Any]] = []
    raw_fallback = crop_map.get("fallback")
    if isinstance(raw_fallback, list):
        fallback_queue = [item for item in raw_fallback if isinstance(item, dict)]

    attendance_payload: List[Dict[str, Any]] = []
    crop_source_counter: Counter[str] = Counter()
    for entry in attendance_entries:
        crop_info = None
        crop_source = None
        prediction_id = entry.get("prediction_id")
        if prediction_id is not None:
            key = str(prediction_id).strip()
            if key:
                crop_info = crop_by_prediction.get(key)
                if crop_info:
                    crop_source = "prediction_id"
        if not crop_info:
            roll_number = entry.get("roll_number")
            if isinstance(roll_number, str):
                crop_info = crop_by_label.get(roll_number)
                if crop_info:
                    crop_source = "label"
                else:
                    crop_info = crop_by_label_lower.get(roll_number.lower())
                    if crop_info:
                        crop_source = "label"
        if not crop_info and fallback_queue:
            crop_info = fallback_queue.pop(0)
            if crop_info:
                crop_source = "fallback"

        attendance_payload.append(
            {
                **entry,
                "crop_image": crop_info.get("image") if crop_info else None,
                "crop_label": crop_info.get("label") if crop_info else None,
                "crop_metadata": crop_info.get("metadata") if crop_info and isinstance(crop_info.get("metadata"), dict) else None,
                "crop_prediction_id": crop_info.get("prediction_id") if crop_info else None,
                "crop_source": crop_source,
            }
        )
        if crop_source:
            crop_source_counter[crop_source] += 1

    message = "Attendance processed" if attendance_payload else "No students detected"
    detections = [pred.to_dict() for pred in predictions]
    debug_info = {
        "detection_count": len(predictions),
        "detection_classes": sorted({pred.class_name for pred in predictions if pred.class_name}),
        "attendance_count": len(attendance_payload),
        "crop_sources": dict(crop_source_counter),
        "crop_map_counts": {
            "by_prediction_id": len(crop_by_prediction),
            "by_label": len(crop_by_label),
            "fallback_remaining": len(fallback_queue),
        },
        "crop_stats": crop_map.get("stats", {}),
    }

    if os.environ.get("DEBUG_RAW", "0").lower() in {"1", "true", "yes"}:
        try:
            # Provide a truncated raw workflow blob for troubleshooting (avoid huge responses)
            serialized = json.dumps(raw_result, separators=(",", ":"))
            debug_info["raw_result_truncated"] = serialized[:8000] + ("..." if len(serialized) > 8000 else "")
        except Exception:  # noqa: BLE001
            debug_info["raw_result_truncated"] = "<serialization_failed>"
    response_payload = {
        "message": message,
        "attendance": attendance_payload,
        "detections": detections,
        "annotated_image": annotated_image,
        "crops": crop_map,
        "debug": debug_info,
    }
    return JSONResponse(response_payload)


__all__ = ["app"]


# ------------------------ Inline Edit API ------------------------
class AttendanceUpdate(BaseModel):
    row_id: int
    roll_number: str
    status: str

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        allowed = {"present", "absent"}
        lv = (v or "").strip().lower()
        if lv not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return lv


@app.post("/api/attendance/update", response_class=JSONResponse)
async def attendance_update(payload: AttendanceUpdate) -> JSONResponse:
    try:
        updated = {
            "row_id": payload.row_id,
            "roll_number": payload.roll_number.strip(),
            "status": payload.status.strip().lower(),
        }
        return JSONResponse({"ok": True, "updated": updated})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
