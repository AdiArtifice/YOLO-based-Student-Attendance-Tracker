from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from inference_sdk import InferenceHTTPClient  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore

ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "http://localhost:9001")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE")
ROBOFLOW_WORKFLOW = os.environ.get("ROBOFLOW_WORKFLOW")

_client: Optional[InferenceHTTPClient] = None
_client_lock = Lock()


@dataclass
class Prediction:
    class_name: str
    confidence: float
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def _ensure_client() -> InferenceHTTPClient:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not ROBOFLOW_API_KEY:
            raise RuntimeError("ROBOFLOW_API_KEY is not set. Populate .env or environment.")
        if not ROBOFLOW_WORKSPACE or not ROBOFLOW_WORKFLOW:
            raise RuntimeError("ROBOFLOW_WORKSPACE or ROBOFLOW_WORKFLOW is not configured.")
        _client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
        return _client


def run_workflow(image_path: Path) -> Dict[str, Any]:
    client = _ensure_client()
    return client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW,
        images={"image": str(image_path)},
    )


def extract_predictions(result: Dict[str, Any]) -> List[Prediction]:
    predictions: List[Prediction] = []

    def scan(node: Any):
        if isinstance(node, dict):
            if "predictions" in node and isinstance(node["predictions"], list):
                for item in node["predictions"]:
                    if not isinstance(item, dict):
                        continue
                    predictions.append(
                        Prediction(
                            class_name=str(item.get("class", item.get("class_name", "unknown"))),
                            confidence=float(item.get("confidence", 0.0)),
                            x=_maybe_float(item.get("x")),
                            y=_maybe_float(item.get("y")),
                            width=_maybe_float(item.get("width")),
                            height=_maybe_float(item.get("height")),
                        )
                    )
            for value in node.values():
                scan(value)
        elif isinstance(node, list):
            for value in node:
                scan(value)

    scan(result)
    return predictions


def build_attendance(predictions: List[Prediction]) -> List[Dict[str, Any]]:
    seen: Dict[str, float] = {}
    for pred in predictions:
        key = pred.class_name or "unknown"
        if key not in seen or pred.confidence > seen[key]:
            seen[key] = pred.confidence
    return [
        {
            "roll_number": roll,
            "status": "present",
            "confidence": round(conf, 4),
        }
        for roll, conf in seen.items()
    ]


def create_annotated_image_data(image_path: Path, predictions: List[Prediction]) -> Optional[str]:
    boxes = [p for p in predictions if None not in (p.x, p.y, p.width, p.height)]
    if not boxes:
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except Exception:  # pragma: no cover - safe fallback
                font = None

            width, height = img.size

            def measure(label: str) -> tuple[int, int]:
                if font is None:
                    return len(label) * 7, 12
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)  # type: ignore[attr-defined]
                    return bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    pass
                if hasattr(draw, "textsize"):
                    try:
                        result = draw.textsize(label, font=font)  # type: ignore[attr-defined]
                        return int(result[0]), int(result[1])
                    except Exception:
                        pass
                return len(label) * 7, 12

            for idx, pred in enumerate(boxes):
                x1 = int(pred.x - pred.width / 2)
                y1 = int(pred.y - pred.height / 2)
                x2 = int(pred.x + pred.width / 2)
                y2 = int(pred.y + pred.height / 2)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                label = f"{pred.class_name}:{pred.confidence:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                tw, th = measure(label)
                box_top = max(0, y1 - th - 2)
                draw.rectangle([x1, box_top, x1 + tw + 4, box_top + th + 4], fill="lime")
                text_pos = (x1 + 2, box_top + 2)
                if font:
                    draw.text(text_pos, label, fill="black", font=font)
                else:
                    draw.text(text_pos, label, fill="black")

            buffered = BytesIO()
            format_hint = (image_path.suffix[1:] or "jpeg").upper()
            if format_hint.lower() not in {"jpeg", "jpg", "png"}:
                format_hint = "JPEG"
            if format_hint == "JPG":
                format_hint = "JPEG"
            img.save(buffered, format=format_hint)
            encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
            mime = "jpeg" if format_hint == "JPEG" else format_hint.lower()
            return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
