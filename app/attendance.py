from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from inference_sdk import InferenceHTTPClient  # type: ignore

ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "http://localhost:9001")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE")
ROBOFLOW_WORKFLOW = os.environ.get("ROBOFLOW_WORKFLOW")
IGNORED_ATTENDANCE_CLASSES = {"student-faces"}
VISUALIZATION_KEYS = {
    "visualization",
    "visualizations",
    "visualization_output",
    "output_image",
    "annotated_image",
    "image",
}
CROP_KEYS = {
    "crops",
    "crop",
    "crop_results",
    "crop_images",
    "face_crops",
    "faces",
}
PREDICTION_ID_KEYS = (
    "prediction_id",
    "predictionId",
    "predictionID",
    "id",
    "uuid",
    "annotation_id",
    "source_prediction_id",
    "sourcePredictionId",
    "source_predictionId",
)
LABEL_KEYS = (
    "label",
    "class",
    "class_name",
    "name",
    "roll_number",
    "rollNumber",
    "student_id",
    "studentId",
    "tag",
)

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
    prediction_id: str | None = None
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "prediction_id": self.prediction_id,
            "metadata": self.metadata or {},
        }


def _add_stat(stats: Optional[Dict[str, int]], key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _coerce_base64_data(data: str, stats: Optional[Dict[str, int]] = None, mime_hint: Optional[str] = None) -> Optional[str]:
    cleaned = data.strip().replace("\n", "")
    if not cleaned:
        return None
    padded = cleaned + "=" * (-len(cleaned) % 4)
    try:
        base64.b64decode(padded, validate=True)
    except Exception:
        return None
    mime = (mime_hint or "image/png").strip().lower()
    if not mime:
        mime = "image/png"
    if not mime.startswith("image/"):
        mime = f"image/{mime}"
    _add_stat(stats, "base64_normalized")
    return f"data:{mime};base64,{padded}"


def _normalise_image_value(value: Any, stats: Optional[Dict[str, int]] = None) -> Optional[str]:
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.startswith("data:image/"):
            _add_stat(stats, "data_uri")
            return candidate
        if candidate.startswith("http://") or candidate.startswith("https://"):
            _add_stat(stats, "http_uri")
            return candidate
        _add_stat(stats, "string_candidate")
        return _coerce_base64_data(candidate, stats)
    if isinstance(value, dict):
        type_field = value.get("type")
        if isinstance(type_field, str) and type_field.lower() == "base64" and "value" in value:
            _add_stat(stats, "dict_type_base64")
            mime_hint = value.get("mime_type") or value.get("content_type") or value.get("format")
            return _coerce_base64_data(str(value.get("value", "")), stats, mime_hint)
        for key in ("image", "image_base64", "base64", "url", "visualization", "data", "output"):
            if key in value:
                nested = _normalise_image_value(value[key], stats)
                if nested:
                    return nested
        # Some workflows nest payloads inside generic "value" / "content" objects.
        for key in ("value", "content", "payload"):
            if key in value:
                nested = _normalise_image_value(value[key], stats)
                if nested:
                    return nested
    if isinstance(value, list):
        for item in value:
            nested = _normalise_image_value(item, stats)
            if nested:
                return nested
    return None


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
                    raw_id = (
                        item.get("prediction_id")
                        or item.get("predictionId")
                        or item.get("predictionID")
                        or item.get("id")
                        or item.get("uuid")
                        or item.get("annotation_id")
                    )
                    prediction_id = str(raw_id).strip() if raw_id is not None else None
                    metadata = item.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = None
                    predictions.append(
                        Prediction(
                            class_name=str(item.get("class", item.get("class_name", "unknown"))),
                            confidence=float(item.get("confidence", 0.0)),
                            x=_maybe_float(item.get("x")),
                            y=_maybe_float(item.get("y")),
                            width=_maybe_float(item.get("width")),
                            height=_maybe_float(item.get("height")),
                            prediction_id=prediction_id or None,
                            metadata=metadata,
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
    seen: Dict[str, Prediction] = {}
    for pred in predictions:
        label = (pred.class_name or "").strip()
        if not label:
            continue
        if label.lower() in IGNORED_ATTENDANCE_CLASSES:
            continue
        key = label
        if key not in seen or pred.confidence > seen[key].confidence:
            seen[key] = pred
    return [
        {
            "roll_number": roll,
            "status": "present",
            "confidence": round(pred.confidence, 4),
            "prediction_id": pred.prediction_id,
            "metadata": pred.metadata or {},
        }
        for roll, pred in seen.items()
    ]


def create_annotated_image_data(result: Dict[str, Any]) -> Optional[str]:
    def scan(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            # Prioritise well-known visualization keys
            for key, value in node.items():
                if isinstance(key, str) and key.lower() in VISUALIZATION_KEYS:
                    candidate = _normalise_image_value(value)
                    if candidate:
                        return candidate
            # Otherwise dive deeper
            for value in node.values():
                candidate = scan(value)
                if candidate:
                    return candidate
        elif isinstance(node, list):
            for value in node:
                candidate = scan(value)
                if candidate:
                    return candidate
        return None

    return scan(result)


def extract_crops(result: Dict[str, Any]) -> Dict[str, Any]:
    by_prediction: Dict[str, Dict[str, Any]] = {}
    by_label: Dict[str, Dict[str, Any]] = {}
    fallback_entries: List[Dict[str, Any]] = []
    fallback_counter = 0
    stats = {
        "crops_scanned": 0,
        "crops_with_image_field": 0,
        "crops_image_normalized": 0,
    }

    def _coerce_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value = f"{value}"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    def _first_from_keys(container: Optional[Dict[str, Any]], keys: tuple[str, ...]) -> Optional[str]:
        if not isinstance(container, dict):
            return None
        for key in keys:
            if key in container:
                value = _coerce_str(container[key])
                if value:
                    return value
        return None

    def record(
        pred_id: Optional[str],
        label: Optional[str],
        image_value: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        nonlocal fallback_counter
        image = _normalise_image_value(image_value, stats)
        if not image:
            _add_stat(stats, "crops_missing_image")
            return False

        metadata_payload = meta.copy() if meta else {}
        if pred_id and "prediction_id" not in metadata_payload:
            metadata_payload["prediction_id"] = pred_id
        if label:
            metadata_payload.setdefault("roll_number", label)
            metadata_payload.setdefault("label", label)

        payload: Dict[str, Any] = {
            "image": image,
            "label": label or None,
        }
        if pred_id:
            payload["prediction_id"] = pred_id
        if metadata_payload:
            payload["metadata"] = metadata_payload

        stored_direct = False
        if pred_id:
            key = pred_id.strip()
            if key and key not in by_prediction:
                by_prediction[key] = payload
            stored_direct = True
        if label:
            label_key = label.strip()
            if label_key and label_key not in by_label:
                by_label[label_key] = payload
            stored_direct = True

        if not stored_direct:
            fallback_counter += 1
            fallback_key = f"fallback-{fallback_counter}"
            by_prediction.setdefault(fallback_key, payload)
            fallback_entries.append(payload)

        return True

    def collect_from_list(items: List[Any]) -> bool:
        recorded_any = False
        for crop in items:
            if not isinstance(crop, dict):
                continue
            stats["crops_scanned"] += 1
            metadata = crop.get("metadata") if isinstance(crop.get("metadata"), dict) else {}
            pred_id = _first_from_keys(crop, PREDICTION_ID_KEYS) or _first_from_keys(metadata, PREDICTION_ID_KEYS)
            label = _first_from_keys(crop, LABEL_KEYS) or _first_from_keys(metadata, LABEL_KEYS)
            image_value = (
                crop.get("image")
                or crop.get("image_base64")
                or crop.get("base64")
                or crop.get("url")
                or metadata.get("image")
                or crop
            )
            if any(k in crop for k in ("image", "image_base64", "base64", "url")):
                stats["crops_with_image_field"] += 1
            if isinstance(image_value, dict) and image_value.get("type") == "base64":
                _add_stat(stats, "crops_with_type_base64")
            if record(pred_id, label, image_value, metadata):
                stats["crops_image_normalized"] += 1
                recorded_any = True
        return recorded_any

    def scan(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                processed = False
                if isinstance(value, list):
                    if isinstance(key, str) and key.lower() in CROP_KEYS:
                        processed = collect_from_list(value)
                    else:
                        processed = collect_from_list(value)
                if not processed:
                    scan(value)
        elif isinstance(node, list):
            if not collect_from_list(node):
                for item in node:
                    scan(item)

    scan(result)
    return {
        "by_prediction_id": by_prediction,
        "by_label": by_label,
        "fallback": fallback_entries,
        "stats": stats,
    }


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def generate_manual_crops(image_path: Path, predictions: List[Prediction]) -> List[Dict[str, Any]]:
    """Generate inline base64 crops directly from the source image as a fallback.

    This is only used when the workflow response does not provide dynamic crops.
    """

    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError:
        return []

    if not image_path.exists():
        return []

    valid_predictions = [
        pred
        for pred in predictions
        if pred.class_name
        and pred.class_name.lower() not in IGNORED_ATTENDANCE_CLASSES
        and None not in (pred.x, pred.y, pred.width, pred.height)
    ]
    if not valid_predictions:
        return []

    try:
        with Image.open(image_path).convert("RGB") as img:
            width, height = img.size
            fallback_payloads: List[Dict[str, Any]] = []
            for pred in valid_predictions:
                x1 = int((pred.x or 0) - (pred.width or 0) / 2)
                y1 = int((pred.y or 0) - (pred.height or 0) / 2)
                x2 = int((pred.x or 0) + (pred.width or 0) / 2)
                y2 = int((pred.y or 0) + (pred.height or 0) / 2)

                x1 = max(0, min(width, x1))
                y1 = max(0, min(height, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                try:
                    crop = img.crop((x1, y1, x2, y2))
                except Exception:  # noqa: BLE001
                    continue

                buffer = io.BytesIO()
                try:
                    crop.save(buffer, format="JPEG", quality=90)
                except Exception:  # noqa: BLE001
                    continue

                data_uri = _coerce_base64_data(base64.b64encode(buffer.getvalue()).decode("ascii"), None, "image/jpeg")
                if not data_uri:
                    continue

                label = pred.class_name.strip()
                metadata = {
                    "source": "manual_bbox",
                    "confidence": pred.confidence,
                }
                if pred.prediction_id:
                    metadata["prediction_id"] = pred.prediction_id
                fallback_payloads.append(
                    {
                        "image": data_uri,
                        "label": label,
                        "prediction_id": pred.prediction_id,
                        "metadata": metadata,
                    }
                )

            return fallback_payloads
    except Exception:  # noqa: BLE001
        return []

