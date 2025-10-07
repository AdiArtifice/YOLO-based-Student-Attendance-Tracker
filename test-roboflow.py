"""Roboflow Local Inference Workflow Runner

Usage (PowerShell examples):
	# 1. (One-time) create & activate a virtual environment (recommended)
	python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1

  # 2. Install dependencies
  pip install --upgrade inference-sdk inference-cli opencv-python requests python-dotenv

  # 3. Start the local inference server in a separate terminal (keeps running):
  inference server start --port 9001

  # 4. Run this script against an image
  python test-roboflow.py --image path/to/image.jpg

  # Or run on every image in a folder
  python test-roboflow.py --folder path/to/images

  # Or live webcam (press q to quit)
  python test-roboflow.py --webcam 0

Environment Variables:
  ROBOFLOW_API_KEY   (optional) If not provided, falls back to hard-coded placeholder.
  ROBOFLOW_API_URL   (optional) Default: http://localhost:9001

Security Note:
  Avoid committing real API keys. Put them in a .env file (ROBOFLOW_API_KEY=...) and it will be loaded automatically.
"""

from __future__ import annotations

import argparse
import csv
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
	from dotenv import load_dotenv  # type: ignore
except ImportError:  # Lightweight fallback if python-dotenv not installed
	def load_dotenv(*_args, **_kwargs):  # noqa: D401
		"""No-op load_dotenv; install python-dotenv for .env support."""
		return False

from inference_sdk import InferenceHTTPClient  # type: ignore


DEFAULT_API_URL = os.environ.get("ROBOFLOW_API_URL", "http://localhost:9001")
HOSTED_FALLBACK_URL = "https://infer.roboflow.com"
DEFAULT_API_KEY = os.environ.get("ROBOFLOW_API_KEY")  # No hard-coded fallback; must be provided via environment or --api-key.
WORKSPACE_NAME = os.environ.get("ROBOFLOW_WORKSPACE", "tray-detection-cfllw")
WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW", "detect-and-classify")


@dataclass
class Prediction:
	"""Generic prediction structure extracted from the workflow result."""

	class_name: str
	confidence: float
	x: float | None = None
	y: float | None = None
	width: float | None = None
	height: float | None = None


def build_client(api_url: str, api_key: str) -> InferenceHTTPClient:
	return InferenceHTTPClient(api_url=api_url, api_key=api_key)


def test_health(api_url: str) -> bool:
	import json
	import urllib.request
	from urllib.error import HTTPError
	try:
		with urllib.request.urlopen(f"{api_url.rstrip('/')}/health", timeout=2) as resp:
			_ = json.loads(resp.read().decode("utf-8"))
		return True
	except HTTPError as e:  # A 404 still indicates the server is reachable
		if e.code == 404:
			return True
		return False
	except Exception:
		return False


def run_workflow_on_image(client: InferenceHTTPClient, image_path: Path) -> Dict[str, Any]:
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")
	return client.run_workflow(
		workspace_name=WORKSPACE_NAME,
		workflow_id=WORKFLOW_ID,
		images={"image": str(image_path)},
	)


def extract_predictions(result: Dict[str, Any]) -> List[Prediction]:
	"""Attempt to normalize predictions from a workflow response.

	Roboflow workflows can embed predictions in various nodes; we scan for dicts
	containing 'predictions'. Adjust this logic per your workflow graph structure.
	"""
	preds: List[Prediction] = []

	def scan(node: Any):
		if isinstance(node, dict):
			if "predictions" in node and isinstance(node["predictions"], list):
				for p in node["predictions"]:
					if not isinstance(p, dict):
						continue
					preds.append(
						Prediction(
							class_name=p.get("class", p.get("class_name", "unknown")),
							confidence=float(p.get("confidence", 0.0)),
							x=p.get("x"),
							y=p.get("y"),
							width=p.get("width"),
							height=p.get("height"),
						)
					)
			for v in node.values():
				scan(v)
		elif isinstance(node, list):
			for item in node:
				scan(item)

	scan(result)
	return preds


def save_predictions_csv(predictions: List[Prediction], output_csv: Path) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["class", "confidence", "x", "y", "width", "height"])
		for p in predictions:
			writer.writerow([p.class_name, f"{p.confidence:.4f}", p.x, p.y, p.width, p.height])


def annotate_image(original: Path, predictions: List[Prediction], annotated_path: Path, crops_dir: Optional[Path] = None) -> None:
	"""Draw bounding boxes (if coordinate data present) and optionally save crops.

	This uses Pillow to avoid requiring OpenCV for still-image paths.
	"""
	from PIL import Image, ImageDraw, ImageFont  # type: ignore

	try:
		img = Image.open(original).convert("RGB")
	except Exception as e:  # noqa: BLE001
		print(f"Could not open image for annotation: {e}")
		return

	draw = ImageDraw.Draw(img)
	# Try a default font; fallback silently if unavailable
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None  # type: ignore

	def measure(label: str) -> tuple[int, int]:
		"""Return text width/height using whatever APIs are available in this Pillow version."""
		if font is None:
			return len(label) * 7, 12
		# Preferred: textbbox (newer Pillow)
		try:
			bbox = draw.textbbox((0, 0), label, font=font)  # type: ignore[attr-defined]
			return bbox[2] - bbox[0], bbox[3] - bbox[1]
		except Exception:
			pass
		# Legacy: textsize (older Pillow) -- may not exist
		if hasattr(draw, "textsize"):
			try:  # type: ignore[attr-defined]
				w, h = draw.textsize(label, font=font)  # type: ignore[attr-defined]
				return int(w), int(h)
			except Exception:
				pass
		# Fallback rough estimate
		return len(label) * 7, 12

	if crops_dir:
		crops_dir.mkdir(parents=True, exist_ok=True)

	width, height = img.size
	for idx, p in enumerate(predictions):
		if None in (p.x, p.y, p.width, p.height):
			continue
		x1 = int(p.x - p.width / 2)
		y1 = int(p.y - p.height / 2)
		x2 = int(p.x + p.width / 2)
		y2 = int(p.y + p.height / 2)
		# Clip to image bounds
		x1c, y1c = max(0, x1), max(0, y1)
		x2c, y2c = min(width - 1, x2), min(height - 1, y2)
		if x2c <= x1c or y2c <= y1c:
			continue
		draw.rectangle([x1c, y1c, x2c, y2c], outline="lime", width=2)
		label = f"{p.class_name}:{p.confidence:.2f}"
		tw, th = measure(label)
		box_top = max(0, y1c - th - 2)
		draw.rectangle([x1c, box_top, x1c + tw + 2, box_top + th + 2], fill="lime")
		if font:
			draw.text((x1c + 1, box_top + 1), label, fill="black", font=font)
		else:
			draw.text((x1c, box_top), label, fill="black")

		if crops_dir:
			try:
				crop = img.crop((x1c, y1c, x2c, y2c))
				crop.save(crops_dir / f"crop_{idx}_{p.class_name}.jpg")
			except Exception as ce:  # noqa: BLE001
				print(f"Failed saving crop {idx}: {ce}")

	annotated_path.parent.mkdir(parents=True, exist_ok=True)
	img.save(annotated_path)
	print(f"Annotated image saved: {annotated_path}")


def process_folder(client: InferenceHTTPClient, folder: Path, output_dir: Path) -> None:
	images = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
	if not images:
		print(f"No images found in {folder}")
		return
	for img in images:
		print(f"Processing {img.name}...")
		result = run_workflow_on_image(client, img)
		preds = extract_predictions(result)
		csv_path = output_dir / f"{img.stem}_predictions.csv"
		save_predictions_csv(preds, csv_path)
		print(f"  -> {len(preds)} predictions saved to {csv_path}")


def process_webcam(client: InferenceHTTPClient, index: int, every_n_frames: int = 5) -> None:
	try:
		import cv2  # type: ignore
	except ModuleNotFoundError:
		raise RuntimeError("OpenCV (cv2) not installed. Install with 'pip install opencv-python' or 'pip install opencv-python-headless'.") from None

	cap = cv2.VideoCapture(index)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open webcam index {index}")
	frame_idx = 0
	print("Press 'q' to quit.")
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Failed to read frame; exiting.")
			break
		display_frame = frame.copy()
		if frame_idx % every_n_frames == 0:
			# Save frame temporarily to send (API currently expects path or base64)
			tmp_path = Path("_temp_frame.jpg")
			cv2.imwrite(str(tmp_path), frame)
			try:
				result = run_workflow_on_image(client, tmp_path)
				preds = extract_predictions(result)
				# Draw boxes if coordinates present
				for p in preds:
					if None not in (p.x, p.y, p.width, p.height):
						x1 = int(p.x - p.width / 2)
						y1 = int(p.y - p.height / 2)
						x2 = int(p.x + p.width / 2)
						y2 = int(p.y + p.height / 2)
						cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
						cv2.putText(
							display_frame,
							f"{p.class_name}:{p.confidence:.2f}",
							(x1, max(0, y1 - 5)),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.5,
							(0, 255, 0),
							1,
							cv2.LINE_AA,
						)
				print(f"Frame {frame_idx}: {len(preds)} predictions")
			except Exception as e:  # noqa: BLE001
				print(f"Inference error: {e}")
			finally:
				if tmp_path.exists():
					try:
						tmp_path.unlink()
					except Exception:
						pass
		cv2.imshow("Roboflow Inference (q to quit)", display_frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
		frame_idx += 1
	cap.release()
	cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Roboflow workflow locally via inference server.")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--image", type=str, help="Path to a single image.")
	group.add_argument("--folder", type=str, help="Path to a folder of images.")
	group.add_argument("--webcam", type=int, help="Webcam index (e.g., 0).")
	parser.add_argument(
		"--api-url",
		type=str,
		default=DEFAULT_API_URL,
		help="Inference server URL or 'auto' to try local then hosted (default: %(default)s)",
	)
	parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key (required; set ROBOFLOW_API_KEY env var or pass explicitly)")
	parser.add_argument("--output", type=str, default="outputs", help="Directory for CSV outputs.")
	parser.add_argument("--every-n-frames", type=int, default=5, help="Run inference every N frames for webcam.")
	parser.add_argument("--save-annotated", action="store_true", help="Save annotated image(s) with bounding boxes.")
	parser.add_argument("--save-crops", action="store_true", help="Save cropped detection regions (requires boxes).")
	parser.add_argument("--json", action="store_true", help="Also dump raw workflow JSON response next to CSV.")
	return parser.parse_args()


def main() -> None:
	load_dotenv()  # Safe even if .env missing
	args = parse_args()
	chosen_api_url = args.api_url
	if args.api_url.lower() == "auto":
		# Try local first, then hosted fallback
		local_url = DEFAULT_API_URL
		if test_health(local_url):
			chosen_api_url = local_url
			print(f"Auto mode: Local server healthy at {local_url}")
		else:
			print("Auto mode: Local server not reachable; falling back to hosted endpoint.")
			chosen_api_url = HOSTED_FALLBACK_URL
	else:
		# If user passed explicit URL, optionally warn if unreachable
		if not test_health(chosen_api_url):
			print(
				f"Warning: Health check failed for {chosen_api_url}. If this was intended (e.g., hosted workflow without /health), proceed anyway."
			)

	if not args.api_key:
		raise SystemExit("No API key supplied. Set ROBOFLOW_API_KEY or pass --api-key <key>.")
	client = build_client(chosen_api_url, args.api_key)
	print(f"Using server: {chosen_api_url}")
	print(f"Workspace: {WORKSPACE_NAME} | Workflow: {WORKFLOW_ID}")

	if args.image:
		img_path = Path(args.image)
		try:
			result = run_workflow_on_image(client, img_path)
		except FileNotFoundError:
			print(
				"ERROR: Image file not found: "
				f"'{img_path}'.\n"
				"Tips: \n"
				"  - If the path has spaces, wrap it in quotes.\n"
				"  - Don't mix relative and absolute forms (avoid patterns like .\\images\\C:\\...).\n"
				f"  - Current working directory: {os.getcwd()}\n"
				"  - Example: python test-roboflow.py --image \".\\images\\my_image.jpg\" --api-url auto\n"
			)
			return
		preds = extract_predictions(result)
		output_dir = Path(args.output)
		output_dir.mkdir(parents=True, exist_ok=True)
		csv_path = output_dir / f"{img_path.stem}_predictions.csv"
		save_predictions_csv(preds, csv_path)
		print(f"{len(preds)} predictions saved to {csv_path}")
		if args.json:
			json_path = output_dir / f"{img_path.stem}_raw.json"
			json_path.write_text(json.dumps(result, indent=2))
			print(f"Raw JSON saved to {json_path}")
		if args.save_annotated and preds:
			annotated_path = output_dir / f"{img_path.stem}_annotated.jpg"
			crops_dir = output_dir / f"{img_path.stem}_crops" if args.save_crops else None
			annotate_image(img_path, preds, annotated_path, crops_dir)
	elif args.folder:
		process_folder(client, Path(args.folder), Path(args.output))
	elif args.webcam is not None:
		process_webcam(client, args.webcam, args.every_n_frames)
	else:  # Should not happen due to mutually exclusive group
		raise SystemExit("Please specify one of --image, --folder, or --webcam")


if __name__ == "__main__":
	main()

