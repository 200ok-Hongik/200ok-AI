import io
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

load_dotenv()

CLASS_TO_ITEM = {
    "clear_pet_bottle": {
        "itemCode": "PET_BOTTLE",
    },
    "colored_pet_bottle": {
        "itemCode": "PET_BOTTLE",
    },
    "pet_bottle": {
        "itemCode": "PET_BOTTLE",
    }
}

YOLO_TRANSPARENCY_HINT = {
    "clear_pet_bottle": "true",
    "colored_pet_bottle": "false",
}

DEBUG_MAX_DETECTIONS = 20

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model

    if _model is None:
        model_path = os.getenv("YOLO_MODEL_PATH", "models/best.pt")
        resolved_path = Path(model_path)

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"YOLO model file not found: {resolved_path}. "
                "Set YOLO_MODEL_PATH in .env or place best.pt under models/."
            )

        _model = YOLO(str(resolved_path))

    return _model


def _get_img_size() -> int:
    return int(os.getenv("YOLO_IMG_SIZE", "640"))


def _get_conf_threshold() -> float:
    return float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))


def _to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def run_yolo_and_crop(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> tuple[dict[str, Any], bytes, str]:
    """
    Run AI-1 YOLO detection and create a crop image.

    Input is the original image bytes from Spring Boot.
    1. Detect recyclable item with YOLO.
    2. Select the bbox with the highest confidence.
    3. Crop the detected bbox region.
    4. Return yolo_result, crop_bytes, crop_mime_type.
    """
    model = _get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    results = model.predict(
        source=image,
        imgsz=_get_img_size(),
        conf=_get_conf_threshold(),
        save=False,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError("No recyclable item was detected.")

    boxes = result.boxes
    best_idx = int(boxes.conf.argmax())

    cls_id = int(boxes.cls[best_idx])
    confidence = float(boxes.conf[best_idx])
    yolo_class = model.names[cls_id]

    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    bbox = [round(x1), round(y1), round(x2), round(y2)]

    item_info = CLASS_TO_ITEM.get(yolo_class)
    if item_info is None and "pet" in str(yolo_class).lower() and "bottle" in str(yolo_class).lower():
        item_info = {"itemCode": "PET_BOTTLE"}
    if item_info is None:
        item_info = {"itemCode": "UNKNOWN"}

    # Keep crop coordinates inside the original image boundary.
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid YOLO bbox: {bbox}")

    crop = image.crop((x1, y1, x2, y2))

    yolo_result = {
        "itemCode": item_info["itemCode"],
        "yoloClass": yolo_class,
        "confidence": round(confidence, 3),
        "bbox": bbox,
    }

    return yolo_result, _to_jpeg_bytes(crop), "image/jpeg"


def debug_yolo_detection(
    image_bytes: bytes,
    conf_threshold: float | None = None,
) -> dict[str, Any]:
    model = _get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    threshold = _get_conf_threshold() if conf_threshold is None or conf_threshold <= 0 else conf_threshold

    results = model.predict(
        source=image,
        imgsz=_get_img_size(),
        conf=threshold,
        save=False,
        verbose=False,
    )

    result = results[0]
    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        detection_count_before_limit = len(result.boxes)
        for index in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[index])
            class_name = model.names[cls_id]
            x1, y1, x2, y2 = result.boxes.xyxy[index].tolist()
            detections.append(
                {
                    "classId": cls_id,
                    "className": class_name,
                    "transparencyHint": YOLO_TRANSPARENCY_HINT.get(class_name),
                    "transparencySource": "yolo" if class_name in YOLO_TRANSPARENCY_HINT else "gpt_fallback",
                    "confidence": round(float(result.boxes.conf[index]), 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                }
            )
    else:
        detection_count_before_limit = 0

    detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    detections = detections[:DEBUG_MAX_DETECTIONS]

    supports_transparency_classes = any(
        class_name in YOLO_TRANSPARENCY_HINT for class_name in model.names.values()
    )

    return {
        "modelPath": os.getenv("YOLO_MODEL_PATH", "models/best.pt"),
        "modelClasses": model.names,
        "supportsTransparencyClasses": supports_transparency_classes,
        "transparencyRule": {
            "clear_pet_bottle": "isTransparent=true",
            "colored_pet_bottle": "isTransparent=false",
            "pet_bottle": "isTransparent is judged by GPT Vision fallback",
        },
        "imageSize": {"width": image.size[0], "height": image.size[1]},
        "imgsz": _get_img_size(),
        "confThreshold": threshold,
        "detectionCount": detection_count_before_limit,
        "shownDetectionCount": len(detections),
        "maxShownDetections": DEBUG_MAX_DETECTIONS,
        "detections": detections,
    }
