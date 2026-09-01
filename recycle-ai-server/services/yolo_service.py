import io
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

load_dotenv()

CLASS_TO_ITEM = {
    "clear_pet_bottle": {"categoryCode": "PET_BOTTLE"},
    "colored_pet_bottle": {"categoryCode": "PET_BOTTLE"},
    "pet_bottle": {"categoryCode": "PET_BOTTLE"},
    "plastic_container": {"categoryCode": "PLASTIC_CONTAINER"},
    "can": {"categoryCode": "CAN"},
    "glass_bottle": {"categoryCode": "GLASS_BOTTLE"},
    "paper": {"categoryCode": "PAPER"},
    "paper_pack": {"categoryCode": "PAPER_PACK"},
    "vinyl": {"categoryCode": "VINYL"},
    "styrofoam": {"categoryCode": "STYROFOAM"},
}

YOLO_TRANSPARENCY_HINT = {
    "clear_pet_bottle": "true",
    "colored_pet_bottle": "false",
}

DEBUG_MAX_DETECTIONS = 20
NMS_IOU_THRESHOLD = 0.7
SMALL_OBJECT_AREA_RATIO = 0.01

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


def _load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


def _to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _class_to_category_code(yolo_class: str) -> str:
    item_info = CLASS_TO_ITEM.get(yolo_class)
    if item_info:
        return item_info["categoryCode"]

    lowered = yolo_class.lower()
    if "pet" in lowered and "bottle" in lowered:
        return "PET_BOTTLE"

    return "UNKNOWN"


def _bbox_iou(first: list[int], second: list[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    intersection_width = max(0, right - left)
    intersection_height = max(0, bottom - top)
    intersection_area = intersection_width * intersection_height

    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union_area = first_area + second_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def _remove_duplicate_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if all(_bbox_iou(detection["bbox"], kept_detection["bbox"]) < NMS_IOU_THRESHOLD for kept_detection in kept):
            kept.append(detection)
    return kept


def _mark_small_objects(detections: list[dict[str, Any]], image_width: int, image_height: int) -> None:
    image_area = image_width * image_height
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        detection["isSmallYoloObject"] = (bbox_area / image_area) < SMALL_OBJECT_AREA_RATIO if image_area else True


def _add_object_ids(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_detections = sorted(detections, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for object_id, detection in enumerate(sorted_detections, start=1):
        detection["objectId"] = object_id
    return sorted_detections


def _draw_numbered_bboxes(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(18, image.width // 45))
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        object_id = str(detection["objectId"])
        color = (255, 0, 0)

        line_width = max(3, image.width // 300)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = f"#{object_id}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        padding = 6
        label_left = x1
        label_top = max(0, y1 - label_height - padding * 2)
        draw.rectangle(
            (
                label_left,
                label_top,
                label_left + label_width + padding * 2,
                label_top + label_height + padding * 2,
            ),
            fill=color,
        )
        draw.text((label_left + padding, label_top + padding), label, fill=(255, 255, 255), font=font)

    return annotated


def _collect_detections(model: YOLO, image: Image.Image, conf_threshold: float | None = None) -> list[dict[str, Any]]:
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

    if result.boxes is None:
        return detections

    for index in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[index])
        yolo_class = str(model.names[cls_id])
        x1, y1, x2, y2 = result.boxes.xyxy[index].tolist()
        detections.append(
            {
                "categoryCode": _class_to_category_code(yolo_class),
                "yoloClass": yolo_class,
                "classId": cls_id,
                "confidence": round(float(result.boxes.conf[index]), 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            }
        )

    width, height = image.size
    detections = _remove_duplicate_detections(detections)
    _mark_small_objects(detections, width, height)
    return _add_object_ids(detections)


def run_yolo_multi_object(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> tuple[dict[str, Any], bytes, str]:
    """
    Detect all YOLO objects, assign object IDs, and draw numbered bboxes
    on the corrected original image for VLM whole-image analysis.
    """
    model = _get_model()
    image = _load_image(image_bytes)
    width, height = image.size
    detections = _collect_detections(model, image)
    annotated_image = _draw_numbered_bboxes(image, detections)

    yolo_result = {
        "imageSize": {"width": width, "height": height},
        "objects": detections,
        "objectCount": len(detections),
        "hasBbox": len(detections) > 0,
    }

    return yolo_result, _to_jpeg_bytes(annotated_image), "image/jpeg"


def run_yolo_and_crop(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> tuple[dict[str, Any], bytes, str]:
    """
    Legacy single-object path for /analyze-ai2.
    """
    model = _get_model()
    image = _load_image(image_bytes)
    width, height = image.size
    detections = _collect_detections(model, image)

    if not detections:
        raise ValueError("No recyclable item was detected.")

    best_detection = max(detections, key=lambda item: item["confidence"])
    x1, y1, x2, y2 = best_detection["bbox"]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid YOLO bbox: {best_detection['bbox']}")

    crop = image.crop((x1, y1, x2, y2))
    yolo_result = {
        "itemCode": best_detection["categoryCode"],
        "categoryCode": best_detection["categoryCode"],
        "yoloClass": best_detection["yoloClass"],
        "confidence": best_detection["confidence"],
        "bbox": best_detection["bbox"],
    }

    return yolo_result, _to_jpeg_bytes(crop), "image/jpeg"


def debug_yolo_detection(
    image_bytes: bytes,
    conf_threshold: float | None = None,
) -> dict[str, Any]:
    model = _get_model()
    image = _load_image(image_bytes)
    detections = _collect_detections(model, image, conf_threshold=conf_threshold)
    shown_detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)[:DEBUG_MAX_DETECTIONS]

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
        "confThreshold": _get_conf_threshold() if conf_threshold is None or conf_threshold <= 0 else conf_threshold,
        "detectionCount": len(detections),
        "shownDetectionCount": len(shown_detections),
        "maxShownDetections": DEBUG_MAX_DETECTIONS,
        "detections": [
            {
                **detection,
                "className": detection["yoloClass"],
                "transparencyHint": YOLO_TRANSPARENCY_HINT.get(detection["yoloClass"]),
                "transparencySource": (
                    "yolo" if detection["yoloClass"] in YOLO_TRANSPARENCY_HINT else "gpt_fallback"
                ),
            }
            for detection in shown_detections
        ],
    }
