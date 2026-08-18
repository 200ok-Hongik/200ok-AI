import json
from typing import Any


def parse_yolo_json(yolo_json: str) -> dict[str, Any]:
    yolo_result = json.loads(yolo_json)
    validate_yolo_result(yolo_result)
    return yolo_result


def validate_yolo_result(yolo_result: dict[str, Any]) -> None:
    required = ("itemCode", "bbox", "confidence")
    missing = [field for field in required if field not in yolo_result]

    if missing:
        raise ValueError(f"YOLO result missing required fields: {', '.join(missing)}")

    bbox = yolo_result["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("YOLO result 'bbox' must be a list of 4 numbers.")

    if not all(isinstance(value, (int, float)) for value in bbox):
        raise ValueError("YOLO result 'bbox' values must be numbers.")

    confidence = yolo_result["confidence"]
    if not isinstance(confidence, (int, float)):
        raise ValueError("YOLO result 'confidence' must be a number.")
