import json
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from services.openai_vision_service import analyze_pet_bottle_image
from services.result_builder import build_analysis_result, build_rule_input
from services.yolo_parser import parse_yolo_json
from services.yolo_service import debug_yolo_detection, run_yolo_and_crop

app = FastAPI(
    title="Recycle AI Server",
    description="Original image -> YOLO crop -> GPT Vision checklist JSON",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_full_pipeline(
    image: UploadFile = File(..., description="Original image file"),
    rule_input_only: bool = Form(False, description="Return compact rule input only when true"),
    region_code: str | None = Form(None, description="Optional region code. Example: SEOUL_MAPO_GU"),
) -> dict[str, Any]:
    try:
        image_bytes = await image.read()

        if not image_bytes:
            raise ValueError("image file is empty.")

        yolo_result, crop_bytes, crop_mime_type = run_yolo_and_crop(
            image_bytes=image_bytes,
            mime_type=image.content_type or "image/jpeg",
        )

        state_result = analyze_pet_bottle_image(
            image_bytes=crop_bytes,
            mime_type=crop_mime_type,
        )

        analysis_result = build_analysis_result(yolo_result, state_result)

        if rule_input_only:
            return build_rule_input(analysis_result, region_code=region_code)

        return analysis_result

    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze-ai2")
async def analyze_ai2_only(
    crop: UploadFile = File(..., description="YOLO crop image file"),
    yolo_json: str = Form(..., description="YOLO result JSON string"),
    rule_input_only: bool = Form(False, description="Return compact rule input only when true"),
    region_code: str | None = Form(None, description="Optional region code. Example: SEOUL_MAPO_GU"),
) -> dict[str, Any]:
    try:
        yolo_result = parse_yolo_json(yolo_json)
        crop_bytes = await crop.read()

        if not crop_bytes:
            raise ValueError("crop image file is empty.")

        state_result = analyze_pet_bottle_image(
            image_bytes=crop_bytes,
            mime_type=crop.content_type or "image/jpeg",
        )

        analysis_result = build_analysis_result(yolo_result, state_result)

        if rule_input_only:
            return build_rule_input(analysis_result, region_code=region_code)

        return analysis_result

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="yolo_json must be valid JSON.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/debug-yolo")
async def debug_yolo(
    image: UploadFile = File(..., description="Image file for YOLO debug"),
    conf_threshold: float | None = Form(None, description="Optional YOLO confidence threshold"),
) -> dict[str, Any]:
    try:
        image_bytes = await image.read()

        if not image_bytes:
            raise ValueError("image file is empty.")

        return debug_yolo_detection(
            image_bytes=image_bytes,
            conf_threshold=conf_threshold,
        )

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
