# 2026-10-09: /analyze 응답의 물체 요약(objectId/탐지 주체/YOLO 클래스/GPT 품목·이름/bbox)과 HIGH 재분석 이유를
# 서버 로그에 남기도록 _log_analysis_summary 추가 - 응답 본문이 로그에 없어 앱 화면 없이는 중복·오인식 원인을
# 알 수 없었기 때문. 사진·상태값·키 같은 민감/대용량 정보는 남기지 않음.
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import APITimeoutError

from services.openai_vision_service import analyze_numbered_original_image, analyze_pet_bottle_image
from services.result_builder import (
    build_analysis_result,
    build_multi_object_analysis_result,
    build_multi_rule_input,
    build_rule_input,
    should_reanalyze_high,
)
from services.yolo_parser import parse_yolo_json
from services.yolo_service import debug_yolo_detection, run_yolo_and_crop, run_yolo_multi_object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recycle_ai_server")

app = FastAPI(
    title="Recycle AI Server",
    description="Original image -> YOLO numbered bboxes -> GPT Vision object checklist JSON",
    version="1.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def _log_analysis_summary(analysis_result: dict[str, Any], high_reasons: list[dict[str, Any]]) -> None:
    """물체마다 한 줄씩 요약을 남긴다 - YOLO 중복인지 GPT 추가 탐지인지 로그만으로 구분하기 위함."""
    if high_reasons:
        logger.info("analysis high_reasons=%s", sorted({r.get("code") for r in high_reasons if r.get("code")}))
    for item in analysis_result.get("objects", []):
        yolo = item.get("yolo") or {}
        vlm = item.get("vlm") or {}
        logger.info(
            "analysis object id=%s by=%s yolo=%s(%s) vlm=%s name=%s bbox=%s",
            item.get("objectId"),
            item.get("detectedBy"),
            yolo.get("className"),
            yolo.get("confidence"),
            vlm.get("itemCode"),
            vlm.get("objectName"),
            item.get("bbox"),
        )
    for extra in analysis_result.get("additionalObjects", []):
        logger.info("analysis additional item=%s name=%s bbox=None", extra.get("itemCode"), extra.get("objectName"))


@app.post("/analyze")
async def analyze_full_pipeline(
    image: UploadFile = File(..., description="Original image file"),
    rule_input_only: bool = Form(False, description="Return compact rule input only when true"),
    region_code: str | None = Form(None, description="Optional region code. Example: SEOUL_MAPO_GU"),
    image_id: str | None = Form(None, description="Optional image id from backend storage. Example: image_001"),
) -> dict[str, Any]:
    total_started_at = time.perf_counter()
    try:
        logger.info("analysis request started image_id=%s filename=%s", image_id, image.filename)
        read_started_at = time.perf_counter()
        image_bytes = await image.read()
        logger.info("analysis image_read_ms=%s size_bytes=%s", _elapsed_ms(read_started_at), len(image_bytes))

        if not image_bytes:
            raise ValueError("image file is empty.")

        yolo_started_at = time.perf_counter()
        yolo_result, annotated_image_bytes, annotated_mime_type = run_yolo_multi_object(
            image_bytes=image_bytes,
            mime_type=image.content_type or "image/jpeg",
        )
        logger.info(
            "analysis yolo_ms=%s object_count=%s",
            _elapsed_ms(yolo_started_at),
            yolo_result.get("objectCount"),
        )

        low_started_at = time.perf_counter()
        low_vlm_result = analyze_numbered_original_image(
            image_bytes=annotated_image_bytes,
            yolo_objects=yolo_result["objects"],
            mime_type=annotated_mime_type,
            detail="low",
        )
        logger.info("analysis vlm_low_ms=%s", _elapsed_ms(low_started_at))

        should_use_high, low_review_reasons = should_reanalyze_high(yolo_result, low_vlm_result)
        final_vlm_result = low_vlm_result
        detail_used = "low"

        if should_use_high:
            high_started_at = time.perf_counter()
            final_vlm_result = analyze_numbered_original_image(
                image_bytes=annotated_image_bytes,
                yolo_objects=yolo_result["objects"],
                mime_type=annotated_mime_type,
                detail="high",
            )
            detail_used = "high"
            logger.info("analysis vlm_high_ms=%s", _elapsed_ms(high_started_at))

        build_started_at = time.perf_counter()
        analysis_result = build_multi_object_analysis_result(
            yolo_result=yolo_result,
            vlm_result=final_vlm_result,
            detail_used=detail_used,
            reanalyzed_with_high=should_use_high,
            low_review_reasons=low_review_reasons,
            image_id=image_id,
        )
        logger.info("analysis build_result_ms=%s", _elapsed_ms(build_started_at))
        _log_analysis_summary(analysis_result, low_review_reasons)

        if rule_input_only:
            logger.info("analysis completed total_ms=%s response=v2", _elapsed_ms(total_started_at))
            return build_multi_rule_input(analysis_result, region_code=region_code)

        logger.info("analysis completed total_ms=%s response=v1", _elapsed_ms(total_started_at))
        return analysis_result

    except APITimeoutError as exc:
        logger.exception("analysis openai_timeout total_ms=%s", _elapsed_ms(total_started_at))
        raise HTTPException(status_code=504, detail="AI analysis timed out while calling GPT Vision.") from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("analysis failed total_ms=%s", _elapsed_ms(total_started_at))
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
