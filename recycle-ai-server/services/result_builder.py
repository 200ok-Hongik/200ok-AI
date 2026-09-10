# /finalize 도입에 맞춰 additionalObjects를 V1 스펙(itemCode/confidence/bbox/states)으로 정규화하는
# _normalize_additional_objects를 추가함 (기존엔 missedObjects를 형식 검증 없이 그대로 내보냈음).
from typing import Any

from schemas.analysis_schema import normalize_checklist_results, normalize_states

MODEL_VERSION = "v1.0.3"
MULTI_OBJECT_MODEL_VERSION = "v1.1.0"
ANALYSIS_MODE = "ORIGINAL_NUMBERED_BBOX"
REVIEW_STATUS_PENDING = "PENDING"
REVIEW_STATUS_CONFIRMED = "CONFIRMED"
REVIEW_STATUS_CORRECTED = "CORRECTED"
REVIEW_STATUS_ADDED = "ADDED"
RESULT_SOURCE_VLM_LOW = "VLM_LOW"
RESULT_SOURCE_VLM_HIGH = "VLM_HIGH"
RESULT_SOURCE_USER_CONFIRMED = "USER_CONFIRMED"
RESULT_SOURCE_USER_CORRECTED = "USER_CORRECTED"
RESULT_SOURCE_USER_ADDED = "USER_ADDED"
ALLOWED_RESULT_SOURCES = {
    RESULT_SOURCE_VLM_LOW,
    RESULT_SOURCE_VLM_HIGH,
    RESULT_SOURCE_USER_CONFIRMED,
    RESULT_SOURCE_USER_CORRECTED,
    RESULT_SOURCE_USER_ADDED,
}

# 2026-09-10 재학습된 pet_bottle_clear_colored_v2 모델의 실제 클래스명(data.yaml 기준)은
# "clear_pet_bottle"이 아니라 "pet_bottle_clear"/"pet_bottle_colored" 순서라, 이전에 가정했던
# 이름으로는 항상 매칭 실패해서 isTransparent가 GPT Vision 폴백으로만 갔음 - 실제 학습 결과
# 클래스명에 맞춰 수정함.
YOLO_TRANSPARENCY_BY_CLASS = {
    "pet_bottle_clear": "true",
    "pet_bottle_colored": "false",
}


def _get_yolo_transparency(yolo_class: str) -> str | None:
    """
    Only clear/colored YOLO classes are trusted for transparency.
    Generic pet_bottle means GPT Vision keeps responsibility for isTransparent.
    """
    return YOLO_TRANSPARENCY_BY_CLASS.get(yolo_class)


def build_analysis_result(
    yolo_result: dict[str, Any],
    state_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Legacy V1.0.3 single-crop result for /analyze-ai2 compatibility.
    """
    item_code = str(yolo_result.get("itemCode") or yolo_result.get("categoryCode") or "PET_BOTTLE")
    category_confidence = float(yolo_result.get("confidence") or state_result.get("categoryConfidence") or 0.0)
    yolo_class = str(yolo_result.get("yoloClass") or "")
    yolo_transparency = _get_yolo_transparency(yolo_class)

    check_items = []
    for item in normalize_checklist_results(state_result):
        status_value = item["statusValue"]
        confidence = item["confidence"]

        if item["checkItemName"] == "isTransparent" and yolo_transparency is not None:
            status_value = yolo_transparency
            confidence = category_confidence

        check_items.append(
            {
                "checkItemName": item["checkItemName"],
                "statusValue": status_value,
                "confidence": confidence,
            }
        )

    return {
        "categoryCode": item_code,
        "categoryConfidence": category_confidence,
        "modelVersion": str(state_result.get("modelVersion") or MODEL_VERSION),
        "rawJson": f'{{"itemCode":"{item_code}"}}',
        "checkItems": check_items,
    }


def build_rule_input(analysis_result: dict[str, Any], region_code: str | None = None) -> dict[str, Any]:
    """
    Legacy single-object Rule DB input for /analyze-ai2 compatibility.
    """
    rule_input: dict[str, Any] = {
        "categoryCode": analysis_result["categoryCode"],
        "states": normalize_states({"checklistResults": analysis_result.get("checkItems", [])}),
    }

    if region_code:
        rule_input["regionCode"] = region_code

    return rule_input


def _make_object_id(value: Any) -> str:
    if isinstance(value, str) and value.startswith("object_"):
        return value
    return f"object_{value}"


def _parse_object_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.replace("object_", "", 1) if value.startswith("object_") else value
        if normalized.isdigit():
            return int(normalized)
    return None


def _find_yolo_object(yolo_objects: list[dict[str, Any]], object_id: Any) -> dict[str, Any]:
    numeric_object_id = _parse_object_id(object_id)
    for yolo_object in yolo_objects:
        if yolo_object.get("objectId") == numeric_object_id:
            return yolo_object
    return {}


def _bbox_to_object(bbox: Any) -> dict[str, int] | None:
    if isinstance(bbox, dict):
        return {
            "xMin": int(bbox.get("xMin", 0)),
            "yMin": int(bbox.get("yMin", 0)),
            "xMax": int(bbox.get("xMax", 0)),
            "yMax": int(bbox.get("yMax", 0)),
        }

    if isinstance(bbox, list) and len(bbox) == 4:
        return {
            "xMin": int(bbox[0]),
            "yMin": int(bbox[1]),
            "xMax": int(bbox[2]),
            "yMax": int(bbox[3]),
        }

    return None


def _states_from_checklist(checklist_results: list[dict[str, Any]]) -> dict[str, bool | None]:
    return normalize_states({"checklistResults": checklist_results})


def _normalize_additional_objects(raw_missed_objects: Any) -> list[dict[str, Any]]:
    """
    vlm_result.missedObjects(자유 형식) -> V1 additionalObjects 스펙(itemCode/confidence/bbox/states)으로 정규화.
    """
    if not isinstance(raw_missed_objects, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_missed_objects:
        if not isinstance(item, dict):
            continue

        confidence_raw = item.get("categoryConfidence")
        normalized.append(
            {
                "itemCode": str(item.get("categoryCode") or "UNKNOWN"),
                "confidence": float(confidence_raw) if confidence_raw is not None else None,
                "bbox": _bbox_to_object(item.get("bbox")),
                "states": _states_from_checklist(normalize_checklist_results(item)),
            }
        )

    return normalized


def _result_source_from_level(current_level: Any) -> str:
    if str(current_level).upper() == "HIGH":
        return RESULT_SOURCE_VLM_HIGH
    return RESULT_SOURCE_VLM_LOW


def _normalize_result_source(value: Any, fallback: str) -> str:
    source = str(value or "").upper()
    if source in ALLOWED_RESULT_SOURCES:
        return source
    return fallback


def _reason(code: str, object_id: Any = None, field: str | None = None) -> dict[str, str | None]:
    return {
        "code": code,
        "objectId": _make_object_id(object_id) if object_id is not None else None,
        "field": field,
    }


def _dedupe_reasons(reasons: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    unique_reasons: list[dict[str, str | None]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    for item in reasons:
        key = (item.get("code"), item.get("objectId"), item.get("field"))
        if key not in seen:
            seen.add(key)
            unique_reasons.append(item)

    return unique_reasons


def should_reanalyze_high(
    yolo_result: dict[str, Any],
    vlm_result: dict[str, Any],
) -> tuple[bool, list[dict[str, str | None]]]:
    reasons: list[dict[str, str | None]] = []
    yolo_objects = yolo_result.get("objects", [])

    if not yolo_result.get("hasBbox"):
        reasons.append(_reason("YOLO_BBOX_MISSING"))

    if vlm_result.get("needsReview") is True:
        reasons.append(_reason("VLM_NEEDS_REVIEW"))

    if str(vlm_result.get("imageQuality") or "").lower() in {"poor", "bad", "low", "insufficient"}:
        reasons.append(_reason("POOR_IMAGE_QUALITY"))

    if vlm_result.get("missedObjects"):
        reasons.append(_reason("ADDITIONAL_OBJECT_FOUND"))

    for vlm_object in vlm_result.get("objects", []):
        if not isinstance(vlm_object, dict):
            continue

        object_id = vlm_object.get("objectId")
        yolo_object = _find_yolo_object(yolo_objects, object_id)
        yolo_item_code = yolo_object.get("categoryCode")
        vlm_item_code = vlm_object.get("categoryCode")

        if yolo_item_code and vlm_item_code and yolo_item_code != vlm_item_code:
            reasons.append(_reason("YOLO_VLM_ITEM_MISMATCH", object_id))

        if vlm_object.get("yoloCategoryMatches") is False:
            reasons.append(_reason("YOLO_VLM_ITEM_MISMATCH", object_id))

        for checklist_item in normalize_checklist_results(vlm_object):
            if checklist_item["statusValue"] is None:
                reasons.append(_reason("REQUIRED_STATE_NULL", object_id, checklist_item["checkItemName"]))

        if vlm_object.get("bboxHasMultipleObjects") is True:
            reasons.append(_reason("MULTIPLE_OBJECTS_IN_BBOX", object_id))

        if vlm_object.get("hasUnlabeledNearbyObject") is True:
            reasons.append(_reason("UNLABELED_NEARBY_OBJECT", object_id))

        if vlm_object.get("isTooSmallOrBlurred") is True:
            reasons.append(_reason("OBJECT_TOO_SMALL_OR_BLURRED", object_id))

        if vlm_object.get("needsReview") is True:
            reasons.append(_reason("VLM_NEEDS_REVIEW", object_id))

        if yolo_object.get("isSmallYoloObject") is True:
            reasons.append(_reason("OBJECT_TOO_SMALL_OR_BLURRED", object_id))

    return bool(reasons), _dedupe_reasons(reasons)


def _build_object_review_reasons(
    object_id: Any,
    checklist_results: list[dict[str, Any]],
    vlm_object: dict[str, Any],
) -> list[dict[str, str | None]]:
    reasons: list[dict[str, str | None]] = []

    for checklist_item in checklist_results:
        if checklist_item["statusValue"] is None:
            reasons.append(_reason("REQUIRED_STATE_NULL", object_id, checklist_item["checkItemName"]))

    if vlm_object.get("needsReview") is True:
        reasons.append(_reason("VLM_NEEDS_REVIEW", object_id))
    if vlm_object.get("bboxHasMultipleObjects") is True:
        reasons.append(_reason("MULTIPLE_OBJECTS_IN_BBOX", object_id))
    if vlm_object.get("hasUnlabeledNearbyObject") is True:
        reasons.append(_reason("UNLABELED_NEARBY_OBJECT", object_id))
    if vlm_object.get("isTooSmallOrBlurred") is True:
        reasons.append(_reason("OBJECT_TOO_SMALL_OR_BLURRED", object_id))

    return _dedupe_reasons(reasons)


def build_multi_object_analysis_result(
    yolo_result: dict[str, Any],
    vlm_result: dict[str, Any],
    detail_used: str,
    reanalyzed_with_high: bool,
    low_review_reasons: list[dict[str, str | None]] | None = None,
    image_id: str | None = None,
) -> dict[str, Any]:
    """
    V1 detailed analysis JSON.

    This is returned right after AI analysis. The user has not confirmed or
    corrected the objects yet, so finalResult is kept as None.
    """
    yolo_objects = yolo_result.get("objects", [])
    output_objects: list[dict[str, Any]] = []

    for vlm_object in vlm_result.get("objects", []):
        if not isinstance(vlm_object, dict):
            continue

        object_id = vlm_object.get("objectId")
        yolo_object = _find_yolo_object(yolo_objects, object_id)
        normalized_object_id = _make_object_id(object_id)
        item_code = str(vlm_object.get("categoryCode") or yolo_object.get("categoryCode") or "UNKNOWN")
        item_confidence = float(vlm_object.get("categoryConfidence") or yolo_object.get("confidence") or 0.0)
        checklist_results = normalize_checklist_results(vlm_object)
        yolo_transparency = _get_yolo_transparency(str(yolo_object.get("yoloClass") or ""))

        if yolo_transparency is not None and item_code == "PET_BOTTLE":
            for checklist_item in checklist_results:
                if checklist_item["checkItemName"] == "isTransparent":
                    checklist_item["statusValue"] = yolo_transparency
                    checklist_item["confidence"] = yolo_object.get("confidence")

        object_review_reasons = _build_object_review_reasons(object_id, checklist_results, vlm_object)

        output_objects.append(
            {
                "objectId": normalized_object_id,
                "bbox": _bbox_to_object(yolo_object.get("bbox")),
                "yolo": {
                    "itemCode": yolo_object.get("categoryCode"),
                    "className": yolo_object.get("yoloClass"),
                    "confidence": yolo_object.get("confidence"),
                },
                "vlm": {
                    "itemCode": item_code,
                    "confidence": round(item_confidence, 3),
                    "states": _states_from_checklist(checklist_results),
                    "yoloItemMatches": vlm_object.get("yoloCategoryMatches"),
                    "multipleObjectsInBox": vlm_object.get("bboxHasMultipleObjects"),
                    "hasUnlabeledNearbyObject": vlm_object.get("hasUnlabeledNearbyObject"),
                    "isTooSmallOrBlurred": vlm_object.get("isTooSmallOrBlurred"),
                    "needsReview": bool(object_review_reasons),
                    "reviewReasons": object_review_reasons,
                },
                # AI 분석 직후에는 사용자가 아직 확인하지 않았으므로 PENDING으로 둔다.
                "review": {"status": REVIEW_STATUS_PENDING},
                "finalResult": None,
            }
        )

    image_quality_raw = str(vlm_result.get("imageQuality") or "SUFFICIENT").upper()
    image_quality_status = "SUFFICIENT" if image_quality_raw in {"GOOD", "SUFFICIENT"} else "POOR"
    high_trigger_reasons = _dedupe_reasons(low_review_reasons or [])

    return {
        "image": {
            "imageId": image_id,
            "width": (yolo_result.get("imageSize") or {}).get("width"),
            "height": (yolo_result.get("imageSize") or {}).get("height"),
        },
        "imageQuality": {
            "status": image_quality_status,
            "issues": [] if image_quality_status == "SUFFICIENT" else ["POOR_IMAGE_QUALITY"],
        },
        "analysis": {
            "mode": ANALYSIS_MODE,
            "modelVersion": str(vlm_result.get("modelVersion") or MULTI_OBJECT_MODEL_VERSION),
            "currentLevel": detail_used.upper(),
            "attemptedLevels": ["LOW", "HIGH"] if reanalyzed_with_high else ["LOW"],
            "needsHighAnalysis": reanalyzed_with_high,
            "highTriggerReasons": high_trigger_reasons,
        },
        "objects": output_objects,
        "additionalObjects": _normalize_additional_objects(vlm_result.get("missedObjects")),
    }


def build_multi_rule_input(
    analysis_result: dict[str, Any],
    region_code: str | None = None,
) -> dict[str, Any]:
    """
    V2 compact Rule DB JSON.

    Before user confirmation, this falls back to VLM result and marks the
    source as VLM_LOW or VLM_HIGH. After confirmation, finalResult can be used.
    """
    current_level = (analysis_result.get("analysis") or {}).get("currentLevel", "LOW")
    vlm_result_source = _result_source_from_level(current_level)
    rule_objects = []

    for item in analysis_result.get("objects", []):
        final_result = item.get("finalResult")
        if isinstance(final_result, dict):
            item_code = final_result.get("itemCode")
            states = final_result.get("states")
            result_source = _normalize_result_source(final_result.get("source"), vlm_result_source)
        else:
            vlm = item.get("vlm") or {}
            item_code = vlm.get("itemCode")
            states = vlm.get("states")
            review_status = str((item.get("review") or {}).get("status") or "").upper()
            # 사용자가 수정 없이 확인한 경우는 AI값을 쓰더라도 USER_CONFIRMED로 추적한다.
            result_source = (
                RESULT_SOURCE_USER_CONFIRMED
                if review_status == REVIEW_STATUS_CONFIRMED
                else vlm_result_source
            )

        rule_objects.append(
            {
                "objectId": item.get("objectId"),
                "itemCode": item_code,
                "states": states,
                "bbox": item.get("bbox"),
                "resultSource": result_source,
            }
        )

    rule_input: dict[str, Any] = {
        "imageId": (analysis_result.get("image") or {}).get("imageId"),
        "objects": rule_objects,
    }
    if region_code:
        rule_input["regionCode"] = region_code

    return rule_input
