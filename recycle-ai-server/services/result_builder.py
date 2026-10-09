# 2026-10-09: YOLO가 못 찾고 GPT만 찾은 물체(missedObjects - 종이컵 안의 빨대, 학습 안 된 품목 등)를
# additionalObjects로 따로 빼지 않고 objects에 object_N 번호를 이어 붙여 넣도록 변경함(_promote_missed_objects).
# 8품목 밖이면 itemCode=UNKNOWN + GPT가 본 이름(objectName)을 같이 보내서 사용자가 품목을 고르게 함.
# bbox가 없거나 비정상인 것만 additionalObjects에 남김 - 백엔드 검증이 objects[].bbox를 필수로 요구하기 때문.
# /finalize 도입에 맞춰 additionalObjects를 V1 스펙(itemCode/confidence/bbox/states)으로 정규화하는
# _normalize_additional_objects를 추가함 (기존엔 missedObjects를 형식 검증 없이 그대로 내보냈음).
from typing import Any

from schemas.analysis_schema import CATEGORY_STATE_FIELDS, normalize_checklist_results, normalize_states

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
    # 2026-09-27 8클래스 통합 모델의 무색 페트병 클래스명(소문자로 조회). 무색 클래스이므로 투명=true.
    "clear_pet_bottle": "true",
    "pet_bottle_colored": "false",
}


def _get_yolo_transparency(yolo_class: str) -> str | None:
    """
    Only clear/colored YOLO classes are trusted for transparency.
    Generic pet_bottle means GPT Vision keeps responsibility for isTransparent.
    """
    # 2026-09-27 통합 모델은 대문자 클래스명을 내보내므로 소문자로 통일해서 조회함
    return YOLO_TRANSPARENCY_BY_CLASS.get(yolo_class.lower())


def build_analysis_result(
    yolo_result: dict[str, Any],
    state_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Legacy V1.0.3 single-crop result for /analyze-ai2 compatibility.
    """
    # 기본값 PLASTIC: "규칙 수집" 최종안 기준 무색 투명 페트병(CLEAR_PET_BOTTLE)만 별도배출
    # 대상이고 나머지는 전부 일반 플라스틱류라, 뭔지 모를 때 PLASTIC이 더 안전한 기본값임
    # (반대로 하면 유색 병을 별도배출 대상이라고 잘못 안내하게 됨).
    item_code = str(yolo_result.get("itemCode") or yolo_result.get("categoryCode") or "PLASTIC")
    category_confidence = float(yolo_result.get("confidence") or state_result.get("categoryConfidence") or 0.0)
    yolo_class = str(yolo_result.get("yoloClass") or "")
    yolo_transparency = _get_yolo_transparency(yolo_class)

    check_items = []
    for item in normalize_checklist_results(state_result, category_code=item_code):
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
        "states": normalize_states(
            {"checklistResults": analysis_result.get("checkItems", [])},
            category_code=analysis_result["categoryCode"],
        ),
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


def _states_from_checklist(
    checklist_results: list[dict[str, Any]], category_code: str | None = None
) -> dict[str, bool | None]:
    return normalize_states({"checklistResults": checklist_results}, category_code=category_code)


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
        item_code = str(item.get("categoryCode") or "UNKNOWN")
        # 8품목 밖의 코드를 GPT가 지어내면 상태 필드/규칙 매칭이 안 되므로 UNKNOWN으로 묶어 사용자 선택에 맡김
        if item_code not in CATEGORY_STATE_FIELDS:
            item_code = "UNKNOWN"
        normalized.append(
            {
                "itemCode": item_code,
                "objectName": item.get("objectName"),
                "confidence": float(confidence_raw) if confidence_raw is not None else None,
                "bbox": _bbox_to_object(item.get("bbox")),
                "states": _states_from_checklist(
                    normalize_checklist_results(item, category_code=item_code), category_code=item_code
                ),
            }
        )

    return normalized


def _clamp_valid_bbox(bbox: dict[str, int] | None, image_size: dict[str, Any]) -> dict[str, int] | None:
    """GPT가 추정한 bbox를 이미지 범위로 자르고, 넓이가 없으면 None (백엔드 bbox 검증 통과용)."""
    if not bbox:
        return None
    width = image_size.get("width")
    height = image_size.get("height")
    x_min, y_min = max(0, bbox["xMin"]), max(0, bbox["yMin"])
    x_max, y_max = bbox["xMax"], bbox["yMax"]
    # GPT 좌표는 추정치라 이미지 밖으로 나갈 수 있어서 실제 이미지 크기로 자름
    if isinstance(width, int):
        x_max = min(x_max, width)
    if isinstance(height, int):
        y_max = min(y_max, height)
    if x_max <= x_min or y_max <= y_min:
        return None
    return {"xMin": x_min, "yMin": y_min, "xMax": x_max, "yMax": y_max}


def _promote_missed_objects(
    missed_objects: list[dict[str, Any]],
    next_number: int,
    image_size: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    정규화된 missedObjects 중 bbox가 있는 것을 objects 항목(object_N)으로 바꾼다.
    반환: (objects에 이어 붙일 항목들, bbox가 없어 additionalObjects에 남길 항목들)
    """
    promoted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    for missed in missed_objects:
        bbox = _clamp_valid_bbox(missed.get("bbox"), image_size)
        if bbox is None:
            remaining.append(missed)
            continue

        object_id = f"object_{next_number}"
        next_number += 1
        # 품목을 모르면 판단할 상태 필드도 없음 - 기본(페트병) 필드 6개가 전부 null로 붙는 걸 막고 빈 상태로 보냄.
        # 사용자가 품목을 고른 뒤 그 품목의 상태를 채운다.
        states = {} if missed["itemCode"] == "UNKNOWN" else (missed.get("states") or {})

        # YOLO 검증 없이 GPT 단독 판단이라 품목이 무엇이든 항상 사용자 확인 대상으로 둔다
        review_reasons = [_reason("ADDITIONAL_OBJECT_FOUND", object_id)]
        if missed["itemCode"] == "UNKNOWN":
            review_reasons.append(_reason("UNKNOWN_ITEM", object_id))
        review_reasons += [
            _reason("REQUIRED_STATE_NULL", object_id, field) for field, value in states.items() if value is None
        ]

        promoted.append(
            {
                "objectId": object_id,
                "bbox": bbox,
                "detectedBy": "VLM",
                "yolo": {"itemCode": None, "className": None, "confidence": None},
                "vlm": {
                    "itemCode": missed["itemCode"],
                    "objectName": missed.get("objectName"),
                    "confidence": missed.get("confidence"),
                    "states": states,
                    "yoloItemMatches": None,
                    "multipleObjectsInBox": None,
                    "hasUnlabeledNearbyObject": None,
                    "isTooSmallOrBlurred": None,
                    "needsReview": True,
                    "reviewReasons": _dedupe_reasons(review_reasons),
                },
                "review": {"status": REVIEW_STATUS_PENDING},
                "finalResult": None,
            }
        )

    return promoted, remaining


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

        checklist_category_code = vlm_item_code or yolo_item_code
        for checklist_item in normalize_checklist_results(vlm_object, category_code=checklist_category_code):
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
        checklist_results = normalize_checklist_results(vlm_object, category_code=item_code)
        yolo_transparency = _get_yolo_transparency(str(yolo_object.get("yoloClass") or ""))

        # CLEAR_PET_BOTTLE(무색 투명 페트병)과 PLASTIC(유색 페트병 포함 일반 플라스틱) 둘 다
        # YOLO clear/colored 판정으로 isTransparent를 신뢰할 수 있어서 둘 다 검사함.
        if yolo_transparency is not None and item_code in ("CLEAR_PET_BOTTLE", "PLASTIC"):
            for checklist_item in checklist_results:
                if checklist_item["checkItemName"] == "isTransparent":
                    checklist_item["statusValue"] = yolo_transparency
                    checklist_item["confidence"] = yolo_object.get("confidence")

        object_review_reasons = _build_object_review_reasons(object_id, checklist_results, vlm_object)

        output_objects.append(
            {
                "objectId": normalized_object_id,
                "bbox": _bbox_to_object(yolo_object.get("bbox")),
                "detectedBy": "YOLO",
                "yolo": {
                    "itemCode": yolo_object.get("categoryCode"),
                    "className": yolo_object.get("yoloClass"),
                    "confidence": yolo_object.get("confidence"),
                },
                "vlm": {
                    "itemCode": item_code,
                    "confidence": round(item_confidence, 3),
                    "states": _states_from_checklist(checklist_results, category_code=item_code),
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

    # YOLO 번호 다음부터 이어서 매겨야 object_N이 겹치지 않음 (YOLO가 0개면 GPT가 찾은 물체가 object_1부터)
    used_numbers = [n for n in (_parse_object_id(o["objectId"]) for o in output_objects) if n is not None]
    promoted_objects, additional_objects = _promote_missed_objects(
        _normalize_additional_objects(vlm_result.get("missedObjects")),
        next_number=max(used_numbers, default=0) + 1,
        image_size=yolo_result.get("imageSize") or {},
    )
    output_objects.extend(promoted_objects)

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
        "additionalObjects": additional_objects,
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
