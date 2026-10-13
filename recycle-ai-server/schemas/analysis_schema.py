from typing import Any

# 2026-09-10 "규칙 수집" 최종안 기준 - 상태 필드가 품목마다 다름(예: CAN엔 isTransparent가
# 없고, GLASS_BOTTLE은 isCrushed 대신 isBroken, PAPER류는 아예 다른 필드셋을 씀).
# 예전엔 STATE_FIELDS 6개가 전 품목 공통이라고 가정했는데(사실상 PET_BOTTLE 기준이었음),
# 그게 틀렸던 부분 - 품목별로 CATEGORY_STATE_FIELDS에 나눠서 관리함.
#
# PET_BOTTLE/PLASTIC_CONTAINER는 CLEAR_PET_BOTTLE/PLASTIC으로 이름이 바뀜 - 실제 분리배출
# 기준상 무색 투명 페트병(CLEAR_PET_BOTTLE)만 별도배출 대상이고, 유색 페트병과 그 외
# 일반 플라스틱 용기는 전부 PLASTIC 하나로 합쳐서 배출하기 때문 (yolo_service.CLASS_TO_ITEM과
# 동일하게 맞춤).
CATEGORY_STATE_FIELDS: dict[str, tuple[str, ...]] = {
    "CLEAR_PET_BOTTLE": ("isTransparent", "hasLabel", "isEmpty", "hasCap", "isContaminated", "isCrushed"),
    "PLASTIC": ("isTransparent", "hasLabel", "isEmpty", "hasCap", "isContaminated", "isCrushed"),
    "CAN": ("isEmpty", "isContaminated", "isCrushed"),
    "GLASS_BOTTLE": ("hasLabel", "hasCap", "isEmpty", "isContaminated", "isBroken"),
    "VINYL": ("isEmpty", "isContaminated"),
    "PAPER": ("isContaminated", "isWet", "hasTape", "isFlattened"),
    "PAPER_PACK": ("hasCap", "hasStraw", "isEmpty", "isContaminated", "isFlattened"),
    "STYROFOAM": ("hasLabel", "hasTape", "isContaminated"),
}

# category_code를 모르는(아직 카테고리별로 안 나뉜) 기존 호출부를 위한 기본값 -
# 예전 STATE_FIELDS와 동일한 6개(CLEAR_PET_BOTTLE/PLASTIC 공통 필드셋). 기존 동작을
# 안 깨기 위한 하위호환용.
LEGACY_STATE_FIELDS = CATEGORY_STATE_FIELDS["CLEAR_PET_BOTTLE"]

# 체크리스트 ID는 전 품목에 걸쳐 하나의 전역 번호 체계를 씀 (필드 이름 하나당 번호 하나 고정).
# 1~6은 기존에 쓰던 번호 그대로 유지(하위호환), 7번부터 이번에 새로 필요해진 필드.
CHECKLIST_ID_TO_FIELD = {
    1: "isTransparent",
    2: "hasLabel",
    3: "isEmpty",
    4: "hasCap",
    5: "isContaminated",
    6: "isCrushed",
    7: "isBroken",
    8: "isWet",
    9: "hasTape",
    10: "isFlattened",
    11: "hasStraw",
}

FIELD_TO_CHECKLIST_ID = {field: checklist_id for checklist_id, field in CHECKLIST_ID_TO_FIELD.items()}

# 하위호환: STATE_FIELDS를 직접 import해서 쓰던 기존 코드가 있으면 그대로 동작하도록 유지.
STATE_FIELDS = LEGACY_STATE_FIELDS

# 2026-10-13 "규칙 수집" 원문 기준 - 속이 안 보이는 품목은 isEmpty를 VLM이 판단하지 않고 true로 가정함.
# 캔·종이팩은 항상, 플라스틱은 불투명(isTransparent=false)일 때만. 사진으로 속을 볼 수 없는데 GPT가
# 추측하면 빈 캔에도 "비워주세요"를 잘못 안내할 수 있어서 서버에서 덮어씀.
# 무색 페트병/투명 플라스틱/유리병은 속이 보이므로 GPT 판단 그대로 둠(비닐은 원문상 조건부라 이번엔 제외).
ASSUMED_EMPTY_CATEGORIES = frozenset({"CAN", "PAPER_PACK"})
ASSUMED_EMPTY_CONFIDENCE = 1.0  # 판단이 아니라 규칙상 확정값이라 1.0 - 확인 필요(REQUIRED_STATE_NULL) 사유에서도 빠짐


def apply_assumed_empty(category_code: str | None, checklist_results: list[dict[str, Any]]) -> None:
    """속이 안 보이는 품목의 isEmpty를 "true"로 덮어쓴다(checklist_results를 직접 수정).

    PLASTIC의 투명 여부는 YOLO 판정으로 isTransparent를 덮어쓴 뒤에 호출해야 정확함.
    """
    if category_code in ASSUMED_EMPTY_CATEGORIES:
        assume = True
    elif category_code == "PLASTIC":
        # isTransparent가 null(판단 불가)이면 투명일 수도 있으니 가정하지 않고 GPT 판단을 유지
        assume = any(
            item.get("checkItemName") == "isTransparent" and item.get("statusValue") == "false"
            for item in checklist_results
        )
    else:
        assume = False

    if not assume:
        return

    for item in checklist_results:
        if item.get("checkItemName") == "isEmpty":
            item["statusValue"] = "true"
            item["confidence"] = ASSUMED_EMPTY_CONFIDENCE


def _state_fields_for(category_code: str | None) -> tuple[str, ...]:
    if category_code is None:
        return LEGACY_STATE_FIELDS
    return CATEGORY_STATE_FIELDS.get(category_code, LEGACY_STATE_FIELDS)


def _normalize_checklist_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)

    return None


def _normalize_bool(value: Any) -> bool | None:
    if value is True or value is False or value is None:
        return value

    if isinstance(value, str):
        lowered = value.strip().lower().strip('"').strip("'")
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in ("null", "none", ""):
            return None

    return None


def _normalize_status_value(value: Any) -> str | None:
    normalized = _normalize_bool(value)
    if normalized is True:
        return "true"
    if normalized is False:
        return "false"
    return None


def _normalize_confidence(value: Any, status_value: str | None) -> float | None:
    if status_value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    return max(0.0, min(1.0, confidence))


def normalize_states(raw: dict[str, Any], category_code: str | None = None) -> dict[str, bool | None]:
    # category_code를 안 넘기면 예전 그대로(PET_BOTTLE 6필드 기준)로 동작함 - 기존 호출부는
    # 아직 카테고리를 안 넘기므로 지금 당장 동작이 바뀌지 않음.
    state_fields = _state_fields_for(category_code)

    if "checklistResults" in raw:
        states: dict[str, bool | None] = {field: None for field in state_fields}
        for item in raw.get("checklistResults", []):
            if not isinstance(item, dict):
                continue
            checklist_id = _normalize_checklist_id(item.get("checklistId"))
            field = CHECKLIST_ID_TO_FIELD.get(checklist_id) or item.get("checkItemName")
            if field in states:
                states[field] = _normalize_bool(item.get("statusValue"))
        return states

    return {field: _normalize_bool(raw.get(field)) for field in state_fields}


def normalize_checklist_results(raw: dict[str, Any], category_code: str | None = None) -> list[dict[str, Any]]:
    state_fields = _state_fields_for(category_code)
    raw_by_field: dict[str, dict[str, Any]] = {}

    if isinstance(raw.get("checklistResults"), list):
        for item in raw["checklistResults"]:
            if not isinstance(item, dict):
                continue

            checklist_id = _normalize_checklist_id(item.get("checklistId"))
            field = CHECKLIST_ID_TO_FIELD.get(checklist_id) or item.get("checkItemName")
            if field in state_fields:
                raw_by_field[field] = item
    else:
        for field in state_fields:
            raw_by_field[field] = {
                "statusValue": raw.get(field),
                "confidence": raw.get(f"{field}Confidence"),
            }

    checklist_results: list[dict[str, Any]] = []
    for field in state_fields:
        item = raw_by_field.get(field, {})
        status_value = _normalize_status_value(item.get("statusValue"))
        confidence = _normalize_confidence(item.get("confidence"), status_value)

        checklist_results.append(
            {
                "checklistId": FIELD_TO_CHECKLIST_ID[field],
                "checkItemName": field,
                "statusValue": status_value,
                "confidence": confidence,
            }
        )

    return checklist_results
