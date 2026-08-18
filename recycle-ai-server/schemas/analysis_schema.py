from typing import Any

STATE_FIELDS = (
    "isTransparent",
    "hasLabel",
    "isEmpty",
    "hasCap",
    "isContaminated",
    "isCrushed",
)

CHECKLIST_ID_TO_FIELD = {
    1: "isTransparent",
    2: "hasLabel",
    3: "isEmpty",
    4: "hasCap",
    5: "isContaminated",
    6: "isCrushed",
}

FIELD_TO_CHECKLIST_ID = {field: checklist_id for checklist_id, field in CHECKLIST_ID_TO_FIELD.items()}


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


def normalize_states(raw: dict[str, Any]) -> dict[str, bool | None]:
    if "checklistResults" in raw:
        states: dict[str, bool | None] = {field: None for field in STATE_FIELDS}
        for item in raw.get("checklistResults", []):
            if not isinstance(item, dict):
                continue
            checklist_id = _normalize_checklist_id(item.get("checklistId"))
            field = CHECKLIST_ID_TO_FIELD.get(checklist_id) or item.get("checkItemName")
            if field in states:
                states[field] = _normalize_bool(item.get("statusValue"))
        return states

    return {field: _normalize_bool(raw.get(field)) for field in STATE_FIELDS}


def normalize_checklist_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    raw_by_field: dict[str, dict[str, Any]] = {}

    if isinstance(raw.get("checklistResults"), list):
        for item in raw["checklistResults"]:
            if not isinstance(item, dict):
                continue

            checklist_id = _normalize_checklist_id(item.get("checklistId"))
            field = CHECKLIST_ID_TO_FIELD.get(checklist_id) or item.get("checkItemName")
            if field in STATE_FIELDS:
                raw_by_field[field] = item
    else:
        for field in STATE_FIELDS:
            raw_by_field[field] = {
                "statusValue": raw.get(field),
                "confidence": raw.get(f"{field}Confidence"),
            }

    checklist_results: list[dict[str, Any]] = []
    for field in STATE_FIELDS:
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
