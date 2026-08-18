from typing import Any

from schemas.analysis_schema import normalize_checklist_results, normalize_states

MODEL_VERSION = "v1.0.3"

YOLO_TRANSPARENCY_BY_CLASS = {
    "clear_pet_bottle": "true",
    "colored_pet_bottle": "false",
}


def _get_yolo_transparency(yolo_class: str) -> str | None:
    """
    Only clear/colored YOLO classes are trusted for transparency.
    Generic pet_bottle means the current model did not classify bottle color,
    so GPT Vision keeps responsibility for isTransparent.
    """
    return YOLO_TRANSPARENCY_BY_CLASS.get(yolo_class)


def build_analysis_result(
    yolo_result: dict[str, Any],
    state_result: dict[str, Any],
) -> dict[str, Any]:
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
    rule_input: dict[str, Any] = {
        "categoryCode": analysis_result["categoryCode"],
        "states": normalize_states({"checklistResults": analysis_result.get("checkItems", [])}),
    }

    if region_code:
        rule_input["regionCode"] = region_code

    return rule_input
