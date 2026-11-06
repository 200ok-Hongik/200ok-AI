# 2026-11-06: GPT 응답이 어긋날 때 물체가 사라지거나 백엔드가 모르는 품목 코드가 나가는 문제의 재현·회귀 테스트.
# - 품목 코드 대소문자·지어낸 코드: 백엔드는 itemCode를 품목 테이블에서 찾고 없으면 분석 전체를 실패시킴
#   (200ok-BE AiAnalysisServiceImpl.analyze) → AI 서버가 8품목 대문자 또는 UNKNOWN만 내보내야 함.
# - GPT가 YOLO 물체를 답에서 빼먹으면 그 물체가 결과에서 조용히 사라졌음 → YOLO 판단으로 남기고 확인 필요 표시.
# - GPT가 YOLO에 없는 번호를 지어내면 bbox 없는 물체가 나가 백엔드 검증(bbox 필수)에서 분석 전체가 실패했음 → 제외.
# GPT·YOLO는 실제로 호출하지 않고, 두 결과 dict를 직접 만들어 서버의 후처리 함수에 넣는다.
import pytest

from services.openai_vision_service import _expand_compact_result
from services.result_builder import build_multi_object_analysis_result, should_reanalyze_high

IMAGE_SIZE = {"width": 1000, "height": 1000}
EIGHT_ITEMS = {
    "CLEAR_PET_BOTTLE", "PLASTIC", "CAN", "GLASS_BOTTLE", "VINYL", "PAPER", "PAPER_PACK", "STYROFOAM", "UNKNOWN",
}


def yolo_result(*codes: str) -> dict:
    objects = [
        {
            "objectId": index + 1,
            "categoryCode": code,
            "yoloClass": code,
            "bbox": [index * 200, 0, index * 200 + 100, 100],
            "confidence": 0.8,
        }
        for index, code in enumerate(codes)
    ]
    return {"objects": objects, "imageSize": IMAGE_SIZE, "objectCount": len(objects)}


def gpt_object(object_id: int, code: str | None, states: dict | None = None) -> dict:
    item = {"objectId": object_id, "categoryConfidence": 0.9, "states": states or {}, "flags": []}
    if code is not None:
        item["categoryCode"] = code
    return item


def run_pipeline(yolo: dict, objects: list, missed: list | None = None):
    """main.py /analyze 와 같은 순서: GPT 응답 펼치기 → 재분석 판단 → 결과 생성 (low 결과 기준)."""
    raw = {"imageQuality": "good", "objects": objects, "missedObjects": missed or []}
    vlm = _expand_compact_result(raw, yolo["objects"])
    use_high, reasons = should_reanalyze_high(yolo, vlm)
    result = build_multi_object_analysis_result(yolo, vlm, "low", False, reasons)
    return result, use_high


def by_id(result: dict) -> dict:
    return {item["objectId"]: item for item in result["objects"]}


# ── 품목 코드 대소문자·지어낸 코드 ──

@pytest.mark.parametrize(
    "gpt_code, expected",
    [
        ("can", "CAN"),
        (" Can ", "CAN"),
        ("unknown", "UNKNOWN"),
        ("BATTERY", "UNKNOWN"),
        ("건전지", "UNKNOWN"),
    ],
)
def test_YOLO물체의_GPT품목코드는_8품목_대문자나_UNKNOWN으로만_나간다(gpt_code, expected):
    result, _ = run_pipeline(yolo_result("CAN"), [gpt_object(1, gpt_code)])
    item = by_id(result)["object_1"]
    assert item["vlm"]["itemCode"] == expected
    assert item["vlm"]["itemCode"] in EIGHT_ITEMS


def test_GPT가_소문자로_같은_품목을_답하면_불일치로_보지_않고_재분석하지_않는다():
    states = {"isContaminated": False, "isCrushed": False}
    result, use_high = run_pipeline(yolo_result("CAN"), [gpt_object(1, "can", states)])
    item = by_id(result)["object_1"]
    assert use_high is False
    assert item["vlm"]["yoloItemMatches"] is True
    # CAN 품목의 상태 항목으로 판단돼야 함 (지금까지는 모르는 코드라 페트병 6항목으로 나갔음)
    assert set(item["vlm"]["states"]) == {"isEmpty", "isContaminated", "isCrushed"}


@pytest.mark.parametrize("gpt_code, expected", [("paper", "PAPER"), ("Battery", "UNKNOWN")])
def test_GPT가_새로_찾은_물체의_품목코드도_대소문자를_정리한다(gpt_code, expected):
    missed = [{"objectName": "물체", "categoryCode": gpt_code, "categoryConfidence": 0.8, "bbox": [500, 500, 600, 600], "states": {}}]
    result, _ = run_pipeline(yolo_result(), [], missed)
    assert by_id(result)["object_1"]["vlm"]["itemCode"] == expected


def test_GPT가_품목코드를_비우면_지금처럼_YOLO품목을_쓴다():
    result, _ = run_pipeline(yolo_result("CAN"), [gpt_object(1, None)])
    assert by_id(result)["object_1"]["vlm"]["itemCode"] == "CAN"


# ── GPT가 YOLO 물체를 빼먹은 경우 ──

def test_GPT가_YOLO물체를_빼먹어도_결과에서_사라지지_않는다():
    yolo = yolo_result("CAN", "PAPER")
    result, _ = run_pipeline(yolo, [gpt_object(1, "CAN", {"isContaminated": False, "isCrushed": False})])
    objects = by_id(result)
    assert list(objects) == ["object_1", "object_2"]  # YOLO 번호 순서 유지

    restored = objects["object_2"]
    assert restored["detectedBy"] == "YOLO"
    assert restored["bbox"] == {"xMin": 200, "yMin": 0, "xMax": 300, "yMax": 100}
    assert restored["vlm"]["itemCode"] == "PAPER"  # GPT 판단이 없으니 YOLO 품목
    assert set(restored["vlm"]["states"]) == {"isContaminated", "isWet", "hasTape", "isFlattened"}
    assert all(value is None for value in restored["vlm"]["states"].values())
    assert restored["vlm"]["needsReview"] is True
    assert "VLM_RESULT_MISSING" in {reason["code"] for reason in restored["vlm"]["reviewReasons"]}


def test_GPT가_YOLO물체를_전부_빼먹어도_YOLO결과는_남고_추가물체_번호가_겹치지_않는다():
    missed = [{"objectName": "빨대", "categoryCode": "UNKNOWN", "categoryConfidence": 0.7, "bbox": [500, 500, 600, 900], "states": {}}]
    result, _ = run_pipeline(yolo_result("CAN", "PAPER"), [], missed)
    objects = by_id(result)
    assert list(objects) == ["object_1", "object_2", "object_3"]
    assert objects["object_3"]["detectedBy"] == "VLM"


# ── GPT가 YOLO에 없는 번호를 지어낸 경우 ──

def test_GPT가_YOLO에_없는_번호를_지어내면_bbox없는_물체를_내보내지_않는다():
    result, _ = run_pipeline(yolo_result("CAN"), [gpt_object(1, "CAN"), gpt_object(7, "PAPER")])
    assert list(by_id(result)) == ["object_1"]
    # 백엔드 AiModelResponse.validate 는 objects[].bbox 가 없으면 분석 전체를 실패시킴
    assert all(item["bbox"] is not None for item in result["objects"])


# ── 회귀: 정상 응답은 지금과 같다 ──

def test_정상응답은_물체수_품목_재분석여부가_바뀌지_않는다():
    yolo = yolo_result("CAN", "PAPER")
    objects = [
        gpt_object(1, "CAN", {"isContaminated": False, "isCrushed": False}),
        gpt_object(2, "PAPER", {"isContaminated": False, "isWet": False, "hasTape": False, "isFlattened": True}),
    ]
    result, use_high = run_pipeline(yolo, objects)
    assert use_high is False
    assert [(item["objectId"], item["vlm"]["itemCode"]) for item in result["objects"]] == [
        ("object_1", "CAN"),
        ("object_2", "PAPER"),
    ]
    assert all(item["vlm"]["needsReview"] is False for item in result["objects"])
