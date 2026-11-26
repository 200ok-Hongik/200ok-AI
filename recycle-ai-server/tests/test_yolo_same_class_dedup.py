# (2026-11-24 추가) 같은 품목 박스가 다른 박스 안에 거의 통째로 들어가면 하나로 합치는 규칙 테스트
# - 같은 품목 + 작은 박스의 85% 이상이 큰 박스 안이면 확신도 높은 박스만 남는지
# - 다른 품목이면(컵 안 빨대 등) 겹쳐도 둘 다 남는지
# - 조금만 겹치는 같은 품목(옆에 나란히 놓인 두 물체)은 둘 다 남는지
# - 기준값 0이면 규칙이 꺼지는지 (되돌리기용)
from services import yolo_service as y


def _det(code, conf, bbox):
    return {"categoryCode": code, "confidence": conf, "bbox": bbox}


def test_same_class_contained_box_removed():
    big = _det("PAPER", 0.9, [0, 0, 100, 100])
    inner = _det("PAPER", 0.4, [10, 10, 60, 60])  # IoU 0.25 라 기존 기준으로는 안 지워졌음
    assert y._remove_duplicate_detections([inner, big]) == [big]


def test_different_class_contained_box_kept():
    cup = _det("PLASTIC", 0.9, [0, 0, 100, 100])
    straw = _det("VINYL", 0.6, [40, 0, 50, 90])
    assert len(y._remove_duplicate_detections([cup, straw])) == 2


def test_same_class_side_by_side_kept():
    left = _det("CAN", 0.9, [0, 0, 100, 100])
    right = _det("CAN", 0.8, [80, 0, 180, 100])  # 작은 쪽 기준 20%만 겹침
    assert len(y._remove_duplicate_detections([left, right])) == 2


def test_threshold_zero_disables(monkeypatch):
    monkeypatch.setattr(y, "SAME_CLASS_CONTAINMENT_THRESHOLD", 0.0)
    big = _det("PAPER", 0.9, [0, 0, 100, 100])
    inner = _det("PAPER", 0.4, [10, 10, 60, 60])
    assert len(y._remove_duplicate_detections([inner, big])) == 2
