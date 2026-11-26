# (2026-11-26 추가) high 재분석 때 YOLO 품목 힌트를 빼는 동작 테스트 (실제 GPT 호출 없음)
# - hide_yolo_category=True 면 GPT에 보내는 목록에서 품목 필드가 빠지고 번호·bbox는 남는지, 안내 문구가 붙는지
# - 기본(False)이면 지금처럼 품목 필드가 그대로 가는지
# - 원본 YOLO 리스트는 바뀌지 않는지 (서버 쪽 YOLO·GPT 비교가 원본을 씀)
# - GPT 응답 해석은 원본 YOLO 기준으로 그대로 되는지
import io
import json

from PIL import Image

from services import openai_vision_service as v


class _FakeResponses:
    def __init__(self, sent):
        self.sent = sent

    def create(self, **kwargs):
        self.sent.append(kwargs["input"][0]["content"][0]["text"])
        body = {"imageQuality": "good", "objects": [{"objectId": 1, "categoryCode": "CAN", "categoryConfidence": 0.9, "states": {}}], "missedObjects": []}
        return type("R", (), {"output_text": json.dumps(body)})()


def _run(monkeypatch, hide):
    sent = []
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(v, "_create_client", lambda _key: type("C", (), {"responses": _FakeResponses(sent)})())
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="JPEG")
    yolo = [{"objectId": 1, "bbox": [0, 0, 30, 30], "categoryCode": "PAPER_PACK", "yoloClass": "paper_pack", "classId": 6, "confidence": 0.5}]
    result = v.analyze_numbered_original_image(buf.getvalue(), yolo, detail="high", hide_yolo_category=hide)
    return sent[0], yolo, result


def _sent_objects(prompt):
    line = prompt.split("YOLO 객체 목록:\n", 1)[1].split("\n", 1)[0]
    return json.loads(line)


def test_숨기면_품목필드가_빠지고_번호와_bbox만_간다(monkeypatch):
    prompt, _, _ = _run(monkeypatch, hide=True)
    assert _sent_objects(prompt) == [{"objectId": 1, "bbox": [0, 0, 30, 30]}]
    assert v._HIDDEN_HINT_NOTE in prompt


def test_기본은_지금처럼_품목필드가_그대로_간다(monkeypatch):
    prompt, _, _ = _run(monkeypatch, hide=False)
    assert _sent_objects(prompt)[0]["categoryCode"] == "PAPER_PACK"
    assert v._HIDDEN_HINT_NOTE not in prompt


def test_숨겨도_원본_YOLO리스트는_바뀌지_않는다(monkeypatch):
    _, yolo, _ = _run(monkeypatch, hide=True)
    assert yolo[0]["categoryCode"] == "PAPER_PACK" and yolo[0]["confidence"] == 0.5


def test_숨겨도_GPT응답은_원본_YOLO번호에_맞춰_해석된다(monkeypatch):
    _, _, result = _run(monkeypatch, hide=True)
    assert [(o["objectId"], o["categoryCode"]) for o in result["objects"]] == [(1, "CAN")]
