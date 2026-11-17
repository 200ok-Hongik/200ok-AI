# 2026-11-15: /analyze 동시 요청이 한 줄로 서서 처리되던 문제(async def 안에서 동기 YOLO·GPT 호출)의 회귀 테스트.
# - 동시 요청 여러 건이 GPT 대기 시간을 겹쳐서 쓰는지(병렬) 확인
# - 여러 스레드가 같은 YOLO 모델로 predict 를 동시에 하지 않는지(잠금) 확인
# - 스레드로 넘긴 뒤에도 GPT 타임아웃이 그대로 504 로 나가는지 확인
# 실제 GPT·YOLO 는 부르지 않고, 시간만 잡아먹는 가짜 함수로 바꿔 넣는다.
import asyncio
import threading
import time

import httpx
import pytest
from openai import APITimeoutError

import main
from services import yolo_service
from services.openai_vision_service import _expand_compact_result

FAKE_GPT_SECONDS = 0.5
YOLO_OBJECT = {"objectId": 1, "categoryCode": "CAN", "yoloClass": "CAN", "bbox": [0, 0, 100, 100], "confidence": 0.9}


# 가짜 YOLO: 물체 1개를 바로 돌려준다 (실제 모델 로딩 없이 /analyze 흐름만 태우기 위함)
def fake_yolo(image_bytes, mime_type="image/jpeg"):
    yolo = {"imageSize": {"width": 1000, "height": 1000}, "objects": [dict(YOLO_OBJECT)], "objectCount": 1, "hasBbox": True}
    return yolo, image_bytes, "image/jpeg"


# 가짜 GPT: 실제 동기 OpenAI 클라이언트처럼 time.sleep 으로 스레드를 붙잡은 뒤 CAN 으로 답한다
def fake_gpt(image_bytes, yolo_objects, mime_type="image/jpeg", detail="low"):
    time.sleep(FAKE_GPT_SECONDS)
    raw = {
        "imageQuality": "good",
        "objects": [{"objectId": 1, "categoryCode": "CAN", "categoryConfidence": 0.99, "states": {}, "flags": []}],
        "missedObjects": [],
    }
    return _expand_compact_result(raw, yolo_objects)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """main.py 가 부르는 YOLO·GPT 를 가짜로 바꾸고, high 재분석은 꺼서 시간 계산을 단순하게 한다."""
    monkeypatch.setattr(main, "run_yolo_multi_object", fake_yolo)
    monkeypatch.setattr(main, "analyze_numbered_original_image", fake_gpt)
    monkeypatch.setattr(main, "should_reanalyze_high", lambda yolo, vlm: (False, []))


# ASGI 앱에 같은 사진으로 /analyze 요청 n건을 동시에 보내고 (상태 코드 목록, 전체 걸린 초)를 돌려준다
async def send_concurrent(n: int) -> tuple[list[int], float]:
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
        started = time.perf_counter()
        responses = await asyncio.gather(
            *[client.post("/analyze", files={"image": ("can.jpg", b"fake-image", "image/jpeg")}) for _ in range(n)]
        )
        return [r.status_code for r in responses], time.perf_counter() - started


def test_동시_요청_3건이_GPT_대기를_겹쳐서_처리된다(fake_pipeline):
    codes, elapsed = asyncio.run(send_concurrent(3))

    assert codes == [200, 200, 200]
    # 한 줄로 처리되면 최소 3 × 0.5 = 1.5초. 겹쳐서 처리되면 0.5초 남짓이라 1.5초보다 확실히 짧아야 함
    assert elapsed < FAKE_GPT_SECONDS * 2, f"동시 3건이 {elapsed:.2f}초 걸림 (순차 처리 의심)"


def test_동시_처리해도_응답_내용은_1건일_때와_같다(fake_pipeline):
    async def one_and_three():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
            post = lambda: client.post("/analyze", files={"image": ("can.jpg", b"fake-image", "image/jpeg")})  # noqa: E731
            single = await post()
            many = await asyncio.gather(post(), post(), post())
            return single.json(), [r.json() for r in many]

    single, many = asyncio.run(one_and_three())
    for result in many:
        assert result == single
    # /analyze 는 사용자 확인 전이라 finalResult 가 None 이 정상 → GPT 판단(vlm.itemCode)으로 확인
    assert single["objects"][0]["vlm"]["itemCode"] == "CAN"


def test_GPT_타임아웃은_스레드로_넘긴_뒤에도_504(monkeypatch, fake_pipeline):
    def timeout_gpt(*args, **kwargs):
        raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))

    monkeypatch.setattr(main, "analyze_numbered_original_image", timeout_gpt)
    codes, _ = asyncio.run(send_concurrent(1))

    assert codes == [504]


class CountingModel:
    """predict 가 동시에 몇 개 실행 중인지 세는 가짜 YOLO 모델. 결과 파싱까지 갈 필요는 없어서 기록 후 예외로 끝낸다."""

    def __init__(self):
        self.running = 0
        self.max_running = 0
        self.guard = threading.Lock()

    # 실행 중 개수를 올리고 잠깐 머문 뒤 내린다 (잠금이 없으면 여러 스레드가 겹쳐서 max_running 이 1보다 커짐)
    def predict(self, **kwargs):
        with self.guard:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        time.sleep(0.05)
        with self.guard:
            self.running -= 1
        raise RuntimeError("stop-after-predict")


def test_여러_스레드가_YOLO_predict_를_동시에_하지_않는다():
    model = CountingModel()

    # 스레드 하나에서 _collect_detections 를 불러 predict 까지만 실행시킨다
    def worker():
        with pytest.raises(RuntimeError, match="stop-after-predict"):
            yolo_service._collect_detections(model, image=None)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert model.max_running == 1
