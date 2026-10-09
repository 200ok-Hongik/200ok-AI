"""
AI 서버 /analyze 속도·결과 벤치마크.

실제 /analyze 코드를 그대로 실행하면서(서버를 따로 띄울 필요 없음) 단계별 시간,
GPT 호출 횟수(high 재분석 여부), GPT로 보낸 이미지 크기, 토큰 수, 분석 결과를 기록한다.
개선을 하나 적용할 때마다 같은 사진 세트로 돌리고 --compare로 이전 결과와 비교한다.

준비:
  1) bench/images/ 에 테스트 사진을 넣는다 (jpg/jpeg/png/webp). 매번 같은 사진 세트를 써야 비교가 됨.
  2) (선택) bench/labels.json 에 사진별 정답을 적으면 품목·상태 정답률도 같이 계산한다.
       물체 하나당 품목 코드 하나. 상태까지 채점하려면 {"item": ..., "states": {...}} 형식으로 적는다
       (확실한 상태만 적으면 되고, 적지 않은 상태는 채점하지 않음). 두 형식을 섞어 써도 된다.
       {
         "pet1.jpg": [{"item": "CLEAR_PET_BOTTLE", "states": {"hasLabel": true, "hasCap": true}}],
         "two_items.jpg": ["CAN", "PLASTIC"],
         "empty.jpg": []
       }
     채점은 --compare 할 때 지금의 labels.json 으로 다시 하므로, 정답을 나중에 고쳐도 GPT를 다시 돌릴 필요가 없다.
  3) recycle-ai-server/.env 에 OPENAI_API_KEY 가 있어야 한다 (실제 GPT를 호출하므로 비용 발생).

사용법 (recycle-ai-server 폴더에서 실행):
  python bench/benchmark.py --label baseline              # 사진마다 1회
  python bench/benchmark.py --label resize --repeat 3     # 사진마다 3회 (GPT 응답 시간 편차 완화)
  python bench/benchmark.py --compare baseline resize     # 저장된 두 결과 비교 (GPT 호출 없음)

결과는 bench/results/<label>.json 에 저장된다.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
SERVER_DIR = BENCH_DIR.parent
IMAGES_DIR = BENCH_DIR / "images"
RESULTS_DIR = BENCH_DIR / "results"
LABELS_PATH = BENCH_DIR / "labels.json"
MIME_BY_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


# ---------------------------------------------------------------- 측정용 계측


class _Recorder:
    def __init__(self) -> None:
        self.yolo_ms: list[float] = []
        self.gpt_calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.yolo_ms = []
        self.gpt_calls = []


REC = _Recorder()


def _instrument(main_module: Any) -> None:
    """서버 코드는 건드리지 않고, 실행 중에만 YOLO/GPT 호출을 감싸서 시간과 크기를 잰다."""
    import openai.resources.responses as responses_module

    original_create = responses_module.Responses.create

    def timed_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        image_b64_chars = 0
        detail = None
        for message in kwargs.get("input") or []:
            for part in message.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    image_url = str(part.get("image_url") or "")
                    image_b64_chars += len(image_url.split(",", 1)[-1])
                    detail = part.get("detail")

        started = time.perf_counter()
        response = original_create(self, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000

        usage = getattr(response, "usage", None)
        output_details = getattr(usage, "output_tokens_details", None)
        REC.gpt_calls.append(
            {
                "detail": detail,
                "model": kwargs.get("model"),
                "ms": round(elapsed_ms),
                "imageKB": round(image_b64_chars * 3 / 4 / 1024),
                "inputTokens": getattr(usage, "input_tokens", None),
                "outputTokens": getattr(usage, "output_tokens", None),
                "reasoningTokens": getattr(output_details, "reasoning_tokens", None),
            }
        )
        return response

    responses_module.Responses.create = timed_create

    original_yolo = main_module.run_yolo_multi_object

    def timed_yolo(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_yolo(*args, **kwargs)
        finally:
            REC.yolo_ms.append((time.perf_counter() - started) * 1000)

    main_module.run_yolo_multi_object = timed_yolo


# ---------------------------------------------------------------- 실행


def _git_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=SERVER_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=SERVER_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{commit}{' (+로컬 수정)' if dirty else ''}"
    except Exception:
        return "unknown"


def _load_labels() -> dict[str, list[Any]]:
    if not LABELS_PATH.exists():
        return {}
    with open(LABELS_PATH, encoding="utf-8") as file:
        return json.load(file)


def _label_objects(entries: list[Any]) -> list[dict[str, Any]]:
    """labels.json 한 사진의 정답을 [{"item": 코드, "states": {...}}] 형태로 통일한다."""
    objects = []
    for entry in entries:
        if isinstance(entry, str):
            objects.append({"item": entry, "states": {}})
        elif isinstance(entry, dict):
            objects.append({"item": entry.get("item"), "states": entry.get("states") or {}})
    return objects


def _grade(run: dict[str, Any], labels: dict[str, list[Any]]) -> dict[str, Any] | None:
    """
    한 번의 실행 결과를 정답과 비교한다. 정답이 없는 사진이거나 실패한 실행이면 None.
    - 품목: YOLO가 박스를 친 물체(objects)의 품목 목록이 정답 목록과 완전히 같아야 정답.
      (GPT가 따로 찾은 추가 후보는 채점하지 않음)
    - 상태: 정답의 각 물체를 같은 품목의 예측 물체와 짝지은 뒤, 정답에 적힌 상태만 비교.
      맞음 / 틀림 / null(AI가 판단 못함) / 놓침(그 물체 자체를 못 찾음)으로 센다.
    """
    if run.get("status") != 200 or run["image"] not in labels:
        return None
    truth = _label_objects(labels[run["image"]])
    predicted = run.get("objects") or []

    grade = {
        "itemCorrect": sorted(str(obj["itemCode"]) for obj in predicted) == sorted(str(t["item"]) for t in truth),
        "stateCorrect": 0,
        "stateWrong": 0,
        "stateNull": 0,
        "stateMissed": 0,
        "wrongStates": [],
    }
    unused = list(range(len(predicted)))
    for target in truth:
        if not target["states"]:
            continue
        candidates = [i for i in unused if predicted[i]["itemCode"] == target["item"]]

        def agreement(index: int) -> int:
            states = predicted[index].get("states") or {}
            return sum(1 for key, value in target["states"].items() if states.get(key) == value)

        if not candidates:
            grade["stateMissed"] += len(target["states"])
            continue
        best = max(candidates, key=agreement)
        unused.remove(best)
        states = predicted[best].get("states") or {}
        for key, expected in target["states"].items():
            actual = states.get(key)
            if actual is None:
                grade["stateNull"] += 1
            elif actual == expected:
                grade["stateCorrect"] += 1
            else:
                grade["stateWrong"] += 1
                grade["wrongStates"].append(f"{target['item']}.{key}={actual}")
    return grade


def _summarize_body(body: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for item in body.get("objects") or []:
        vlm = item.get("vlm") or {}
        yolo = item.get("yolo") or {}
        objects.append(
            {
                "objectId": item.get("objectId"),
                "yoloItem": yolo.get("itemCode"),
                "itemCode": vlm.get("itemCode"),
                "states": vlm.get("states"),
            }
        )
    analysis = body.get("analysis") or {}
    return {
        "objects": objects,
        "additionalObjectCount": len(body.get("additionalObjects") or []),
        "currentLevel": analysis.get("currentLevel"),
        "highTriggerReasons": sorted({reason.get("code") for reason in analysis.get("highTriggerReasons") or []}),
    }


def run_benchmark(label: str, repeat: int, limit: int | None) -> Path:
    if not IMAGES_DIR.exists():
        IMAGES_DIR.mkdir(parents=True)
    images = sorted(path for path in IMAGES_DIR.iterdir() if path.suffix.lower() in MIME_BY_EXT)
    if limit:
        images = images[:limit]
    if not images:
        sys.exit(f"테스트 사진이 없습니다. {IMAGES_DIR} 에 jpg/png 사진을 넣어주세요.")

    os.chdir(SERVER_DIR)  # models/best.pt, .env 를 서버 실행 때와 같은 기준으로 찾기 위해
    sys.path.insert(0, str(SERVER_DIR))
    import main  # noqa: E402  (chdir 이후에 import 해야 .env 를 읽음)
    from fastapi.testclient import TestClient

    _instrument(main)
    import logging

    # 서버 자체 로그는 결과 표를 가리므로 이 스크립트 실행 중에는 경고 이상만 표시한다.
    for logger_name in ("recycle_ai_server", "httpx", "httpx2"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    client = TestClient(main.app)
    labels = _load_labels()

    print(f"[{label}] 사진 {len(images)}장 x {repeat}회, 모델={os.getenv('OPENAI_MODEL', '기본값')}, 코드={_git_version()}")
    # 첫 요청에만 붙는 모델 로딩·첫 추론 지연(약 2초)이 측정에 섞이지 않도록 미리 한 번 돌려 둔다.
    # 실제 서버에서도 첫 요청 이후에는 모델이 메모리에 남아 있으므로 이 쪽이 실사용에 가깝다.
    from PIL import Image
    from services.yolo_service import _get_img_size, _get_model

    _get_model().predict(source=Image.new("RGB", (640, 640)), imgsz=_get_img_size(), verbose=False)

    runs: list[dict[str, Any]] = []
    for image_path in images:
        for attempt in range(1, repeat + 1):
            REC.reset()
            started = time.perf_counter()
            with open(image_path, "rb") as image_file:
                response = client.post(
                    "/analyze",
                    files={"image": (image_path.name, image_file, MIME_BY_EXT[image_path.suffix.lower()])},
                )
            total_ms = (time.perf_counter() - started) * 1000

            run: dict[str, Any] = {
                "image": image_path.name,
                "attempt": attempt,
                "status": response.status_code,
                "totalMs": round(total_ms),
                "yoloMs": round(sum(REC.yolo_ms)),
                "gptCalls": REC.gpt_calls,
            }
            if response.status_code == 200:
                run.update(_summarize_body(response.json()))
            else:
                run["error"] = response.text[:500]

            runs.append(run)
            _print_run(run, labels)

    summary = _summarize_runs(runs, labels)
    result = {
        "label": label,
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "openaiModel": os.getenv("OPENAI_MODEL"),
        "codeVersion": _git_version(),
        "repeat": repeat,
        "summary": summary,
        "runs": runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}.json"
    with open(out_path, "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, ensure_ascii=False, indent=2)

    print()
    _print_summary(label, summary)
    print(f"\n저장: {out_path}")
    return out_path


# ---------------------------------------------------------------- 집계·출력


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _summarize_runs(runs: list[dict[str, Any]], labels: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    ok_runs = [run for run in runs if run["status"] == 200]
    totals = [run["totalMs"] for run in ok_runs]
    low_calls = [call for run in ok_runs for call in run["gptCalls"] if call["detail"] == "low"]
    high_calls = [call for run in ok_runs for call in run["gptCalls"] if call["detail"] == "high"]
    all_calls = low_calls + high_calls
    grades = [grade for grade in (_grade(run, labels or {}) for run in ok_runs) if grade is not None]
    reason_counts: dict[str, int] = {}
    for run in ok_runs:
        for reason in run.get("highTriggerReasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "runCount": len(runs),
        "errorCount": len(runs) - len(ok_runs),
        "totalMsMean": _mean(totals),
        "totalMsMedian": statistics.median(totals) if totals else None,
        "totalMsMax": max(totals) if totals else None,
        "yoloMsMean": _mean([run["yoloMs"] for run in ok_runs]),
        "lowMsMean": _mean([call["ms"] for call in low_calls]),
        "highMsMean": _mean([call["ms"] for call in high_calls]),
        "highRunCount": sum(1 for run in ok_runs if any(c["detail"] == "high" for c in run["gptCalls"])),
        "highRatePct": (
            sum(1 for run in ok_runs if any(c["detail"] == "high" for c in run["gptCalls"])) / len(ok_runs) * 100
            if ok_runs
            else None
        ),
        "highReasonCounts": dict(sorted(reason_counts.items(), key=lambda item: -item[1])),
        "imageKBMean": _mean([call["imageKB"] for call in all_calls]),
        "inputTokensMean": _mean([c["inputTokens"] for c in all_calls if c["inputTokens"] is not None]),
        "outputTokensMean": _mean([c["outputTokens"] for c in all_calls if c["outputTokens"] is not None]),
        "itemCorrect": sum(1 for grade in grades if grade["itemCorrect"]),
        "itemLabeled": len(grades),
        "stateCorrect": sum(grade["stateCorrect"] for grade in grades),
        "stateWrong": sum(grade["stateWrong"] for grade in grades),
        "stateNull": sum(grade["stateNull"] for grade in grades),
        "stateMissed": sum(grade["stateMissed"] for grade in grades),
    }


def _sec(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}s"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def _print_run(run: dict[str, Any], labels: dict[str, list[Any]] | None = None) -> None:
    head = f"  {run['image']} #{run['attempt']}: {_sec(run['totalMs'])}"
    if run["status"] != 200:
        print(f"{head}  [에러 {run['status']}] {run.get('error', '')[:120]}")
        return
    # 한 번의 호출이 타임아웃보다 길면 SDK가 타임아웃 후 처음부터 다시 요청했다는 뜻이다.
    timeout_ms = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")) * 1000
    gpt = ", ".join(
        f"{c['detail']} {_sec(c['ms'])}{' [타임아웃 후 재시도됨]' if c['ms'] > timeout_ms else ''}"
        for c in run["gptCalls"]
    )
    items = [obj["itemCode"] for obj in run["objects"]]
    extra = f" +후보{run['additionalObjectCount']}" if run["additionalObjectCount"] else ""
    grade = _grade(run, labels or {})
    correct = ""
    if grade is not None:
        correct = "  품목O" if grade["itemCorrect"] else "  품목X"
        labeled_states = grade["stateCorrect"] + grade["stateWrong"] + grade["stateNull"] + grade["stateMissed"]
        if labeled_states:
            correct += f" 상태 {grade['stateCorrect']}/{labeled_states}"
            if grade["wrongStates"]:
                correct += f" (틀림: {', '.join(grade['wrongStates'])})"
    print(f"{head}  (YOLO {_sec(run['yoloMs'])} | GPT {gpt})  품목={items}{extra}{correct}")


def _print_summary(label: str, s: dict[str, Any]) -> None:
    ok = s["runCount"] - s["errorCount"]
    print(f"[{label}] 요약 ({s['runCount']}회 중 성공 {ok}회)")
    print(f"  전체 시간   평균 {_sec(s['totalMsMean'])} / 중앙값 {_sec(s['totalMsMedian'])} / 최대 {_sec(s['totalMsMax'])}")
    print(f"  YOLO {_sec(s['yoloMsMean'])} | GPT low {_sec(s['lowMsMean'])} | GPT high {_sec(s['highMsMean'])}")
    print(f"  high 재분석 {s['highRunCount']}/{ok}회  사유: {s['highReasonCounts'] or '-'}")
    print(
        f"  GPT 1회당 이미지 {_num(s['imageKBMean'])}KB / 입력 토큰 {_num(s['inputTokensMean'])}"
        f" / 출력 토큰 {_num(s['outputTokensMean'])}"
    )
    if s["itemLabeled"]:
        print(f"  품목 정답률 {s['itemCorrect']}/{s['itemLabeled']} (labels.json 기준)")
    if _state_total(s):
        print(f"  상태 정답 {_state_text(s)}")


def _state_total(s: dict[str, Any]) -> int:
    return sum(s.get(key) or 0 for key in ("stateCorrect", "stateWrong", "stateNull", "stateMissed"))


def _state_text(s: dict[str, Any]) -> str:
    total = _state_total(s)
    return (
        f"{s['stateCorrect']}/{total} (틀림 {s['stateWrong']}, AI가 판단 못함(null) {s['stateNull']},"
        f" 물체를 못 찾음 {s['stateMissed']})"
    )


def compare(label_a: str, label_b: str) -> None:
    results = []
    for label in (label_a, label_b):
        path = RESULTS_DIR / f"{label}.json"
        if not path.exists():
            sys.exit(f"결과 파일이 없습니다: {path}")
        with open(path, encoding="utf-8") as file:
            results.append(json.load(file))
    a, b = results

    # 사진 구성이 다르면 숫자 차이가 수정 때문인지 사진 때문인지 알 수 없으므로,
    # 양쪽 결과에 모두 있는 사진만 골라서 다시 집계한다.
    # 한쪽에서 에러(크레딧 부족 등)로 실패한 사진도 시간 비교가 안 되므로 같이 제외한다.
    def succeeded(result: dict[str, Any]) -> set[str]:
        return {run["image"] for run in result["runs"] if run["status"] == 200}

    common = succeeded(a) & succeeded(b)
    if not common:
        sys.exit("두 결과에서 모두 성공한 사진이 없어서 비교할 수 없습니다.")
    only_a = {run["image"] for run in a["runs"]} - common
    only_b = {run["image"] for run in b["runs"]} - common
    a["runs"] = [run for run in a["runs"] if run["image"] in common]
    b["runs"] = [run for run in b["runs"] if run["image"] in common]
    labels = _load_labels()  # 저장 당시가 아니라 지금의 정답으로 양쪽을 다시 채점
    sa, sb = _summarize_runs(a["runs"], labels), _summarize_runs(b["runs"], labels)
    print(f"양쪽 모두 성공한 사진 {len(common)}장 기준으로 비교 (각각 {len(a['runs'])}회 / {len(b['runs'])}회 실행)")
    if only_a or only_b:
        print(f"  제외된 사진(한쪽에만 있거나 한쪽에서 실패): {label_a} 쪽 {len(only_a)}장, {label_b} 쪽 {len(only_b)}장")

    # 사진별 시간 (첫 시도 기준) - 평균 하나만 보면 이상치 하나에 결과가 뒤집힐 수 있어서 같이 보여준다.
    time_a = {run["image"]: run["totalMs"] for run in a["runs"] if run["attempt"] == 1}
    time_b = {run["image"]: run["totalMs"] for run in b["runs"] if run["attempt"] == 1}
    print("\n사진별 시간 (첫 시도 기준):")
    for image in sorted(common):
        high_a = any(c["detail"] == "high" for r in a["runs"] if r["image"] == image and r["attempt"] == 1 for c in r["gptCalls"])
        high_b = any(c["detail"] == "high" for r in b["runs"] if r["image"] == image and r["attempt"] == 1 for c in r["gptCalls"])
        mark = lambda used: "high" if used else "low만"  # noqa: E731
        print(f"  {image[:36]:<36} {_sec(time_a[image]):>7} ({mark(high_a)}) -> {_sec(time_b[image]):>7} ({mark(high_b)})")
    print()

    def row(name: str, key: str, fmt: Any, lower_is_better: bool = True) -> None:
        va, vb = sa.get(key), sb.get(key)
        change = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            pct = (vb - va) / va * 100
            change = f"  ({pct:+.0f}%"
            if abs(pct) >= 5:  # 5% 미만은 측정 편차로 보고 판정하지 않음
                change += ", 개선" if (pct < 0) == lower_is_better else ", 악화"
            change += ")"
        print(f"  - {name}: {fmt(va)} -> {fmt(vb)}{change}")

    print(f"비교: {label_a} ({a['codeVersion']}, {a['openaiModel']}) -> {label_b} ({b['codeVersion']}, {b['openaiModel']})")
    row("전체 평균", "totalMsMean", _sec)
    row("전체 중앙값", "totalMsMedian", _sec)
    row("전체 최대", "totalMsMax", _sec)
    row("GPT low 평균", "lowMsMean", _sec)
    row("GPT high 평균", "highMsMean", _sec)
    row("high 재분석 비율", "highRatePct", lambda v: "-" if v is None else f"{v:.0f}%")
    row("이미지 KB", "imageKBMean", _num)
    row("출력 토큰", "outputTokensMean", _num)
    row("에러 수", "errorCount", _num)
    if sa["itemLabeled"] or sb["itemLabeled"]:
        print(f"  - 품목 정답: {sa['itemCorrect']}/{sa['itemLabeled']} -> {sb['itemCorrect']}/{sb['itemLabeled']}")
    if _state_total(sa) or _state_total(sb):
        print(f"  - 상태 정답: {_state_text(sa)}")
        print(f"           -> {_state_text(sb)}")

    # 사진별 결과 차이 (각 사진의 첫 번째 시도끼리 비교)
    first_a = {run["image"]: run for run in a["runs"] if run["attempt"] == 1}
    first_b = {run["image"]: run for run in b["runs"] if run["attempt"] == 1}
    print("\n사진별 결과 차이 (첫 시도 기준):")
    any_diff = False
    for image in sorted(set(first_a) | set(first_b)):
        ra, rb = first_a.get(image), first_b.get(image)
        if not ra or not rb:
            print(f"  {image}: 한쪽 결과에만 있음")
            any_diff = True
            continue
        items_a = [obj["itemCode"] for obj in ra.get("objects", [])]
        items_b = [obj["itemCode"] for obj in rb.get("objects", [])]
        notes = []
        if ra["status"] != rb["status"]:
            notes.append(f"상태코드 {ra['status']} -> {rb['status']}")
        if items_a != items_b:
            notes.append(f"품목 {items_a} -> {items_b}")
        else:
            for obj_a, obj_b in zip(ra.get("objects", []), rb.get("objects", [])):
                changed = [
                    f"{key} {obj_a['states'].get(key)}->{(obj_b['states'] or {}).get(key)}"
                    for key in (obj_a.get("states") or {})
                    if (obj_b.get("states") or {}).get(key) != obj_a["states"].get(key)
                ]
                if changed:
                    notes.append(f"{obj_a['objectId']} 상태 " + ", ".join(changed))
        if ra.get("additionalObjectCount") != rb.get("additionalObjectCount"):
            notes.append(f"추가후보 {ra.get('additionalObjectCount')} -> {rb.get('additionalObjectCount')}")
        if notes:
            any_diff = True
            print(f"  {image}: " + " / ".join(notes))
    if not any_diff:
        print("  차이 없음")
    print("\n참고: GPT는 같은 사진에도 매번 조금씩 다르게 답할 수 있어서, 상태값 1~2개 차이는 개선 때문이 아닐 수도 있어요.")


def main_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI 서버 /analyze 벤치마크")
    parser.add_argument("--label", help="이번 측정 이름 (결과 파일명). 예: baseline, resize")
    parser.add_argument("--repeat", type=int, default=1, help="사진마다 반복 횟수 (기본 1)")
    parser.add_argument("--limit", type=int, help="앞에서부터 N장만 사용 (빠른 확인용)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="저장된 두 결과 비교")
    args = parser.parse_args(argv)

    if args.compare:
        compare(*args.compare)
    elif args.label:
        run_benchmark(args.label, max(1, args.repeat), args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
