# 2026-11-06: GPT 응답을 펼칠 때 categoryCode를 8품목 대문자/UNKNOWN으로 정리(_normalize_category_code_field) -
# "can"·"BATTERY"가 YOLO 탐지 물체 쪽으로 그대로 나가 백엔드 품목 조회가 실패할 수 있었고, 소문자 같은 품목도
# YOLO와 불일치로 잡혀 불필요한 high 재분석이 일어났음.
# 2026-11-05: 응답 시간 단축을 위해 GPT 출력 JSON을 간결한 형태(states 객체 + flags 배열)로 바꾸고, 받은 직후
# _expand_compact_result로 기존 내부 형태(checklistResults, bboxHasMultipleObjects 등)로 펼침 - result_builder와
# 백엔드 응답 계약은 그대로. GPT에 보내는 YOLO 목록도 들여쓰기 없이 보내 입력 토큰을 줄임.
# 2026-10-13: GPT에 보내기 전 이미지를 OpenAI가 high 단계에서 실제로 쓰는 크기(긴 변 2048px, 32px 패치 2,500개
# 이하)로 미리 줄여 보내도록 _prepare_image_for_vlm 추가 - 휴대폰 원본(4000px급, 수 MB)을 그대로 보내도 OpenAI가
# 같은 크기로 줄여서 보므로 정확도는 같고 업로드·처리 시간만 줄어듦. 줄인 이미지 기준으로 YOLO bbox를 바꿔 보내고,
# GPT가 돌려준 missedObjects bbox는 원본 좌표로 되돌림 (백엔드/앱은 계속 원본 좌표를 받음).
# 줄이기 전에 EXIF 회전을 YOLO와 같은 방식으로 적용 (휴대폰 세로 사진 좌표 어긋남 방지).
# 2026-10-09: 프롬프트에 "현재 분석 단계: low/high"를 붙여 보내도록 변경 - 프롬프트 규칙은 단계별로 bbox 의무가
# 다른데(high는 필수) GPT가 단계를 알 방법이 없어서, high에서도 bbox를 비우면 추가 물체가 object_N이 되지 못했음.
import base64
import copy
import io
import json
import math
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps

from prompts.pet_bottle_prompt import MULTI_OBJECT_STATE_PROMPT, PET_BOTTLE_STATE_PROMPT
from schemas.analysis_schema import normalize_item_code
load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
# 2026-10-09: 벤치마크(bench/) 측정 결과 GPT 응답 시간의 대부분이 추론(reasoning) 토큰이었고
# (출력 토큰의 70~80%), 시간이 60초를 넘는 호출도 있었음.
# - 타임아웃: 60초를 넘으면 SDK가 처음부터 다시 요청해서(기본 2회 재시도) 한 번에 3분 가까이 걸린 사례가
#   있었음. 타임아웃을 넉넉히 두고 재시도는 1회로 제한해서, 오래 걸리는 요청을 버리고 다시 시작하지 않게 함.
# - 추론 강도: OPENAI_REASONING_EFFORT 로 조절. 비워 두면 reasoning 옵션을 보내지 않음(모델 기본값,
#   추론을 지원하지 않는 모델로 바꿀 때 사용).
# - 2026-11-13: 기본값 low → none. 생활사진 72장 측정에서 품목 정답은 같고(75/112) 사진당 9.6s → 5.6s.
#   운영(EC2)은 docker-compose.yml environment 에서 none 으로 고정하고, 여기 기본값은 로컬 실행을 맞추는 용도.
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none").strip()  # 11-13: low → none
# OpenAI 이미지 가이드(developers.openai.com/api/docs/guides/images-vision) 기준 gpt-5.5의 high 처리 한도:
# 긴 변 2048px, 32x32 패치 2,500개. 이보다 큰 이미지는 OpenAI가 어차피 이 크기로 줄여서 본다.
# VLM_MAX_IMAGE_SIDE=0 이면 줄이지 않음(이전 동작으로 되돌리기용).
VLM_MAX_IMAGE_SIDE = int(os.getenv("VLM_MAX_IMAGE_SIDE", "2048"))
VLM_MAX_IMAGE_PATCHES = int(os.getenv("VLM_MAX_IMAGE_PATCHES", "2500"))
VLM_JPEG_QUALITY = int(os.getenv("VLM_JPEG_QUALITY", "85"))
_PATCH_SIZE = 32


def _vlm_target_size(width: int, height: int) -> tuple[int, int]:
    """긴 변·패치 수 한도에 맞는 크기. 이미 작으면 그대로 (OpenAI도 확대하지 않음)."""
    scale = min(1.0, VLM_MAX_IMAGE_SIDE / max(width, height))
    while scale > 0:
        w, h = max(1, int(width * scale)), max(1, int(height * scale))
        if math.ceil(w / _PATCH_SIZE) * math.ceil(h / _PATCH_SIZE) <= VLM_MAX_IMAGE_PATCHES:
            return w, h
        # 패치 수 한도를 넘으면 면적 비율만큼 줄이고, 반올림 오차로 다시 넘으면 조금씩 더 줄임
        scale *= min(0.99, math.sqrt(VLM_MAX_IMAGE_PATCHES / (math.ceil(w / _PATCH_SIZE) * math.ceil(h / _PATCH_SIZE))))
    return width, height


def _prepare_image_for_vlm(image_bytes: bytes, mime_type: str) -> tuple[bytes, str, float]:
    """(보낼 이미지, mime, 원본 대비 배율). 줄일 필요가 없으면 원본 그대로 보내고 배율 1.0."""
    if VLM_MAX_IMAGE_SIDE <= 0:
        return image_bytes, mime_type, 1.0
    # YOLO(yolo_service._load_image)와 똑같이 EXIF 회전을 먼저 적용해야 좌표 기준이 맞음 -
    # 저장 시 EXIF가 빠지므로, 안 하면 세로로 찍은 휴대폰 사진이 눕혀진 채로 GPT에 가서 bbox가 어긋남
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    target = _vlm_target_size(*image.size)
    if target == image.size:
        return image_bytes, mime_type, 1.0
    resized = image.resize(target, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=VLM_JPEG_QUALITY)
    return buffer.getvalue(), "image/jpeg", target[0] / image.size[0]


def _scale_bbox(bbox: Any, factor: float) -> Any:
    """[x1,y1,x2,y2] 또는 {"xMin"...} 형태 bbox 좌표에 배율을 곱한다. 형식이 다르면 그대로 둔다."""
    if isinstance(bbox, list) and len(bbox) == 4:
        return [round(v * factor) for v in bbox]
    if isinstance(bbox, dict) and all(k in bbox for k in ("xMin", "yMin", "xMax", "yMax")):
        try:
            return {k: round(float(bbox[k]) * factor) for k in ("xMin", "yMin", "xMax", "yMax")}
        except (TypeError, ValueError):
            return bbox
    return bbox


def _create_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES)


def _reasoning_options() -> dict[str, Any]:
    if not OPENAI_REASONING_EFFORT:
        return {}
    return {"reasoning": {"effort": OPENAI_REASONING_EFFORT}}


def _image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    text = _strip_json_fence(text)
    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in GPT response.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    raise ValueError("GPT response JSON object is incomplete.")


def _parse_gpt_json(text: str) -> dict[str, Any]:
    json_text = _extract_first_json_object(text)
    parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        raise ValueError("GPT response must be a JSON object.")

    return parsed


# 간결한 GPT 응답의 flags 값 -> 기존 내부 필드명. result_builder가 이 필드명으로 high 재분석·검토 사유를 판단함
_FLAG_TO_FIELD = {
    "MULTIPLE_OBJECTS": "bboxHasMultipleObjects",
    "NEARBY_UNLABELED": "hasUnlabeledNearbyObject",
    "TOO_SMALL_OR_BLURRED": "isTooSmallOrBlurred",
    "NEEDS_REVIEW": "needsReview",
}


def _states_to_checklist(states: dict[str, Any]) -> list[dict[str, Any]]:
    """{"hasCap": true, ...} -> [{"checkItemName": "hasCap", "statusValue": "true", ...}] (기존 checklistResults 형태)."""
    checklist = []
    for name, value in states.items():
        # 상태 항목별 confidence는 GPT에서 더 이상 받지 않음 - 최종 states가 true/false/null만 쓰기 때문
        if value is True or value is False:
            checklist.append({"checkItemName": name, "statusValue": str(value).lower(), "confidence": None})
        else:
            checklist.append({"checkItemName": name, "statusValue": value, "confidence": None})
    return checklist


def _expand_compact_result(result: dict[str, Any], yolo_objects: list[dict[str, Any]]) -> dict[str, Any]:
    """간결한 GPT 응답을 result_builder가 기대하는 기존 형태로 펼친다. 이미 기존 형태인 항목은 그대로 둔다."""
    yolo_category_by_id = {item.get("objectId"): item.get("categoryCode") for item in yolo_objects}

    for item in result.get("objects") or []:
        if not isinstance(item, dict):
            continue
        _normalize_category_code_field(item)  # 아래 YOLO 일치 비교 전에 정리해야 "can" vs "CAN"이 불일치로 안 잡힘
        if isinstance(item.get("states"), dict) and "checklistResults" not in item:
            item["checklistResults"] = _states_to_checklist(item["states"])
        if isinstance(item.get("flags"), list):
            flags = {str(flag).upper() for flag in item["flags"]}
            for flag, field in _FLAG_TO_FIELD.items():
                # flags에 없으면 해당 없음(false) - GPT가 해당할 때만 적게 해서 출력 토큰을 줄였기 때문
                item.setdefault(field, flag in flags)
        if "yoloCategoryMatches" not in item:
            # GPT에게 묻지 않고 서버가 직접 비교 (should_reanalyze_high도 같은 비교를 이미 하고 있음)
            yolo_category = yolo_category_by_id.get(item.get("objectId"))
            vlm_category = item.get("categoryCode")
            item["yoloCategoryMatches"] = (
                None if not yolo_category or not vlm_category else yolo_category == vlm_category
            )

    for item in result.get("missedObjects") or []:
        if isinstance(item, dict):
            _normalize_category_code_field(item)  # "paper" → PAPER (전엔 8품목인데도 UNKNOWN으로 바뀌었음)
        if isinstance(item, dict) and isinstance(item.get("states"), dict) and "checklistResults" not in item:
            item["checklistResults"] = _states_to_checklist(item["states"])

    return result


def _normalize_category_code_field(item: dict[str, Any]) -> None:
    """item["categoryCode"]를 8품목 대문자/UNKNOWN으로 바꾼다. 비어 있으면 키를 지워 YOLO 품목 대체 동작을 유지."""
    if "categoryCode" not in item:
        return
    code = normalize_item_code(item["categoryCode"])
    if code is None:
        item.pop("categoryCode")  # 빈 문자열로 남기면 기존처럼 YOLO 품목으로 대체되는 흐름이 명확하지 않음
    else:
        item["categoryCode"] = code


def analyze_pet_bottle_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = DEFAULT_MODEL,
    detail: str = "auto",
) -> dict[str, Any]:
    """
    crop 이미지를 GPT Vision에 보내 페트병 상태 JSON을 반환한다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    client = _create_client(api_key)

    response = client.responses.create(
        model=model,
        **_reasoning_options(),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": PET_BOTTLE_STATE_PROMPT,
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(image_bytes, mime_type),
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    raw_text = response.output_text or ""
    return _parse_gpt_json(raw_text)


def analyze_numbered_original_image(
    image_bytes: bytes,
    yolo_objects: list[dict[str, Any]],
    mime_type: str = "image/jpeg",
    detail: str = "low",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    YOLO 번호 bbox가 표시된 원본 이미지를 GPT Vision에 보내
    객체별 품목과 상태 분석 JSON을 반환한다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    client = _create_client(api_key)
    vlm_image_bytes, vlm_mime_type, scale = _prepare_image_for_vlm(image_bytes, mime_type)
    # GPT가 보는 이미지와 좌표계를 맞추기 위해 YOLO bbox도 같은 배율로 줄여서 보낸다 (원본 리스트는 건드리지 않음)
    vlm_yolo_objects = copy.deepcopy(yolo_objects)
    if scale != 1.0:
        for item in vlm_yolo_objects:
            item["bbox"] = _scale_bbox(item.get("bbox"), scale)
    prompt = (
        f"{MULTI_OBJECT_STATE_PROMPT}\n\n"
        # detail 값이 곧 프롬프트의 "low"/"high" 단계라서 그대로 알려준다
        f"현재 분석 단계: {detail}\n\n"
        "YOLO 객체 목록:\n"
        # 들여쓰기 없이 보냄 - 줄바꿈·공백도 입력 토큰이라 물체가 많을수록 손해
        f"{json.dumps(vlm_yolo_objects, ensure_ascii=False, separators=(',', ':'))}"
    )

    response = client.responses.create(
        model=model,
        **_reasoning_options(),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(vlm_image_bytes, vlm_mime_type),
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    raw_text = response.output_text or ""
    result = _expand_compact_result(_parse_gpt_json(raw_text), yolo_objects)
    # GPT가 준 추가 물체 좌표는 줄인 이미지 기준이라 원본 좌표로 되돌린다 (응답·백엔드는 원본 좌표 사용)
    if scale != 1.0 and isinstance(result.get("missedObjects"), list):
        for item in result["missedObjects"]:
            if isinstance(item, dict) and item.get("bbox") is not None:
                item["bbox"] = _scale_bbox(item["bbox"], 1 / scale)
    return result
