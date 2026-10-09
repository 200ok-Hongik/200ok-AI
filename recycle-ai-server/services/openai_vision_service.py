# 2026-10-09: 프롬프트에 "현재 분석 단계: low/high"를 붙여 보내도록 변경 - 프롬프트 규칙은 단계별로 bbox 의무가
# 다른데(high는 필수) GPT가 단계를 알 방법이 없어서, high에서도 bbox를 비우면 추가 물체가 object_N이 되지 못했음.
import base64
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from prompts.pet_bottle_prompt import MULTI_OBJECT_STATE_PROMPT, PET_BOTTLE_STATE_PROMPT
load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
# 2026-10-09: 벤치마크(bench/) 측정 결과 GPT 응답 시간의 대부분이 추론(reasoning) 토큰이었고
# (출력 토큰의 70~80%), 시간이 60초를 넘는 호출도 있었음.
# - 타임아웃: 60초를 넘으면 SDK가 처음부터 다시 요청해서(기본 2회 재시도) 한 번에 3분 가까이 걸린 사례가
#   있었음. 타임아웃을 넉넉히 두고 재시도는 1회로 제한해서, 오래 걸리는 요청을 버리고 다시 시작하지 않게 함.
# - 추론 강도: OPENAI_REASONING_EFFORT 로 조절. 비워 두면 reasoning 옵션을 보내지 않음(모델 기본값,
#   추론을 지원하지 않는 모델로 바꿀 때 사용).
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low").strip()


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
    prompt = (
        f"{MULTI_OBJECT_STATE_PROMPT}\n\n"
        # detail 값이 곧 프롬프트의 "low"/"high" 단계라서 그대로 알려준다
        f"현재 분석 단계: {detail}\n\n"
        "YOLO 객체 목록:\n"
        f"{json.dumps(yolo_objects, ensure_ascii=False, indent=2)}"
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
                        "image_url": _image_to_data_url(image_bytes, mime_type),
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    raw_text = response.output_text or ""
    return _parse_gpt_json(raw_text)
