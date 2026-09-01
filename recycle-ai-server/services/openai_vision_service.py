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
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))


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

    client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)

    response = client.responses.create(
        model=model,
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

    client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)
    prompt = (
        f"{MULTI_OBJECT_STATE_PROMPT}\n\n"
        "YOLO 객체 목록:\n"
        f"{json.dumps(yolo_objects, ensure_ascii=False, indent=2)}"
    )

    response = client.responses.create(
        model=model,
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
