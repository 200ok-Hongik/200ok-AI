# Recycle AI Server

분리배출 AI 분석용 FastAPI 서버입니다.

현재 핵심 구조는 아래 방식입니다.

```text
사용자 원본 이미지
-> EXIF 회전/이미지 형식 보정
-> YOLO 다중 객체 탐지
-> 중복 bbox 제거
-> 객체 ID 부여
-> 원본 이미지 위에 번호 bbox 표시
-> GPT Vision low 분석
-> 불확실하면 동일한 원본 이미지를 GPT Vision high 재분석
-> 객체별 JSON 반환
-> Spring Boot Rule DB 매핑
```

중요: `/analyze`는 더 이상 crop 이미지를 만들지 않습니다. YOLO가 표시한 번호 bbox가 있는 원본 이미지를 GPT Vision에 전달합니다.

## 설치

```powershell
cd recycle-ai-server
python -m pip install -r requirements.txt
```

## 환경변수

`.env.example`을 참고해서 `.env` 파일을 만듭니다.

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.5
YOLO_MODEL_PATH=models/best.pt
YOLO_IMG_SIZE=640
YOLO_CONF_THRESHOLD=0.25
```

AI-1 YOLO 모델 파일은 기본적으로 아래 위치에 둡니다.

```text
models/best.pt
```

## 실행

```powershell
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8010
```

Swagger:

```text
http://127.0.0.1:8010/docs
```

Health check:

```text
http://127.0.0.1:8010/health
```

## API

### POST /analyze

백엔드가 사용하는 메인 분석 API입니다.

Spring Boot가 원본 이미지를 보내면 FastAPI 서버 내부에서 YOLO 다중 객체 탐지와 GPT Vision 상태 분석을 수행합니다.

전송 방식:

```text
multipart/form-data
```

필수 필드:

| Key | Type | 설명 |
| --- | --- | --- |
| `image` | File | 사용자가 업로드한 원본 이미지 |

선택 필드:

| Key | Type | 설명 |
| --- | --- | --- |
| `rule_input_only` | Boolean | `true`면 Rule DB용 최소 JSON만 반환 |
| `region_code` | Text | 예: `SEOUL_MAPO_GU` |
| `image_id` | Text | 백엔드가 이미지 저장 후 발급한 ID. 예: `image_001` |

응답 예시:

```json
{
  "image": {
    "imageId": "image_001",
    "width": 2560,
    "height": 1440
  },
  "imageQuality": {
    "status": "SUFFICIENT",
    "issues": []
  },
  "analysis": {
    "mode": "ORIGINAL_NUMBERED_BBOX",
    "modelVersion": "v1.1.0",
    "currentLevel": "HIGH",
    "attemptedLevels": ["LOW", "HIGH"],
    "needsHighAnalysis": true,
    "highTriggerReasons": [
      {
        "code": "REQUIRED_STATE_NULL",
        "objectId": "object_1",
        "field": "isEmpty"
      }
    ]
  },
  "objects": [
    {
      "objectId": "object_1",
      "bbox": {
        "xMin": 1286,
        "yMin": 714,
        "xMax": 2760,
        "yMax": 1220
      },
      "yolo": {
        "itemCode": "PET_BOTTLE",
        "className": "pet_bottle",
        "confidence": 0.935
      },
      "vlm": {
        "itemCode": "PET_BOTTLE",
        "confidence": 0.95,
        "states": {
          "isTransparent": true,
          "hasLabel": true,
          "isEmpty": null,
          "hasCap": false,
          "isContaminated": false,
          "isCrushed": false
        },
        "yoloItemMatches": true,
        "multipleObjectsInBox": false,
        "hasUnlabeledNearbyObject": false,
        "isTooSmallOrBlurred": false,
        "needsReview": true,
        "reviewReasons": [
          {
            "code": "REQUIRED_STATE_NULL",
            "objectId": "object_1",
            "field": "isEmpty"
          }
        ]
      },
      "review": {
        "status": "PENDING"
      },
      "finalResult": null
    }
  ],
  "additionalObjects": []
}
```

### 불확실 판정 조건

아래 중 하나라도 해당하면 동일한 번호 bbox 원본 이미지를 `detail=high`로 재분석합니다.

- YOLO와 VLM의 품목이 다름
- VLM이 `needsReview=true` 반환
- 필수 상태값이 `null`
- bbox 하나에 여러 객체가 있다고 판단
- bbox가 없는 추가 객체 발견
- 객체가 너무 작거나 흐림
- 이미지 품질이 부족함

### POST /analyze-ai2

예전 crop 기반 AI-2 단독 테스트용 API입니다.

새 메인 흐름에서는 사용하지 않지만, AI-1이 이미 만든 crop 이미지와 yolo_json으로 GPT Vision 상태분석만 따로 테스트할 때 남겨둔 호환 API입니다.

필수 필드:

| Key | Type | 설명 |
| --- | --- | --- |
| `crop` | File | YOLO가 만든 crop 이미지 |
| `yolo_json` | Text | YOLO 탐지 결과 JSON 문자열 |

### POST /debug-yolo

YOLO 모델이 어떤 클래스를 탐지하는지 확인하는 디버그 API입니다.

필수 필드:

| Key | Type | 설명 |
| --- | --- | --- |
| `image` | File | YOLO 탐지 테스트용 이미지 |

## 백엔드 연결

백엔드는 아래 주소로 원본 이미지를 전송합니다.

```text
POST http://AI서버주소:8010/analyze
```

백엔드는 응답의 `objects` 배열을 받아 각 객체별 `vlm.itemCode`, `vlm.states`, `bbox`를 사용자 확인 화면에 연결합니다.

사용자 확인이 끝나면 `finalResult` 또는 별도 V2 JSON의 `objects[].itemCode`, `objects[].states`를 Rule DB 판단에 사용합니다.

### resultSource 값

V2의 `resultSource`는 최종 결과가 어디에서 확정됐는지 나타냅니다.

| 값 | 의미 |
| --- | --- |
| `VLM_LOW` | GPT Vision low 분석 결과를 그대로 사용 |
| `VLM_HIGH` | GPT Vision high 재분석 결과를 그대로 사용 |
| `USER_CONFIRMED` | 사용자가 AI 결과를 수정 없이 확인 |
| `USER_CORRECTED` | 사용자가 AI 결과의 품목 또는 상태를 수정 |
| `USER_ADDED` | 사용자가 누락된 객체를 직접 추가 |

### review.status 값

V1의 `review.status`는 사용자 검증 진행 상태를 나타냅니다.

| 값 | 의미 |
| --- | --- |
| `PENDING` | AI 분석 직후, 사용자 확인 전 |
| `CONFIRMED` | 사용자가 AI 결과를 수정 없이 확인 |
| `CORRECTED` | 사용자가 AI 결과의 품목 또는 상태를 수정 |
| `ADDED` | 사용자가 누락된 객체를 직접 추가 |
