# Recycle AI Server

AI-2 상태분석용 FastAPI 서버입니다.

Colab에서 실험하던 흐름:

```text
crop 이미지 -> GPT Vision -> 상태 JSON 출력
```

을 Spring Boot가 호출 가능한 API 서버 형태로 바꾼 프로젝트입니다.

```text
Spring Boot
-> POST /analyze
-> FastAPI AI Server
-> GPT Vision
-> JSON Response
-> Spring Boot
```

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
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## API

### POST /analyze

최종 A 구조용 API입니다.

Spring Boot가 원본 이미지를 보내면 FastAPI 서버 내부에서 YOLO -> crop -> GPT Vision을 실행합니다.

`services/yolo_service.py`에서 AI-1 YOLO 모델을 로드하고 bbox 기준 crop을 생성합니다.

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
| `rule_input_only` | Boolean | true면 Rule DB용 최소 JSON만 반환 |
| `region_code` | Text | 예: SEOUL |

### POST /analyze-ai2

AI-2 단독 테스트용 API입니다.

AI-1이 이미 만든 crop 이미지와 yolo_json을 넣어 GPT Vision 상태분석만 테스트합니다.

전송 방식:

```text
multipart/form-data
```

필수 필드:

| Key | Type | 설명 |
| --- | --- | --- |
| `crop` | File | YOLO가 만든 crop 이미지 |
| `yolo_json` | Text | YOLO 탐지 결과 JSON 문자열 |

선택 필드:

| Key | Type | 설명 |
| --- | --- | --- |
| `rule_input_only` | Boolean | true면 Rule DB용 최소 JSON만 반환 |
| `region_code` | Text | 예: SEOUL |

`yolo_json` 예시:

```json
{"itemCode":"PET_BOTTLE","bbox":[1,1,2560,1440],"confidence":0.991}
```

응답 예시:

```json
{
  "itemCode": "PET_BOTTLE",
  "itemName": "페트병",
  "bbox": [1, 1, 2560, 1440],
  "confidence": 0.991,
  "states": {
    "isTransparent": true,
    "hasLabel": true,
    "isEmpty": true,
    "hasCap": true,
    "isContaminated": false,
    "isCrushed": false
  },
  "needUserCheck": []
}
```

## 백엔드 연결

최종 A 구조에서 Spring Boot는 아래 주소로 요청합니다.

```text
POST http://AI서버IP:8000/analyze
```

전송 필드는 `image` 원본 이미지 파일입니다.

AI-2 단독 테스트가 필요하면 아래 주소를 사용합니다.

```text
POST http://AI서버IP:8000/analyze-ai2
```

전송 필드는 `crop`, `yolo_json`입니다.

다른 노트북에서 호출할 때는 `localhost`가 아니라 AI 서버가 실행 중인 노트북의 IPv4 주소를 사용해야 합니다.
