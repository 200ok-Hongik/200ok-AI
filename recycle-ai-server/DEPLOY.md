# Recycle AI Server Deploy

이 서버는 FastAPI + YOLO + GPT Vision 추론 서버입니다.

## 현재 서버 역할

```text
Spring Boot Backend
-> POST /analyze
-> FastAPI AI Server
-> YOLO 탐지 + crop 생성
-> GPT Vision 상태 분석
-> checklist JSON 반환
```

## 배포 전 확인

필수 파일:

```text
main.py
requirements.txt
models/best.pt
services/
schemas/
prompts/
Dockerfile
.dockerignore
```

필수 환경변수:

```text
OPENAI_API_KEY
OPENAI_MODEL
YOLO_MODEL_PATH=models/best.pt
YOLO_IMG_SIZE=640
YOLO_CONF_THRESHOLD=0.25
```

주의:

```text
.env 파일은 배포에 올리지 않습니다.
OPENAI_API_KEY는 배포 서비스의 Environment Variables에 직접 등록합니다.
```

## 로컬 실행

```powershell
cd "C:\Users\myjxj\OneDrive\문서\분리배출 졸프\recycle-ai-server"
py -m uvicorn main:app --reload --host 0.0.0.0 --port 8020
```

확인:

```text
http://127.0.0.1:8020/health
http://127.0.0.1:8020/docs
```

## Docker 로컬 실행

```powershell
docker build -t recycle-ai-server .
docker run --rm -p 8020:8000 --env-file .env recycle-ai-server
```

확인:

```text
http://127.0.0.1:8020/health
```

## Render 배포 흐름

1. GitHub에 `recycle-ai-server` 폴더를 포함한 저장소를 올립니다.
2. Render에서 New Web Service를 생성합니다.
3. Runtime은 Docker를 선택합니다.
4. Root Directory를 `recycle-ai-server`로 설정합니다.
5. Environment Variables에 아래 값을 등록합니다.

```text
OPENAI_API_KEY=실제 API 키
OPENAI_MODEL=gpt-5.5
YOLO_MODEL_PATH=models/best.pt
YOLO_IMG_SIZE=640
YOLO_CONF_THRESHOLD=0.25
```

6. 배포 후 아래 주소를 확인합니다.

```text
https://배포주소/health
https://배포주소/docs
```

## 백엔드에 전달할 주소

배포 후 Spring Boot 백엔드에는 아래 주소를 전달합니다.

```text
POST https://배포주소/analyze
```

전송 방식:

```text
multipart/form-data
```

필드:

```text
image: File
rule_input_only: false
region_code: SEOUL_MAPO_GU
```

## API

### GET /health

서버 상태 확인.

응답:

```json
{"status": "ok"}
```

### POST /debug-yolo

YOLO 탐지 결과 확인용.

### POST /analyze

백엔드 연결용 최종 API.

### POST /analyze-ai2

AI-1이 만든 crop + yolo_json으로 AI-2만 테스트하는 API.
