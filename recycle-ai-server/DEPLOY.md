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

선택 환경변수 (없으면 아래 기본값 사용, 2026-10-09 분석 속도 개선 때 추가):

```text
OPENAI_REASONING_EFFORT=low     # GPT 추론 강도. 상태 판단이 이상하면 medium으로 올리거나 값을 비우면(모델 기본값) 이전 동작
OPENAI_TIMEOUT_SECONDS=120      # 60으로 두면 오래 걸리는 요청이 타임아웃 후 처음부터 재시도돼서 오히려 느려짐
OPENAI_MAX_RETRIES=1
```

EC2의 .env에 OPENAI_TIMEOUT_SECONDS=60이 남아 있으면 기본값(120)보다 우선하므로 120으로 바꾸거나 줄을 지웁니다.

주의:

```text
.env 파일은 배포에 올리지 않습니다.
OPENAI_API_KEY는 EC2 서버의 .env 파일에 직접 둡니다.
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

## AWS EC2 배포 흐름

> 2026-10-09 Render에서 AWS EC2로 이전했습니다. Render 서비스와 `render.yaml`은 더 이상 쓰지 않습니다.

현재 구성:

```text
AWS 계정: 2103-7956-1794 (서울 ap-northeast-2)
인스턴스: recycle-ai-server (t3.small, Ubuntu 24.04, gp3 20GB)
고정 IP(Elastic IP): 3.37.251.117
보안 그룹: recycle-ai-server-sg
  - 8000: 백엔드 Lightsail 고정 IP(15.164.34.219)만 허용
  - 22: 관리자 PC IP만 허용
실행 방식: Docker Compose (docker-compose.yml, restart: unless-stopped)
```

서버에 처음 올리거나 코드를 갱신할 때:

1. 로컬에서 main 기준 코드를 묶어 EC2로 복사합니다. (EC2에는 GitHub 인증이 없습니다)

```bash
git archive --format=tar.gz -o ai-main.tar.gz origin/main
scp -i ~/.ssh/recycle-ai-ec2.pem ai-main.tar.gz ubuntu@3.37.251.117:~/recycle-ai-server/
```

2. `.env`는 저장소에 올리지 않고 EC2의 `~/recycle-ai-server/.env`에 직접 둡니다. (권한 600)
3. EC2에서 압축을 풀고 다시 빌드합니다.

```bash
cd ~/recycle-ai-server
tar xzf ai-main.tar.gz && rm ai-main.tar.gz
docker compose up -d --build
curl http://localhost:8000/health
```

문제가 생겨 이전 버전으로 되돌릴 때:

- 설정만 되돌리기: EC2의 `.env`에서 값을 바꾸고 `docker compose up -d` (컨테이너가 새 설정으로 다시 뜸)
- 코드 되돌리기: 로컬에서 되돌릴 커밋을 지정해 위 1~3단계를 똑같이 진행 (예: `git archive --format=tar.gz -o ai-main.tar.gz <이전 커밋 해시>`)

로그 확인:

```bash
docker compose logs -f --tail 100
```

주의:

```text
SSH 키(recycle-ai-ec2.pem)는 AWS에서 재발급되지 않습니다. 저장소에 절대 올리지 않습니다.
관리자 PC IP가 바뀌면 보안 그룹 22번 규칙을 새 IP로 수정해야 SSH가 됩니다.
백엔드 서버 IP가 바뀌면 보안 그룹 8000번 규칙을 새 IP로 수정해야 합니다.
```

## 백엔드에 전달할 주소

Spring Boot 백엔드는 환경변수 `AI_SERVER_URL`로 이 서버 주소를 읽습니다.

```text
AI_SERVER_URL=http://3.37.251.117:8000
POST {AI_SERVER_URL}/analyze
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
