# Elastic Beanstalk Deploy

이 문서는 `recycle-ai-server`를 AWS Elastic Beanstalk Docker 환경에 배포하는 방법입니다.

## 1. 업로드 파일

업로드할 zip은 `recycle-ai-server` 폴더 안의 내용물만 포함해야 합니다.

zip 최상단에 아래 파일이 바로 보여야 합니다.

```text
Dockerfile
Dockerrun.aws.json
main.py
requirements.txt
models/
prompts/
schemas/
services/
```

아래 파일은 포함하면 안 됩니다.

```text
.env
.venv/
__pycache__/
*.log
```

## 2. Elastic Beanstalk 생성

AWS Console:

```text
Elastic Beanstalk
-> Create application
```

설정:

```text
Application name: recycle-ai-server
Environment: Web server environment
Platform: Docker
Platform branch: Docker running on 64bit Amazon Linux 2023
Application code: Upload your code
```

업로드 파일:

```text
recycle-ai-server-beanstalk.zip
```

## 3. 환경변수

Elastic Beanstalk 환경의 Configuration에서 Environment properties에 추가합니다.

```text
OPENAI_API_KEY=실제 OpenAI API 키
OPENAI_MODEL=gpt-5.5
YOLO_MODEL_PATH=models/best.pt
YOLO_IMG_SIZE=640
YOLO_CONF_THRESHOLD=0.25
```

## 4. 배포 후 확인

Elastic Beanstalk URL이 아래처럼 생성됩니다.

```text
http://recycle-ai-server-env.eba-xxxxx.ap-northeast-2.elasticbeanstalk.com
```

확인:

```text
GET http://배포주소/health
GET http://배포주소/docs
```

백엔드 호출:

```text
POST http://배포주소/analyze
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
