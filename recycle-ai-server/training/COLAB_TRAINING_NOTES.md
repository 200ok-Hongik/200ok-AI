# Colab 학습 기록

Colab에서 학습 돌릴 때마다 핵심 코드/명령어랑 "왜 이렇게 했는지"를 여기 남긴다.
코드만 봐서는 왜 이 방법을 골랐는지 까먹기 쉬워서, 시행착오 자체를 기록해둔다.

## [2026-09-10 11:47] PET 병 clear/colored YOLO 재학습 (yolov8n, 2클래스)

### 실행 명령어 (Colab CLI)
```bash
# 1. 세션 생성
colab new -s ssok-yolo --gpu T4

# 2. 필요한 패키지 설치
colab install -s ssok-yolo ultralytics google-auth google-auth-httplib2 google-api-python-client

# 3. 서비스 계정 키 업로드 (로컬 PC -> Colab 세션)
colab upload -s ssok-yolo "<로컬 키 파일 경로>.json" /content/service_account.json

# 4. 학습 실행 (최대 3시간 타임아웃)
colab exec -s ssok-yolo -f train_yolo.py --timeout 10800

# 5. (학습 끝난 뒤) 결과물 로컬로 받기
colab download -s ssok-yolo /content/runs/pet_bottle_clear_colored_v2/weights/best.pt
```
- `--gpu T4`: 공짜 등급에서 쓸 수 있는 가장 기본 GPU. L4/A100 등은 유료 등급 필요.
- `install`은 세션을 새로 만들 때마다 매번 다시 해야 함 (세션=새 VM이라 패키지가 안 남아있음).
- `--timeout 10800`: `colab exec`의 기본 타임아웃이 30초라, 학습처럼 오래 걸리는 작업엔 반드시 넉넉하게 잡아야 함. 안 그러면 학습 중간에 타임아웃으로 끊김.

### 핵심 코드 (`train_yolo.py`)

**1) 데이터 받아오는 방식 — 서비스 계정으로 Drive API 직접 호출**
```python
from google.oauth2 import service_account
from googleapiclient.discovery import build

def _get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)
```
- **왜 `drive.mount()`가 아닌지**: `google.colab.drive.mount()`는 브라우저 팝업으로 사람이 승인해야 하는 인터랙티브 인증인데, `colab exec`로 비대화형 실행하면 그 승인을 받을 방법이 없어서 매번 `mount failed`로 실패함. 브라우저 탭에서 따로 마운트해도 `exec`가 도는 커널과 공유가 안 됨(서로 다른 커널 프로세스로 추정).
- **왜 `gdown`도 아닌지**: `gdown.download_folder()`는 폴더 안 파일 목록을 나열할 때 **50개 넘으면 에러**(`FolderContentsMaximumLimitError`) — 우리 데이터셋은 이미지가 2천 장 넘어서 못 씀.
- 서비스 계정은 브라우저 승인이 전혀 필요 없고(키 파일 자체가 인증서 역할), 파일 개수 제한도 없음. 대신 **Drive에서 대상 폴더를 그 서비스 계정 이메일로 미리 공유**해둬야 함(사람 계정 초대하듯이).

**2) 폴더 전체를 재귀적으로 받는 함수 (페이지네이션 포함)**
```python
def _list_children(service, folder_id: str) -> list[dict]:
    items: list[dict] = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return items
```
- Drive API 한 번 호출로는 최대 1000개까지만 나옴 — `pageToken`으로 끝까지 순회해야 2천 개+ 파일을 다 가져올 수 있음. 이거 빠뜨리면 조용히 일부만 받아지고 끝남(에러도 안 남).
- 파일 하나하나를 개별 API 호출로 받다 보니(진행 로그 없이) 2천 장 받는 데 실제로 10분 넘게 걸림 — 오래 걸려도 정상, 죽은 게 아님.

**3) 학습 파라미터 — 추측 아니고 실제 mAP50 0.988 냈던 값 그대로**
```python
model.train(
    data=str(LOCAL_DATASET_DIR / "data.yaml"),
    epochs=50, imgsz=640, batch=16,
    degrees=20, translate=0.1, scale=0.2,
    cache=True, patience=15,
    project=RUNS_DIR, name=RUN_NAME,
)
```
- 이 값들은 지어낸 게 아니라, `split_train_val_0811.ipynb`의 실제 실행 로그(성공한 학습, mAP50 0.988)에서 그대로 가져온 것.
- `project`(결과 저장 위치)는 Drive가 아니라 **로컬(`/content/runs`)** — 위 마운트 문제 때문에 Drive에 직접 못 씀. 학습 끝나면 `colab download`로 따로 받아야 함.

### 다음에 비슷한 학습할 때 참고할 점
- 새 품목(캔·유리병 등) 데이터를 Drive에 올릴 때, **그 폴더를 서비스 계정 이메일(`recycle-training-bot@ssok-ai-data.iam.gserviceaccount.com`)로 다시 공유**해야 함 — 상위 폴더(`recycle_pet_project`)가 서비스 계정에 공유돼 있지 않아서, 하위에 새로 만든 폴더는 자동으로 안 보임.
- 클래스 개수가 바뀌면(새 품목 추가) YOLO 출력 레이어 구조 자체가 바뀌어서 **기존 `best.pt` 이어학습 불가** — `yolov8n.pt`에서 매번 새로 시작해야 함.
- Colab 세션은 방치하면 자연히 끊김(우리도 중간에 한 번 끊겨서 세션 새로 만들었음) — `colab status`로 살아있는지 먼저 확인하고 작업 시작할 것.

## [2026-09-10 오후] 세션이 자주 끊기는 문제 - 체크포인트 자동 백업 + 이어학습으로 대응

2026-09-09~10 사이 `ssok-yolo` 세션이 `KEEP: error ... 404`/`SSLError`로 하루에도
몇 번씩 죽는 걸 확인함(`colab log -s ssok-yolo`로 이력 확인 가능). Colab 무료 등급
인프라 쪽 문제로 보여서 완전히 막을 방법은 없음 — 대신 **끊겨도 이어서 학습**할 수
있게 `train_yolo.py`를 고침.

- **체크포인트 자동 백업**: `on_train_epoch_end` 콜백으로 `CHECKPOINT_EVERY_EPOCHS`(5)
  에폭마다 `last.pt`를 Drive의 데이터셋 폴더 밑 `_checkpoints/`에 업로드. 세션이 죽어도
  최근 5에폭 이내 상태는 항상 Drive에 남아있음.
- **자동 이어학습**: 스크립트 시작 시 `_checkpoints/classes.json`(직전 학습의 클래스 목록)과
  지금 데이터셋의 클래스가 같으면 `_checkpoints/last.pt`를 받아서 `resume=True`로 이어서
  학습. 클래스 구성이 다르면(품목 추가로 바뀐 경우) 자동으로 `yolov8n.pt`부터 새로 시작.
- **다운로드 재시도**: 파일 하나 실패해도 전체가 죽지 않게 파일당 최대 3회 재시도 추가
  (데이터가 계속 늘어날수록 개별 파일 실패 확률도 올라가서 필요해짐).
- **⚠️ 필수 조치**: 체크포인트를 Drive에 "쓰려면" 서비스 계정 권한이 기존 **뷰어**로는
  안 되고 **편집자(Editor)**여야 함. 기존에 뷰어로 공유해뒀던 `yolo_direct_v2` 폴더를
  편집자로 바꿔줘야 다음 실행부터 체크포인트 백업이 동작함 (안 바꾸면 업로드 시점에
  권한 오류로 실패 - 다만 학습 자체는 계속 진행됨, 체크포인트 백업만 실패).
- 새 품목 폴더를 공유할 때도 처음부터 편집자 권한으로 공유할 것.
