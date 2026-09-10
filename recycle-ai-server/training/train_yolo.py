# Colab에서 실행하는 재활용품목 YOLO 학습 스크립트.
#
# [2026-09-10] 체크포인트를 서비스 계정으로 Drive에 직접 업로드하는 방식을 시도했다가
# 제거함 - 실제로 돌려보니 "Service Accounts do not have storage quota" 403 에러로
# 죽었음. 개인 Gmail Drive에서는 서비스 계정이 자기 소유로 새 파일을 만들 저장 공간이
# 아예 없어서(공유 드라이브는 Workspace 전용이라 우리 계정엔 없음), 심지어 편집자
# 권한으로 공유된 폴더 안이어도 새 파일 업로드가 막힘. 그래서 체크포인트 보호는
# Drive가 아니라 **로컬 PC 쪽에서 `colab download`로 주기적으로 last.pt를 받아오는
# 방식**으로 뺐음 (이 스크립트 밖, orchestration 쪽 책임) - 이 스크립트는 그냥
# Ultralytics가 기본으로 하는 대로 /content/runs에 매 에폭 last.pt를 남기기만 하면 됨.
#
# 데이터 다운로드는 drive.mount()(인터랙티브 인증이 exec 커널에서 계속 실패함)나
# gdown(폴더당 50개 파일 제한에 걸림) 대신 서비스 계정으로 Drive API를 직접 호출하는
# 방식을 씀 - 사람 개입 없이, 개수 제한 없이 받을 수 있음 (읽기 전용이라 뷰어 공유로 충분).
#
# ── 품목 추가/데이터가 늘어날 때마다 해야 할 것 ──
#   1. Drive의 데이터셋 폴더를 서비스 계정 이메일로 뷰어 권한 공유 (읽기만 하면 되니 뷰어로 충분).
#      상위 폴더가 공유돼 있어도 새로 만든 하위 폴더는 자동으로 안 보이니, 새 품목
#      폴더는 그 폴더 자체를 다시 공유해야 함.
#   2. 아래 "품목별로 바꾸는 값"의 DATASET_FOLDER_ID / RUN_NAME을 새 품목에 맞게 수정.
#   3. colab install -s ssok-yolo ultralytics google-auth google-auth-httplib2 google-api-python-client
#
# Colab CLI 실행 예:
#   colab exec -s ssok-yolo -f train_yolo.py --timeout 10800
#   (끝나면) colab download -s ssok-yolo /content/runs/<RUN_NAME>/weights/best.pt
import io
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from ultralytics import YOLO

# ── 품목별로 바꾸는 값 ──
# yolo_direct_v2 폴더의 Drive 파일 ID (공유 링크의 /folders/ 뒤 부분)
DATASET_FOLDER_ID = "11bNM4kGP991WVJCiit1vMz-xEwib0tKK"
RUN_NAME = "pet_bottle_clear_colored_v2"

# ── 보통 안 바꾸는 값 ──
SERVICE_ACCOUNT_KEY = "/content/service_account.json"
LOCAL_DATASET_DIR = Path("/content/yolo_direct_v2")
RUNS_DIR = "/content/runs"

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
# 순차 다운로드는 파일당 ~1.5초라 병렬로 돌림. 너무 높이면 Drive API 레이트리밋에 걸릴 수 있어 8로 둠.
DOWNLOAD_WORKERS = 8
DOWNLOAD_RETRIES = 3
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _build_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY, scopes=DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=creds)


def _list_children(service, folder_id: str) -> list[dict]:
    # Drive API 기본 페이지 크기(100)를 넘는 폴더(데이터셋이 계속 늘어나는 중)를 다 받으려면
    # pageToken으로 끝까지 순회해야 함.
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


def _collect_files_recursive(service, folder_id: str, dest_dir: Path) -> list[tuple[str, Path]]:
    file_jobs: list[tuple[str, Path]] = []
    for item in _list_children(service, folder_id):
        item_dest = dest_dir / item["name"]
        if item["mimeType"] == FOLDER_MIME_TYPE:
            file_jobs.extend(_collect_files_recursive(service, item["id"], item_dest))
        else:
            file_jobs.append((item["id"], item_dest))
    return file_jobs


def _download_one(file_id: str, dest: Path) -> None:
    # httplib2.Http는 스레드 세이프하지 않아서, 스레드마다 별도 service 인스턴스를 새로 만든다.
    service = _build_drive_service()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 파일 개수가 늘어날수록 개별 요청 실패(일시적 네트워크/레이트리밋) 확률도 같이 올라가서
    # 전체를 죽이지 않게 재시도를 넣음.
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            request = service.files().get_media(fileId=file_id)
            with io.FileIO(dest, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            return
        except Exception as exc:  # noqa: BLE001 - Drive API 예외 타입이 다양해서 폭넓게 잡음
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"{dest.name} 다운로드 {DOWNLOAD_RETRIES}회 실패") from last_error


def _download_dataset(service) -> None:
    # 같은 세션 안에서 재실행하는 경우(직전 실행이 다운로드 이후 단계에서 죽은 경우)
    # data.yaml이 이미 있으면 8천 장 넘는 재다운로드를 또 하지 않고 건너뜀.
    if (LOCAL_DATASET_DIR / "data.yaml").exists():
        print("로컬에 데이터셋이 이미 있어 다운로드를 건너뜁니다.")
        return

    if LOCAL_DATASET_DIR.exists():
        shutil.rmtree(LOCAL_DATASET_DIR)

    print("파일 목록 조회 중...")
    file_jobs = _collect_files_recursive(service, DATASET_FOLDER_ID, LOCAL_DATASET_DIR)
    print(f"총 {len(file_jobs)}개 파일, {DOWNLOAD_WORKERS}개씩 병렬 다운로드 시작...")

    completed = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(_download_one, file_id, dest) for file_id, dest in file_jobs]
        for future in as_completed(futures):
            future.result()  # 재시도까지 다 실패한 파일이 있으면 여기서 예외가 올라옴
            completed += 1
            if completed % 200 == 0:
                print(f"  {completed}/{len(file_jobs)} 완료")
    print("데이터셋 다운로드 완료.")


def _fix_data_yaml_root() -> None:
    # yolo_direct_v2/data.yaml은 원래 drive.mount() 기준 절대경로(/content/drive/MyDrive/...)로
    # 만들어져 있어서, 서비스 계정으로 받은 실제 로컬 경로(LOCAL_DATASET_DIR)와 안 맞아
    # "images not found"로 실패함 - 다운로드 위치에 맞게 path를 덮어써야 함.
    yaml_path = LOCAL_DATASET_DIR / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(LOCAL_DATASET_DIR)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


def main() -> None:
    service = _build_drive_service()
    _download_dataset(service)
    _fix_data_yaml_root()

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(LOCAL_DATASET_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        degrees=20,
        translate=0.1,
        scale=0.2,
        cache=True,
        patience=15,
        project=RUNS_DIR,
        name=RUN_NAME,
    )


if __name__ == "__main__":
    main()
