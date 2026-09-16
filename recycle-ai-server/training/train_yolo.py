# Colab에서 실행하는 재활용품목 YOLO 학습 스크립트.
#
# [2026-09-10 저녁] 품목별로 따로 모델을 만들지 않고, **여러 품목을 한 모델에 합쳐서
# 학습**하는 구조로 바꿈 - 실제 서비스에서는 이미지 한 장마다 품목 수만큼 모델을 따로
# 돌리면 추론 시간이 품목 수에 비례해서 늘어나서, 반응 속도(서버가 결과를 최대한 빨리
# 줘야 함)에 불리함. 모델 하나로 여러 클래스를 한 번에 인식하면 추론은 항상 1회로 끝남.
# 대신 품목을 추가할 때마다 지금까지의 모든 품목 데이터를 합쳐서 처음부터 다시
# 학습해야 함 (클래스 수가 바뀌면 기존 가중치를 이어쓸 수 없음 - YOLO 출력 레이어
# 구조 자체가 클래스 수에 종속적이라서).
#
# DATASET_FOLDERS에 나열된 품목 폴더를 각각 원본 그대로 받은 뒤, 클래스 인덱스를
# 전역으로 다시 매겨서(_merge_datasets) 하나의 데이터셋으로 합치고 그걸로 학습한다.
# 폴더가 하나만 있어도(지금처럼 페트병만 있는 상태) 그대로 동작함 - 나중에 새 품목
# 폴더를 추가할 땐 DATASET_FOLDERS에 한 줄만 추가하면 됨.
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
# ── 품목 추가할 때마다 해야 할 것 ──
#   1. 새 품목 폴더를 Drive에 원본 그대로 준비 (images/{train,val}, labels/{train,val},
#      data.yaml - data.yaml의 names는 이 폴더 안에서만 유효한 로컬 클래스 목록이면 됨,
#      다른 품목과 이름이 겹쳐도 상관없음. 전역 클래스 목록은 이 스크립트가 알아서 만듦).
#   2. 그 폴더(또는 상위 공유 폴더 - recycle_training_data 처럼 한 번 공유해두면
#      그 밑에 새로 만드는 폴더는 자동으로 서비스 계정에 보임)를 서비스 계정 이메일로
#      뷰어 권한 공유.
#   3. 아래 DATASET_FOLDERS 리스트에 {"id": ..., "name": ...} 한 줄 추가.
#   4. colab install -s ssok-yolo ultralytics google-auth google-auth-httplib2 google-api-python-client pyyaml
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
# 통합 모델에 포함시킬 품목 폴더 목록. 폴더 ID는 공유 링크의 /folders/ 뒤 부분.
# 새 품목 생기면 이 리스트에 한 줄만 추가하면 됨 (순서는 상관없음).
DATASET_FOLDERS = [
    {"id": "11bNM4kGP991WVJCiit1vMz-xEwib0tKK", "name": "pet_bottle"},
    # {"id": "<유리병 폴더 ID>", "name": "glass_bottle"},
]
RUN_NAME = "recycle_unified_v1"

# ── 보통 안 바꾸는 값 ──
SERVICE_ACCOUNT_KEY = "/content/service_account.json"
RAW_DIR = Path("/content/raw")
MERGED_DATASET_DIR = Path("/content/merged_dataset")
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


def _download_one_item(service, folder_id: str, dest_dir: Path) -> None:
    # 같은 세션 안에서 재실행하는 경우(직전 실행이 다운로드 이후 단계에서 죽은 경우)
    # data.yaml이 이미 있으면 재다운로드를 또 하지 않고 건너뜀.
    if (dest_dir / "data.yaml").exists():
        print(f"  {dest_dir.name}: 로컬에 이미 있어 다운로드를 건너뜁니다.")
        return

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    file_jobs = _collect_files_recursive(service, folder_id, dest_dir)
    print(f"  {dest_dir.name}: 총 {len(file_jobs)}개 파일, {DOWNLOAD_WORKERS}개씩 병렬 다운로드 시작...")

    completed = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(_download_one, file_id, dest) for file_id, dest in file_jobs]
        for future in as_completed(futures):
            future.result()  # 재시도까지 다 실패한 파일이 있으면 여기서 예외가 올라옴
            completed += 1
            if completed % 200 == 0:
                print(f"    {dest_dir.name}: {completed}/{len(file_jobs)} 완료")
    print(f"  {dest_dir.name}: 다운로드 완료.")


def _load_local_names(item_dir: Path) -> list[str]:
    with open(item_dir / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def _merge_datasets(item_names: list[str]) -> Path:
    # 품목마다 로컬 클래스 인덱스(0-based)가 따로 있어서, 전역 클래스 목록을 새로 만들고
    # 각 품목의 로컬 인덱스 -> 전역 인덱스 매핑으로 라벨 파일의 클래스 번호를 다시 씀.
    if MERGED_DATASET_DIR.exists():
        shutil.rmtree(MERGED_DATASET_DIR)
    for split in ("train", "val"):
        (MERGED_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    global_names: list[str] = []

    for item_name in item_names:
        item_dir = RAW_DIR / item_name
        local_names = _load_local_names(item_dir)

        local_to_global: dict[int, int] = {}
        for local_id, name in enumerate(local_names):
            if name not in global_names:
                global_names.append(name)
            local_to_global[local_id] = global_names.index(name)

        for split in ("train", "val"):
            src_images_dir = item_dir / "images" / split
            src_labels_dir = item_dir / "labels" / split
            if not src_images_dir.exists():
                continue

            for image_path in src_images_dir.iterdir():
                if not image_path.is_file():
                    continue
                # 다른 품목 데이터에서 같은 파일명이 나올 수 있어서 품목명을 접두어로 붙여
                # 충돌을 피한다 (예: pet_bottle__img001.jpg).
                dest_stem = f"{item_name}__{image_path.stem}"
                shutil.copy2(image_path, MERGED_DATASET_DIR / "images" / split / f"{dest_stem}{image_path.suffix}")

                label_path = src_labels_dir / f"{image_path.stem}.txt"
                dest_label_path = MERGED_DATASET_DIR / "labels" / split / f"{dest_stem}.txt"
                if not label_path.exists():
                    dest_label_path.write_text("", encoding="utf-8")  # 배경(객체 없음) 이미지
                    continue

                remapped_lines = []
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    parts[0] = str(local_to_global[int(parts[0])])
                    remapped_lines.append(" ".join(parts))
                dest_label_path.write_text(
                    "\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8"
                )

    data_yaml_path = MERGED_DATASET_DIR / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(MERGED_DATASET_DIR),
                "train": "images/train",
                "val": "images/val",
                "nc": len(global_names),
                "names": global_names,
            },
            f,
            allow_unicode=True,
        )
    print(f"병합 완료: 클래스 {len(global_names)}개 {global_names}")
    return data_yaml_path


def main() -> None:
    service = _build_drive_service()

    print("품목별 원본 데이터셋 다운로드 중...")
    for folder in DATASET_FOLDERS:
        _download_one_item(service, folder["id"], RAW_DIR / folder["name"])

    print("전역 클래스로 병합 중...")
    data_yaml_path = _merge_datasets([folder["name"] for folder in DATASET_FOLDERS])

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml_path),
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
