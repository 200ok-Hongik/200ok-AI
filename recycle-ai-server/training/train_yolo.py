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
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
from ultralytics import YOLO

# ── 품목별로 바꾸는 값 ──
# 통합 모델에 포함시킬 품목 폴더 목록. 폴더 ID는 공유 링크의 /folders/ 뒤 부분.
# 새 품목 생기면 이 리스트에 한 줄만 추가하면 됨 (순서는 상관없음).
#
# [2026-09-22] yolo_direct_v2(옛 pet_bottle, pet_bottle_clear/pet_bottle_colored 로컬 이름)는
# clear_pet_bottle_v1/plastic_v1로 대체됐음이 09-21 세션에서 확인돼서 제거함 - 둘 다 남기면
# 같은 클래스 데이터가 중복으로 섞여 들어감. 아래 8개는 recycle_training_data/ 밑에서
# data.yaml이 이미 전역 이름(CLEAR_PET_BOTTLE/PLASTIC/CAN/GLASS_BOTTLE/VINYL/PAPER/STYROFOAM/
# PAPER_PACK)을 쓰고 있어서 name_remap이 필요 없음 (서비스 계정 API로 실제 폴더 ID 조회 +
# data.yaml/라벨 클래스 인덱스까지 직접 확인함).
# paper_v1/styrofoam_v1/paper_pack_v1은 09-22 업로드 완료 확인 후 등록.
DATASET_FOLDERS = [
    {"id": "1w2U1fRcaC-i4FiyAEDCWzjcy9b7vRe5q", "name": "clear_pet_bottle"},
    {"id": "1U7-roBW_Da60mv_NKNor0cmyDo-QFZRY", "name": "plastic"},
    {"id": "1Fx_E0S4-mEBzaeOPeJkLo4aqPZEvTwJz", "name": "can"},
    {"id": "1stAyPAFyfaUUuPVAdkxZpvEcD4pe9d4u", "name": "glass_bottle"},
    {"id": "1bJUnDeyo6RGGQ8xHKhP1XIIO8NxA08_q", "name": "vinyl"},
    {"id": "1tfk6yJwn5xB7nd7EeGUxCLflQmBksXPu", "name": "paper"},
    {"id": "1ligVYT__-LpJcdzLTFjK7iK4BmtOw7v4", "name": "styrofoam"},
    {"id": "1Wv5hm0m3Vj0k8Ufm-CL0XPlx1VOsYop1", "name": "paper_pack"},
    # [2026-11-10] 생활 환경 사진 보강 - 위 8개(AIHub 선별장 사진)만으로는 책상·바닥 사진에서 종이컵을 못 찾고
    # 벽 그림을 PAPER로 오탐했음. 아래 폴더는 training/convert_taco.py, convert_roboflow.py로 만들어 올린 것으로,
    # data.yaml names가 8품목 전부라(한 사진에 여러 품목 박스가 같이 있음) 위 폴더와 같은 이름끼리 자동 병합됨.
    # 폴더 ID·사진/라벨 수는 서비스 계정 API로 조회해 PC 변환 결과와 일치함을 확인함.
    # 출처: TACO(Zenodo 10.5281/zenodo.3587843, CC BY 4.0),
    #       Roboflow Universe school2-iqyyf/paper-pack, varyrecycle/cardboard-paper (둘 다 CC BY 4.0)
    {"id": "1rgaFW8izIhICBO-Mv0Oa45ZL2n8zxqEY", "name": "can_taco"},
    {"id": "1BKvujEjutDvdo-kKEVVzOyxTbw_K30Vq", "name": "clear_pet_bottle_taco"},
    {"id": "12Z0hA9tcaQg-gWqXprtg8smeb0dftwIH", "name": "glass_bottle_taco"},
    {"id": "1ypWj0TQgp4XgwzDf2whMvHsoPSsCmZG9", "name": "paper_pack_taco"},
    {"id": "1S74Xepq8V_heKfmJKG9C6XQrax-AXQM0", "name": "paper_taco"},
    {"id": "1aHRsdHWDKh8ohv1guBugi0PQD7ld6xeM", "name": "plastic_taco"},
    {"id": "1FRePm97ddS6hnSADS9UOCq2A2U_ar_5h", "name": "styrofoam_taco"},
    {"id": "14FMMIsGe8XfmiUh1YIaL3EpG4MDajDVR", "name": "vinyl_taco"},
    {"id": "1-wGOWX9SY5k2gfcNKP6Sy7GECIApCTfe", "name": "background_taco"},  # 물체 없는 사진 - 배경 오탐 줄이기
    {"id": "174rJRooHiaBhI7JbqnG1Slrsdev2ofN_", "name": "paper_pack_roboflow"},  # 한국 우유팩, 929장
    # 품질이 가장 낮음(질감·회전 사진 일부 남음) - 새 모델에서 종이 오탐이 늘면 이 줄부터 빼고 다시 학습
    {"id": "1sXfQpDtwAdaaNhh5BMXYrM5clDoG_2AF", "name": "paper_roboflow"},
    # [2026-11-12] v2 가 종이컵을 PAPER_PACK 으로 오분류(우유팩 929장 vs TACO 종이컵 62장) → 종이컵 294장을 PAPER 로 보충.
    # 팀 기준: 종이컵 = PAPER (환경부 별표1에서도 종이팩과 다른 "골판지 외 종이류" 세부품목).
    # 출처: Roboflow Universe workspace-jyj9a/papercup-34ue8 (CC BY 4.0), convert_roboflow.py --map papercup=PAPER
    {"id": "1biDge9OXiEadKTXzZ_ytlp6ERhVFQGDb", "name": "paper_cup_roboflow"},
    # [2026-11-24] v4 보충 - 생활사진 실험(bench/sets/life_mix 103장)에서 스티로폼→종이(6장 중 4장), 비닐 오인식(6장),
    # 유리병 부족이 주요 YOLO 오류였음. 셋 다 Roboflow Universe CC BY 4.0, convert_roboflow.py(11-24 중복 박스 제거·
    # 원본 단위 분할 적용)로 변환. 유리병은 라벨 없이 캔·페트 등이 같이 찍힌 원본 23개(47장)를 v3 모델로 찾아 제외함.
    #   styrofoam: namseoul-qobk6/styrofoam-qt1kt v1 (한국 바닥 위 흰 트레이·박스) 1,092장
    #   vinyl: new-workspace-of0a7/vinyl-nvzjy v3 (한국 바닥 위 비닐봉지, 원본 500·회전 증강 포함) 962장
    #   glass: can-jniv8/glass-bottle-rxvbe v12 (해외 사진 섞임, 박스 3개 이하만) 609장
    {"id": "1SXCrPC3jb15cjZyjdcJiEiH1_uqovwjh", "name": "vinyl_roboflow"},
    {"id": "1ICBfMI_3MUZb-wriWYrX25gvf42_yNM_", "name": "styrofoam_roboflow"},  # [11-24] 서비스 계정으로 폴더 이름 확인함 (링크 순서가 비닐·스티로폼 반대였음)
    {"id": "1IThAW8IlX9RT9W6x4YVPum0tV8VWsF0Y", "name": "glass_roboflow"},
]
RUN_NAME = "recycle_unified_v4"  # [2026-11-24] v3 → v4 (스티로폼·비닐·유리병 보충). v3=배포 중(11-13 PR #12), v2=11-12 46에폭

# [2026-11-12] v3 파인튜닝 설정 - v2 에서 이어서 짧게 학습 (Colab 세션이 20분~1.5시간 만에 끊겨 50에폭 재학습은 위험)
START_WEIGHTS = Path("/content/start.pt")  # [11-24] PC 에서 올리는 v3 가중치 (배포 중인 models/best.pt 와 같은 것)
FINETUNE_EPOCHS = 20
# v2 캐시 ZIP 에 이미 들어 있는 폴더 - 이 목록에 없는 DATASET_FOLDERS 만 새로 받아 덧붙임 (_prepare_dataset)
BASE_CACHE_FOLDERS = {
    "clear_pet_bottle", "plastic", "can", "glass_bottle", "vinyl", "paper", "styrofoam", "paper_pack",
    "can_taco", "clear_pet_bottle_taco", "glass_bottle_taco", "paper_pack_taco", "paper_taco", "plastic_taco",
    "styrofoam_taco", "vinyl_taco", "background_taco", "paper_pack_roboflow", "paper_roboflow",
    "paper_cup_roboflow",  # [11-24] v3 합본에 들어 있음
}

# ── 보통 안 바꾸는 값 ──
SERVICE_ACCOUNT_KEY = "/content/service_account.json"
RAW_DIR = Path("/content/raw")
MERGED_DATASET_DIR = Path("/content/merged_dataset")
RUNS_DIR = "/content/runs"
RESUME_CHECKPOINT = Path("/content/last.pt")  # 있으면 이어서 학습 (main 참고)
# [2026-11-11] colab drivemount 로 혜원 Drive 를 붙였을 때만 쓰는 캐시 위치 (서비스 계정은 Drive 에 쓸 수 없음)
DRIVE_MOUNT = Path("/content/drive/MyDrive")
CACHE_DIR = DRIVE_MOUNT / "ssok_cache"
DRIVE_LAST = CACHE_DIR / f"{RUN_NAME}_last.pt"
BASE_CACHE_ZIP = CACHE_DIR / "recycle_unified_v3_merged_dataset.zip"  # [2026-11-24] v4 가 덧붙여 쓸 v3 합본 (11-12 v2 → v3)

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
# 순차 다운로드는 파일당 ~1.5초라 병렬로 돌림. [2026-11-10] 8 → 24: 8로는 기존 8폴더(19,184개, 25.6GB)를 받는 동안
# Colab 무료 세션이 끊겼음. 32로 올렸더니 Drive 분당 요청 한도(403)에 걸려서 24로 낮추고 재시도 대기를 늘림.
DOWNLOAD_WORKERS = 24
MAX_IMAGE_SIDE = 1280  # 받은 사진을 이 크기로 줄여 저장 (_shrink_image)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_thread_local = threading.local()
DOWNLOAD_RETRIES = 8  # [2026-11-10] 3 → 8 (레이트리밋 대기 포함)
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


def _thread_drive_service():
    # httplib2.Http는 스레드 세이프하지 않아서 스레드마다 따로 만들되, [2026-11-10] 파일마다 새로 만들던 것을
    # 스레드당 한 번으로 바꿈 - 파일 19,000개를 받는 동안 접속 준비를 19,000번 해서 다운로드가 크게 느려졌음.
    if not hasattr(_thread_local, "service"):
        _thread_local.service = _build_drive_service()
    return _thread_local.service


def _shrink_image(path: Path, suffix: str | None = None) -> None:
    """[2026-11-10] 받은 사진을 긴 변 MAX_IMAGE_SIDE 로 줄인다. 기존 AIHub 원본이 장당 3~4MB(8폴더 25.6GB)라
    디스크·시간이 많이 들었는데, 학습은 imgsz=640 이라 1280 이면 충분함. YOLO 라벨은 0~1 비율이라 그대로 유효.
    EXIF(회전 정보)는 그대로 붙여 저장 - 픽셀 방향을 안 바꾸고 크기만 줄이므로 라벨과 방향 관계가 유지됨."""
    if (suffix or path.suffix).lower() not in IMAGE_SUFFIXES:  # .part 로 받는 중이라 원래 확장자를 따로 받음
        return
    try:
        with Image.open(path) as image:
            if max(image.size) <= MAX_IMAGE_SIDE:
                return
            exif = image.info.get("exif")
            image = image.convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            temp = path.with_suffix(path.suffix + ".tmp")
            image.save(temp, format="JPEG", quality=92, **({"exif": exif} if exif else {}))
        temp.replace(path)
    except Exception as exc:  # noqa: BLE001 - 줄이기 실패는 원본 그대로 학습에 씀 (학습 자체는 가능)
        print(f"    (줄이기 건너뜀) {path.name}: {exc}")


def _download_one(file_id: str, dest: Path) -> None:
    service = _thread_drive_service()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():  # [2026-11-10] 같은 세션에서 중간에 멈춘 뒤 다시 돌리면 받은 파일은 건너뜀
        return
    part = dest.with_name(dest.name + ".part")  # 받다 끊긴 파일이 완료 파일로 보이지 않게 임시 이름으로 받음
    # 파일 개수가 늘어날수록 개별 요청 실패(일시적 네트워크/레이트리밋) 확률도 같이 올라가서
    # 전체를 죽이지 않게 재시도를 넣음.
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            request = service.files().get_media(fileId=file_id)
            with io.FileIO(part, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            _shrink_image(part, dest.suffix)
            part.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - Drive API 예외 타입이 다양해서 폭넓게 잡음
            last_error = exc
            # [2026-11-10] 2·4·6초 고정 대기로는 Drive "분당 요청 한도"(403 rateLimitExceeded)가 풀리기 전에 3회가
            # 다 끝나 전체가 죽었음 → 지수 대기(최대 60초) + 무작위 지연으로 스레드들이 동시에 다시 몰리지 않게 함.
            time.sleep(min(60, 2 ** attempt) + random.uniform(0, 3))
    raise RuntimeError(f"{dest.name} 다운로드 {DOWNLOAD_RETRIES}회 실패") from last_error


def _download_one_item(service, folder_id: str, dest_dir: Path) -> None:
    # 같은 세션 안에서 재실행하는 경우(직전 실행이 다운로드 이후 단계에서 죽은 경우)
    # data.yaml이 이미 있으면 재다운로드를 또 하지 않고 건너뜀.
    # [2026-11-10] data.yaml 대신 완료 표시 파일로 판단 - 파일을 병렬로 받아 data.yaml 이 먼저 받아질 수 있어서,
    # 중간에 끊긴 폴더를 "다 받음"으로 잘못 건너뛸 수 있었음. 끊긴 폴더는 지우지 않고 _download_one 이 이어받음.
    done_marker = dest_dir / ".download_complete"
    if done_marker.exists():
        print(f"  {dest_dir.name}: 로컬에 이미 있어 다운로드를 건너뜁니다.")
        return

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
    done_marker.touch()
    print(f"  {dest_dir.name}: 다운로드 완료.")


def _load_local_names(item_dir: Path, name_remap: dict[str, str] | None = None) -> list[str]:
    with open(item_dir / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    local_names = list(names.values()) if isinstance(names, dict) else list(names)
    if name_remap:
        local_names = [name_remap.get(name, name) for name in local_names]
    return local_names


def _merge_datasets(dataset_folders: list[dict], append: bool = False) -> Path:
    # 품목마다 로컬 클래스 인덱스(0-based)가 따로 있어서, 전역 클래스 목록을 새로 만들고
    # 각 품목의 로컬 인덱스 -> 전역 인덱스 매핑으로 라벨 파일의 클래스 번호를 다시 씀.
    # [2026-11-12] append=True: 캐시에서 푼 기존 합본은 지우지 않고, 그 data.yaml 클래스 순서를 그대로 이어받아
    # 새 폴더만 덧붙임 (v2 가중치의 클래스 순서와 같아야 이어서 학습 가능 - 폴더마다 PAPER_PACK/STYROFOAM 순서가 다름)
    if MERGED_DATASET_DIR.exists() and not append:
        shutil.rmtree(MERGED_DATASET_DIR)
    for split in ("train", "val"):
        (MERGED_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    global_names: list[str] = _load_local_names(MERGED_DATASET_DIR) if append else []

    for folder in dataset_folders:
        item_name = folder["name"]
        item_dir = RAW_DIR / item_name
        local_names = _load_local_names(item_dir, name_remap=folder.get("name_remap"))

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


def _prepare_dataset() -> Path:
    """[2026-11-11] 합친 데이터셋을 준비한다. Drive 가 마운트돼 있고 캐시 ZIP 이 있으면 그걸 풀고(1~3분),
    없으면 품목 폴더를 받아 합친 뒤 캐시 ZIP 을 Drive 에 남긴다 - 무료 Colab 세션이 1시간 안팎으로 끊기는데
    다운로드(19,000여 파일)에만 50분이 걸려 세션마다 학습할 시간이 거의 없었기 때문."""
    cache_zip = CACHE_DIR / f"{RUN_NAME}_merged_dataset.zip"
    if cache_zip.exists():
        _restore_cache_zip(cache_zip)
        return MERGED_DATASET_DIR / "data.yaml"

    service = _build_drive_service()
    # [2026-11-12] 이전 런(v2) 캐시가 있으면 그걸 풀고, 그 캐시에 없는 폴더(예: paper_cup_roboflow)만 받아 덧붙임
    # → 19,000여 파일 재다운로드(50분) 없이 몇 분 만에 v3 데이터셋 준비
    extra_folders = [f for f in DATASET_FOLDERS if f["name"] not in BASE_CACHE_FOLDERS]
    if BASE_CACHE_ZIP.exists() and len(extra_folders) < len(DATASET_FOLDERS):
        _restore_cache_zip(BASE_CACHE_ZIP)
        print(f"추가 폴더만 다운로드: {[f['name'] for f in extra_folders]}")
        for folder in extra_folders:
            _download_one_item(service, folder["id"], RAW_DIR / folder["name"])
        data_yaml_path = _merge_datasets(extra_folders, append=True)
    else:
        print("품목별 원본 데이터셋 다운로드 중...")
        for folder in DATASET_FOLDERS:
            _download_one_item(service, folder["id"], RAW_DIR / folder["name"])
        print("전역 클래스로 병합 중...")
        data_yaml_path = _merge_datasets(DATASET_FOLDERS)

    if DRIVE_MOUNT.exists():
        print(f"다음 세션용 캐시 저장: {cache_zip}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        archive = shutil.make_archive("/content/merged_dataset", "zip", MERGED_DATASET_DIR)  # 사진은 이미 JPEG 라 압축 이득은 적음
        shutil.copy(archive, cache_zip.with_suffix(".zip.part"))
        cache_zip.with_suffix(".zip.part").replace(cache_zip)  # 다 복사된 뒤에만 캐시로 보이게
        Path(archive).unlink()
    return data_yaml_path


def _restore_cache_zip(zip_path: Path) -> None:
    """Drive 캐시 ZIP 을 /content/merged_dataset 로 푼다 (FUSE 위에서 바로 풀면 느려서 로컬로 복사 후 풂)."""
    print(f"캐시에서 데이터셋 복원: {zip_path}")
    if MERGED_DATASET_DIR.exists():
        shutil.rmtree(MERGED_DATASET_DIR)
    local_zip = Path("/content/merged_dataset.zip")
    shutil.copy(zip_path, local_zip)
    shutil.unpack_archive(local_zip, MERGED_DATASET_DIR)
    local_zip.unlink()


def _backup_last_to_drive(trainer) -> None:
    """매 에폭 저장 직후 last.pt 를 Drive 캐시에도 복사 (세션이 끊겨도 마지막 에폭부터 이어갈 수 있게)."""
    last = Path(trainer.save_dir) / "weights" / "last.pt"
    if DRIVE_MOUNT.exists() and last.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(last, DRIVE_LAST)


def main() -> None:
    data_yaml_path = _prepare_dataset()

    # [2026-11-11] 무료 Colab 세션이 학습 도중(2026-11-10 29/50에폭, 11-11 15/50에폭) 끊기면서 결과가 사라졌음.
    # /content/last.pt(PC 에서 올린 것) 또는 Drive 캐시의 last.pt 가 있으면 그 에폭부터 이어서 학습함.
    # (데이터셋은 위에서 같은 경로로 준비하므로, last.pt 안에 저장된 data 경로가 그대로 유효함)
    resume_from = RESUME_CHECKPOINT if RESUME_CHECKPOINT.exists() else DRIVE_LAST if DRIVE_LAST.exists() else None
    if resume_from:
        print(f"{resume_from} 에서 이어서 학습합니다.")
        model = YOLO(str(resume_from))
        model.add_callback("on_model_save", _backup_last_to_drive)
        model.train(resume=True)
        return

    # [2026-11-12] /content/start.pt(이전 런 결과, 예: v2 46에폭)가 있으면 그 가중치에서 짧게 추가 학습(파인튜닝).
    # 클래스 8개·순서가 같아서 출력층을 그대로 이어 쓸 수 있음. 없으면 예전처럼 yolov8n 에서 처음부터 50에폭.
    finetune = START_WEIGHTS.exists()
    print(f"{START_WEIGHTS} 에서 파인튜닝 {FINETUNE_EPOCHS}에폭" if finetune else "yolov8n 에서 처음부터 50에폭")
    model = YOLO(str(START_WEIGHTS) if finetune else "yolov8n.pt")
    model.add_callback("on_model_save", _backup_last_to_drive)
    model.train(
        data=str(data_yaml_path),
        epochs=FINETUNE_EPOCHS if finetune else 50,
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
