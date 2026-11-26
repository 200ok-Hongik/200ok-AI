# 2026-11-10: train_yolo._merge_datasets 검증 - 기존 폴더(data.yaml names가 자기 품목 1개)와 새 보강 폴더
# (names가 8품목 전부)를 섞어도 같은 품목 이름끼리 같은 전역 번호로 합쳐지는지 확인한다.
# 번호가 어긋나면 학습은 돌아가도 캔 박스를 종이로 배우는 식이 되므로 Colab에 올리기 전에 확인.
import pytest
import yaml

pytest.importorskip("googleapiclient")  # train_yolo 가 Drive API 패키지를 import 함 (로컬 venv 에만 설치)
from training import train_yolo  # noqa: E402

EIGHT = ["CLEAR_PET_BOTTLE", "PLASTIC", "CAN", "GLASS_BOTTLE", "VINYL", "PAPER", "PAPER_PACK", "STYROFOAM"]


def _folder(root, name, names, labels):
    for sub in ("images/train", "labels/train"):
        (root / name / sub).mkdir(parents=True)
    (root / name / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for stem, lines in labels.items():
        (root / name / "images/train" / f"{stem}.jpg").write_bytes(b"x")
        (root / name / "labels/train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def test_Drive_캐시_ZIP이_있으면_다운로드없이_데이터셋을_복원한다(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    src = tmp_path / "src"
    (src / "labels/train").mkdir(parents=True)
    (src / "data.yaml").write_text("names: [CAN]\n")
    (src / "labels/train/a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.make_archive(str(cache / f"{train_yolo.RUN_NAME}_merged_dataset"), "zip", src)

    merged = tmp_path / "merged"
    monkeypatch.setattr(train_yolo, "CACHE_DIR", cache)
    monkeypatch.setattr(train_yolo, "MERGED_DATASET_DIR", merged)
    monkeypatch.setattr(train_yolo, "Path", lambda p: tmp_path / Path(p).name if p == "/content/merged_dataset.zip" else Path(p))
    monkeypatch.setattr(train_yolo, "_build_drive_service", lambda: (_ for _ in ()).throw(AssertionError("다운로드하면 안 됨")))

    data_yaml = train_yolo._prepare_dataset()
    assert data_yaml == merged / "data.yaml" and data_yaml.exists()
    assert (merged / "labels/train/a.txt").read_text() == "0 0.5 0.5 0.1 0.1\n"


def test_큰_사진은_비율과_EXIF회전을_유지한채_줄인다(tmp_path):
    from PIL import Image

    path = tmp_path / "big.jpg.part"  # 다운로드 중 임시 이름
    exif = Image.Exif()
    exif[0x0112] = 6  # 세로 사진 회전 표시
    Image.new("RGB", (4000, 3000)).save(path, format="JPEG", exif=exif)

    train_yolo._shrink_image(path, ".jpg")
    with Image.open(path) as image:
        assert image.size == (1280, 960)  # 4:3 유지 → YOLO 비율 좌표 그대로 유효
        assert image.getexif().get(0x0112) == 6  # 회전 정보 유지 → 라벨과 방향 관계 그대로


def test_작은_사진과_라벨파일은_건드리지_않는다(tmp_path):
    from PIL import Image

    small = tmp_path / "s.jpg"
    Image.new("RGB", (640, 480)).save(small)
    before = small.read_bytes()
    train_yolo._shrink_image(small)
    assert small.read_bytes() == before

    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n")
    train_yolo._shrink_image(label)
    assert label.read_text() == "0 0.5 0.5 0.1 0.1\n"


def test_기존_1품목_폴더와_8품목_보강폴더가_같은_이름끼리_합쳐진다(tmp_path, monkeypatch):
    raw, merged = tmp_path / "raw", tmp_path / "merged"
    monkeypatch.setattr(train_yolo, "RAW_DIR", raw)
    monkeypatch.setattr(train_yolo, "MERGED_DATASET_DIR", merged)
    _folder(raw, "can", ["CAN"], {"a": ["0 0.5 0.5 0.1 0.1"]})                       # 기존 can_v1 형식
    _folder(raw, "paper", ["PAPER"], {"b": ["0 0.5 0.5 0.1 0.1"]})                   # 기존 paper_v1 형식
    _folder(raw, "can_taco", EIGHT, {"c": ["2 0.5 0.5 0.1 0.1", "5 0.2 0.2 0.1 0.1"]})  # 보강: CAN + PAPER
    _folder(raw, "background_taco", EIGHT, {"d": []})                                # 배경 사진

    data_yaml = train_yolo._merge_datasets([{"name": n} for n in ("can", "paper", "can_taco", "background_taco")])
    names = yaml.safe_load(data_yaml.read_text())["names"]
    assert len(names) == 8 and len(set(names)) == 8  # 9번째 품목이 생기지 않음

    def classes(stem):
        text = (merged / "labels/train" / f"{stem}.txt").read_text()
        return [names[int(line.split()[0])] for line in text.splitlines() if line]

    assert classes("can__a") == ["CAN"]
    assert classes("paper__b") == ["PAPER"]
    assert classes("can_taco__c") == ["CAN", "PAPER"]
    assert classes("background_taco__d") == []


# [2026-11-12] v3 파인튜닝용: v2 캐시(클래스 순서 = 모델 순서, STYROFOAM=6/PAPER_PACK=7)를 풀고
# 클래스 순서가 다른 새 폴더(PAPER_PACK=6/STYROFOAM=7)만 덧붙일 때, 새 라벨이 '이름 기준'으로 v2 순서에 맞춰지는지 확인.
# 어긋나면 v2 가중치에서 이어 배우면서 종이컵(PAPER)을 엉뚱한 번호로 학습하게 됨.
def test_v2_캐시에_새_폴더만_덧붙이면_기존_클래스_순서를_유지한다(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    model_order = ["CLEAR_PET_BOTTLE", "PLASTIC", "CAN", "GLASS_BOTTLE", "VINYL", "PAPER", "STYROFOAM", "PAPER_PACK"]
    base = tmp_path / "base"
    (base / "labels/train").mkdir(parents=True)
    (base / "images/train").mkdir(parents=True)
    (base / "data.yaml").write_text(yaml.safe_dump({"names": model_order}))
    (base / "labels/train/old.txt").write_text("7 0.5 0.5 0.1 0.1\n")  # 기존 우유팩
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.make_archive(str(cache / "recycle_unified_v2_merged_dataset"), "zip", base)

    raw = tmp_path / "raw"
    _folder(raw, "paper_cup_roboflow", EIGHT, {"cup": ["5 0.5 0.5 0.2 0.2"], "pack": ["6 0.5 0.5 0.2 0.2"]})

    merged = tmp_path / "merged"
    monkeypatch.setattr(train_yolo, "CACHE_DIR", cache)
    monkeypatch.setattr(train_yolo, "BASE_CACHE_ZIP", cache / "recycle_unified_v2_merged_dataset.zip")
    monkeypatch.setattr(train_yolo, "MERGED_DATASET_DIR", merged)
    monkeypatch.setattr(train_yolo, "RAW_DIR", raw)
    monkeypatch.setattr(train_yolo, "DATASET_FOLDERS", [{"id": "x", "name": "can"}, {"id": "y", "name": "paper_cup_roboflow"}])
    monkeypatch.setattr(train_yolo, "BASE_CACHE_FOLDERS", {"can"})
    monkeypatch.setattr(train_yolo, "DRIVE_MOUNT", tmp_path / "no_drive")  # 캐시 저장 단계 건너뜀
    monkeypatch.setattr(train_yolo, "_build_drive_service", lambda: None)
    downloaded = []
    monkeypatch.setattr(train_yolo, "_download_one_item", lambda s, fid, dest: downloaded.append(dest.name))
    monkeypatch.setattr(train_yolo, "Path", lambda p: tmp_path / Path(p).name if p == "/content/merged_dataset.zip" else Path(p))

    data_yaml = train_yolo._prepare_dataset()

    assert downloaded == ["paper_cup_roboflow"]  # v2 캐시에 있는 폴더는 다시 안 받음
    assert yaml.safe_load(data_yaml.read_text())["names"] == model_order  # 클래스 순서 그대로
    labels = merged / "labels/train"
    assert (labels / "old.txt").read_text().split()[0] == "7"  # 기존 라벨 보존
    assert (labels / "paper_cup_roboflow__cup.txt").read_text().split()[0] == "5"  # PAPER
    assert (labels / "paper_cup_roboflow__pack.txt").read_text().split()[0] == "7"  # 폴더의 6(PAPER_PACK) → 모델의 7
