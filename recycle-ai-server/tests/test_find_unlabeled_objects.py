# (2026-11-24 추가) training/find_unlabeled_objects.py 검증 - 모델 없이 판단 규칙만 확인
# - 라벨 박스와 안 겹치는 예측은 "라벨 없는 곳" 으로 보는지 (covered_ratio)
# - 증강 복사본을 같은 원본으로 묶는지 (source_key)
# - 표시된 원본의 복사본까지 _제외_ 폴더로 옮기고 보고서에 남기는지 (move_flagged)
import json

from training import find_unlabeled_objects as fu


def test_라벨과_안_겹치는_예측은_0():
    labels = [fu.to_corners(0.2, 0.2, 0.2, 0.2)]
    assert fu.covered_ratio((0.7, 0.7, 0.9, 0.9), labels) == 0.0


def test_라벨_안에_들어간_예측은_1():
    labels = [fu.to_corners(0.5, 0.5, 0.6, 0.6)]
    assert fu.covered_ratio((0.4, 0.4, 0.6, 0.6), labels) == 1.0


def test_라벨이_없으면_0():
    assert fu.covered_ratio((0.1, 0.1, 0.2, 0.2), []) == 0.0


def test_증강_복사본은_같은_원본():
    assert fu.source_key("IMG_1_jpg.rf.aaa.jpg") == fu.source_key("IMG_1_jpg.rf.bbb.jpg") == "IMG_1_jpg"


def test_표시된_원본의_복사본까지_옮긴다(tmp_path):
    data = tmp_path / "glass"
    for split, name in (("train", "A_jpg.rf.1"), ("val", "A_jpg.rf.2"), ("train", "B_jpg.rf.1")):
        (data / "images" / split).mkdir(parents=True, exist_ok=True)
        (data / "labels" / split).mkdir(parents=True, exist_ok=True)
        (data / "images" / split / f"{name}.jpg").write_bytes(b"x")
        (data / "labels" / split / f"{name}.txt").write_text("3 0.5 0.5 0.2 0.2\n")
    moved = fu.move_flagged(data, [{"image": str(data / "images/train/A_jpg.rf.1.jpg")}])
    assert moved == 2  # A 원본의 train·val 복사본 둘 다
    assert [p.name for p in data.glob("images/*/*.jpg")] == ["B_jpg.rf.1.jpg"]
    assert (tmp_path / "_제외_glass" / "labels" / "val" / "A_jpg.rf.2.txt").exists()
    assert json.loads((data / "convert_report.json").read_text())["removedUnlabeledOthers"]["images"] == 2
