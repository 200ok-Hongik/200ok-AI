# 2026-11-10: training/convert_taco.py(TACO → 8품목 YOLO 변환)의 핵심 계산 검증.
# 박스 좌표가 틀리면 학습은 돌아가도 엉뚱한 위치를 배우므로, 좌표 변환·사진 방향·클래스 매핑을 확인한다.
import pytest
from PIL import Image

from training.convert_taco import DROP, BACKGROUND, OUR_CLASSES, load_upright, resolve_class, to_yolo_line


def test_COCO박스를_YOLO_비율좌표로_바꾼다():
    # 1000x500 사진의 (100,50)에서 가로 200, 세로 100 박스 → 중심 (200,100)
    assert to_yolo_line(2, [100, 50, 200, 100], 1000, 500) == "2 0.200000 0.200000 0.200000 0.200000"


def test_사진_밖으로_나간_박스는_잘라낸다():
    line = to_yolo_line(0, [900, 400, 300, 300], 1000, 500)  # 오른쪽·아래로 넘침 → (900,400)~(1000,500)
    assert line == "0 0.950000 0.900000 0.100000 0.200000"


def test_넓이가_없는_박스는_버린다():
    assert to_yolo_line(0, [1200, 10, 50, 50], 1000, 500) is None


@pytest.mark.parametrize(
    "taco_name, expected",
    [
        ("Drink can", "CAN"),
        ("Clear plastic bottle", "CLEAR_PET_BOTTLE"),
        ("Drink carton", "PAPER_PACK"),
        ("Corrugated carton", "PAPER"),
        ("Plastic film", "VINYL"),
        ("Cigarette", BACKGROUND),      # 8품목 아님 → 배경
        ("Broken glass", BACKGROUND),   # 가이드: 깨진 유리는 GLASS_BOTTLE 아님
        ("Unlabeled litter", DROP),     # 애매 → 사진 제외
    ],
)
def test_TACO클래스를_우리_품목_배경_제외로_나눈다(taco_name, expected):
    assert resolve_class(taco_name, "PAPER") == expected


@pytest.mark.parametrize("choice", ["PAPER", "PAPER_PACK", DROP])
def test_종이컵은_옵션으로_정한다(choice):
    assert resolve_class("Paper cup", choice) == choice


def test_매핑된_품목은_모두_서버_8품목_이름이다():
    from training.convert_taco import TACO_TO_OURS
    from schemas.analysis_schema import CATEGORY_STATE_FIELDS

    assert set(OUR_CLASSES) == set(CATEGORY_STATE_FIELDS)
    assert {v for v in TACO_TO_OURS.values() if v not in (DROP, BACKGROUND)} <= set(OUR_CLASSES)


def test_EXIF_회전된_사진은_어노테이션_크기에_맞춰_세운다(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # 90도 회전 표시 - 저장은 가로(400x200), 보이는 모양은 세로(200x400)
    Image.new("RGB", (400, 200)).save(path, exif=exif)

    assert load_upright(path, 200, 400).size == (200, 400)  # 어노테이션이 회전 후 기준
    assert load_upright(path, 400, 200).size == (400, 200)  # 어노테이션이 회전 전 기준


def test_사진은_박스가_가장_많은_품목_폴더로_간다():
    from training.convert_taco import primary_item

    boxes = [("CAN", [0, 0, 10, 10]), ("VINYL", [0, 0, 50, 50]), ("CAN", [0, 0, 10, 10])]
    assert primary_item(boxes) == "can"  # 개수 우선 (CAN 2개 > VINYL 1개)
    assert primary_item([("CAN", [0, 0, 10, 10]), ("VINYL", [0, 0, 50, 50])]) == "vinyl"  # 같으면 넓이


def test_품목별_폴더로_나누면_다른_품목_박스도_라벨에_남는다(tmp_path, monkeypatch):
    import json
    import sys
    from training import convert_taco

    taco = tmp_path / "taco"
    (taco / "batch_1").mkdir(parents=True)
    for name in ("a", "b", "c"):
        Image.new("RGB", (100, 100)).save(taco / "batch_1" / f"{name}.jpg")
    categories = [{"id": 1, "name": "Drink can"}, {"id": 2, "name": "Plastic film"},
                  {"id": 3, "name": "Cigarette"}, {"id": 4, "name": "Unlabeled litter"}]
    images = [{"id": i, "file_name": f"batch_1/{n}.jpg", "width": 100, "height": 100} for i, n in enumerate("abc", 1)]
    annotations = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20]},  # a: 캔 2 + 비닐 1 → can 폴더
        {"image_id": 1, "category_id": 1, "bbox": [30, 0, 20, 20]},
        {"image_id": 1, "category_id": 2, "bbox": [0, 50, 40, 40]},
        {"image_id": 2, "category_id": 3, "bbox": [0, 0, 5, 5]},    # b: 담배꽁초만 → 배경 사진
        {"image_id": 3, "category_id": 4, "bbox": [0, 0, 5, 5]},    # c: 애매한 물체 → 제외
    ]
    (taco / "annotations.json").write_text(json.dumps({"images": images, "categories": categories, "annotations": annotations}))

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--taco-dir", str(taco), "--out", str(out), "--split-by-item",
                                      "--val-ratio", "0", "--background-ratio", "0.5"])
    convert_taco.main()

    label = (out / "can_taco_v1" / "labels" / "train" / "batch_1_a.txt").read_text().split("\n")
    assert sorted(line.split()[0] for line in label if line) == ["2", "2", "4"]  # CAN 2개 + VINYL 1개 유지
    assert (out / "background_taco_v1" / "labels" / "train" / "batch_1_b.txt").read_text() == ""
    assert not list(out.rglob("batch_1_c.*"))  # 애매한 사진은 어디에도 없음
    assert "names" in (out / "can_taco_v1" / "data.yaml").read_text()


def test_크기가_안_맞는_사진은_에러로_알린다(tmp_path):
    path = tmp_path / "x.jpg"
    Image.new("RGB", (300, 300)).save(path)
    with pytest.raises(ValueError):
        load_upright(path, 640, 480)
