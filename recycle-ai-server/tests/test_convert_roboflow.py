# 2026-11-10: training/convert_roboflow.py(Roboflow YOLO ZIP → 8품목 폴더) 검증.
# 폴리곤→박스 변환, 클래스 매핑(DROP/BACKGROUND), 품질 필터, 회전 증강 사진 판별을 확인한다.
import sys
import zipfile

import pytest
from PIL import Image

from training import convert_roboflow as cr


def test_박스와_폴리곤_라벨을_모두_사각형으로_바꾼다():
    assert cr.label_to_box([0.5, 0.5, 0.2, 0.4]) == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert cr.label_to_box([0.1, 0.2, 0.3, 0.2, 0.3, 0.6, 0.1, 0.6]) == pytest.approx((0.1, 0.2, 0.3, 0.6))
    assert cr.label_to_box([0.1, 0.2, 0.3]) is None  # 형식 이상


def test_사진_밖으로_나간_박스는_잘라낸다():
    assert cr.label_to_box([0.95, 0.5, 0.2, 0.2]) == pytest.approx((0.85, 0.4, 1.0, 0.6))


def test_모서리가_검은_회전_증강_사진을_알아본다():
    plain = Image.new("RGB", (100, 100), (200, 180, 150))
    rotated = plain.rotate(20, expand=True)  # 회전하면 모서리가 검게 채워짐
    assert cr.looks_pre_rotated(plain) is False
    assert cr.looks_pre_rotated(rotated) is True


def test_map_대상은_8품목이나_DROP_BACKGROUND만_허용한다():
    assert cr.parse_map(["CARDBOARD=PAPER", "BIO=DROP"]) == {"CARDBOARD": "PAPER", "BIO": "DROP"}
    with pytest.raises(SystemExit):
        cr.parse_map(["CARDBOARD=Paper"])  # 대소문자 틀리면 학습에 9번째 품목이 생기므로 막음


def _make_zip(path, items):
    """items: [(이름, [라벨 줄])] → Roboflow 형식 ZIP (train 폴더 하나)."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data.yaml", "names: ['BIO', 'CARDBOARD', 'PAPER']\n")
        z.writestr("README.dataset.txt", "License: CC BY 4.0\n")
        for name, lines in items:
            buf = __import__("io").BytesIO()
            Image.new("RGB", (100, 100), (200, 180, 150)).save(buf, "JPEG")
            z.writestr(f"train/images/{name}.jpg", buf.getvalue())
            z.writestr(f"train/labels/{name}.txt", "\n".join(lines))


def test_필터와_매핑을_거쳐_우리_품목_라벨만_남긴다(tmp_path, monkeypatch):
    zip_path = tmp_path / "d.zip"
    _make_zip(zip_path, [
        ("ok", ["1 0.5 0.5 0.4 0.4", "2 0.2 0.2 0.2 0.2"]),         # 상자+종이 → 둘 다 PAPER(5)
        ("bio", ["0 0.5 0.5 0.3 0.3"]),                              # DROP 클래스 → 제외
        ("full", ["2 0.5 0.5 0.98 0.98"]),                           # 사진 전체 덮음 → 넓이 필터 제외
        ("crowd", ["2 0.1 0.1 0.1 0.1"] * 5),                        # 박스 5개 → 개수 필터 제외
    ])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--zip", str(zip_path), "--out", str(out),
                                      "--map", "CARDBOARD=PAPER", "--map", "PAPER=PAPER", "--map", "BIO=DROP",
                                      "--max-boxes", "3", "--min-area", "0.02", "--max-area", "0.7", "--val-ratio", "0"])
    cr.main()

    labels = list((out / "labels" / "train").glob("*.txt"))
    assert [p.stem for p in labels] == ["ok"]
    assert [line.split()[0] for line in labels[0].read_text().split("\n") if line] == ["5", "5"]
    assert "CC BY 4.0" in (out / "source.txt").read_text()


# (2026-11-24 추가) 박스 줄 + 폴리곤 줄 중복 제거, 증강 복사본 원본 묶기
def test_같은_물체의_박스와_폴리곤_중복은_하나로_합친다():
    box = cr.label_to_box([0.5, 0.5, 0.4, 0.4])
    poly = cr.label_to_box([0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.69])
    other = cr.label_to_box([0.1, 0.1, 0.1, 0.1])
    kept = cr.dedupe_targets([("STYROFOAM", box), ("STYROFOAM", poly), ("STYROFOAM", other)])
    assert len(kept) == 2


def test_품목이_다르면_겹쳐도_합치지_않는다():
    box = cr.label_to_box([0.5, 0.5, 0.4, 0.4])
    assert len(cr.dedupe_targets([("CAN", box), ("PAPER", box)])) == 2


def test_증강_복사본은_같은_원본으로_묶는다():
    a = "train/images/IMG_12_jpg.rf.aaa111.jpg"
    b = "valid/images/IMG_12_jpg.rf.bbb222.jpg"
    assert cr.source_key(a) == cr.source_key(b) != cr.source_key("train/images/IMG_13_jpg.rf.ccc.jpg")
