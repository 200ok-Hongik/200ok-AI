"""
라벨이 빠진 물체가 찍힌 학습 사진 찾기·제외 스크립트.

2026-11-24 작성. v4 학습용 공개 데이터(Roboflow 유리병)를 확인하다가, 유리병 옆에 캔·페트가 같이 찍혀 있는데
라벨은 유리병에만 붙은 사진이 섞여 있는 것을 발견해서 만듦.

왜 문제인가:
  YOLO 는 학습할 때 "라벨 박스가 없는 곳 = 물체 없음(배경)" 이라고 배운다. 그래서 캔이 찍혀 있는데 라벨이 없으면
  "이건 캔이 아니다" 라고 거꾸로 배워 캔 인식이 나빠질 수 있다.
  단, 바닥·잔디·책상·손 같은 진짜 배경은 그대로 두는 게 맞다 (배경을 배경으로 배우는 건 오히려 도움이 됨).
  문제는 **우리 8품목에 해당하는 물체**가 라벨 없이 찍힌 경우뿐이다.

찾는 방법:
  지금 쓰는 모델(예: 배포 중인 v3)로 사진을 돌려서, 확신도 --conf(기본 0.5) 이상으로 뭔가를 잡았는데
  그 박스가 어떤 라벨 박스와도 거의 안 겹치면(박스 면적의 30% 미만) "라벨 빠진 물체 후보" 로 본다.
  --keep-class 로 준 품목(예: 유리병 데이터면 GLASS_BOTTLE)은 제외 대상에서 뺀다 - 같은 품목을 모델이 라벨과
  조금 다른 위치에 잡은 경우가 많아서(라벨 박스가 살짝 어긋난 사진 등) 진짜 누락이 아닐 가능성이 높음.
  모델도 틀리므로(예: v3 는 스티로폼을 종이팩으로 잘못 봄) 결과는 사진을 직접 보고 확인하는 걸 권장.

제외 방법:
  같은 원본의 증강 복사본(Roboflow 파일명 "<원본>_jpg.rf.<해시>")은 같이 찍힌 물체도 같으므로 함께 뺀다.
  지우지 않고 <데이터 폴더 옆>/_제외_<폴더 이름>/ 으로 옮기고, convert_report.json 에 제외 수를 기록한다.

사용법 (recycle-ai-server 폴더에서):
  python training/find_unlabeled_objects.py --data .../glass_roboflow_v1 --keep-class GLASS_BOTTLE          # 찾기만
  python training/find_unlabeled_objects.py --data .../glass_roboflow_v1 --keep-class GLASS_BOTTLE --apply  # 옮기기까지

v4 적용 결과(11-24): 유리병 656장 중 다른 품목이 잡힌 원본 23개(복사본 포함 47장) 제외.
비닐(8장)·스티로폼(31장)은 대부분 v3 의 오인식(스티로폼→종이팩 등)이라 제외하지 않음.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def to_corners(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """YOLO 라벨의 (중심x, 중심y, 너비, 높이) → (x1, y1, x2, y2). 전부 사진 크기 대비 0~1 비율."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def covered_ratio(pred: tuple[float, ...], labels: list[tuple[float, ...]]) -> float:
    """예측 박스 면적 중 라벨 박스와 겹치는 비율의 최댓값. 0 이면 어떤 라벨과도 안 겹침 = 라벨 없는 곳."""
    area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    if area <= 0 or not labels:
        return 0.0
    best = 0.0
    for lab in labels:
        iw = max(0.0, min(pred[2], lab[2]) - max(pred[0], lab[0]))
        ih = max(0.0, min(pred[3], lab[3]) - max(pred[1], lab[1]))
        best = max(best, iw * ih / area)
    return best


def source_key(file_name: str) -> str:
    """증강 복사본을 원본 단위로 묶는 키. convert_roboflow 가 저장한 이름은 "<원본>_jpg.rf.<해시>" 형태 (특수문자는 _ 로 바뀔 수 있어 둘 다 처리)."""
    return re.split(r"_jpg_rf_|\.rf\.", file_name)[0]


def find_flags(data_dir: Path, model_path: Path, conf: float, keep_class: set[str], max_covered: float = 0.3) -> list[dict]:
    """사진마다 라벨 밖에서 잡힌 다른 품목을 찾는다. 반환: [{image, class, confidence, box}]"""
    from ultralytics import YOLO  # 무거운 import 라 실제로 돌릴 때만 불러옴 (테스트는 위 함수들만 씀)

    model = YOLO(str(model_path))
    flags = []
    for image in sorted(data_dir.glob("images/*/*.jpg")):
        label_file = data_dir / "labels" / image.parent.name / f"{image.stem}.txt"
        labels = [to_corners(*map(float, line.split()[1:5])) for line in label_file.read_text().splitlines() if line.strip()]
        result = model.predict(str(image), imgsz=640, conf=conf, verbose=False)[0]
        for cls, box, score in zip(result.boxes.cls, result.boxes.xyxyn.tolist(), result.boxes.conf):
            name = model.names[int(cls)]
            if name in keep_class:
                continue  # 데이터의 원래 품목은 위치가 어긋난 것일 가능성이 커서 제외 대상 아님
            if covered_ratio(tuple(box), labels) < max_covered:
                flags.append({"image": str(image), "class": name, "confidence": round(float(score), 2), "box": box})
                break  # 사진당 하나만 찾으면 충분 (그 사진은 어차피 제외)
    return flags


def move_flagged(data_dir: Path, flags: list[dict]) -> int:
    """표시된 사진의 원본(과 그 증강 복사본 전부)을 _제외_ 폴더로 옮긴다. 옮긴 사진 수를 반환."""
    bad_sources = {source_key(Path(f["image"]).name) for f in flags}
    removed_dir = data_dir.parent / f"_제외_{data_dir.name}"
    moved = 0
    for image in list(data_dir.glob("images/*/*.jpg")):
        if source_key(image.name) not in bad_sources:
            continue
        label = data_dir / "labels" / image.parent.name / f"{image.stem}.txt"
        for path in (image, label):
            if path.exists():
                target = removed_dir / path.relative_to(data_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), target)
        moved += 1
    report_path = data_dir / "convert_report.json"  # convert_roboflow.py 가 만든 보고서에 제외 기록을 덧붙임
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["removedUnlabeledOthers"] = {"images": moved, "sources": len(bad_sources)}
    report["images"] = {s: len(list((data_dir / "images" / s).glob("*.jpg"))) for s in ("train", "val")}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨 빠진 물체가 찍힌 학습 사진 찾기·제외")
    parser.add_argument("--data", required=True, type=Path, help="convert_roboflow.py 결과 폴더")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"), help="찾을 때 쓸 모델 (기본: 배포 중 모델)")
    parser.add_argument("--conf", type=float, default=0.5, help="이 확신도 이상으로 잡힌 것만 봄 (낮추면 오탐이 많아짐)")
    parser.add_argument("--keep-class", action="append", default=[], help="이 데이터의 원래 품목 (제외 대상에서 뺌)")
    parser.add_argument("--apply", action="store_true", help="찾은 사진을 실제로 _제외_ 폴더로 옮김")
    args = parser.parse_args()

    flags = find_flags(args.data, args.model, args.conf, set(args.keep_class))
    by_class: dict[str, int] = {}
    for f in flags:
        by_class[f["class"]] = by_class.get(f["class"], 0) + 1
    print(f"라벨 빠진 물체 후보 사진 {len(flags)}장: {by_class}")
    for f in flags:
        print(f"  {Path(f['image']).name}  {f['class']} {f['confidence']}")
    if args.apply:
        print(f"제외 폴더로 옮김: {move_flagged(args.data, flags)}장 (같은 원본의 증강 복사본 포함)")


if __name__ == "__main__":
    main()
