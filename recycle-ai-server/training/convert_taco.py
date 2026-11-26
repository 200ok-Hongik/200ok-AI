"""
TACO(Trash Annotations in Context) 데이터셋 → 우리 8품목 YOLO 학습 폴더 변환 스크립트.

2026-11-10 작성. 지금 학습 데이터가 AIHub 선별장(컨베이어) 사진뿐이라 책상·바닥 같은 생활 환경 사진에서
물체를 놓치거나 배경을 오탐함 → 생활 환경에서 찍힌 공개 데이터 TACO를 보조로 추가하기 위함.

출처·라이선스 (보고서에 표기):
  - TACO: Proença & Simões, "TACO: Trash Annotations in Context for Litter Detection", arXiv:2003.06975
  - 데이터: Zenodo DOI 10.5281/zenodo.3587843, CC BY 4.0 / 코드 레포 github.com/pedropro/TACO (MIT)

하는 일:
  1. data/annotations.json(COCO 형식)의 TACO 클래스 60개를 우리 8품목 / 배경 / 제외 로 나눈다 (TACO_TO_OURS).
     - 우리 품목: 박스를 우리 클래스 이름으로 바꿔 라벨로 씀
     - 배경(BACKGROUND): 담배꽁초·병뚜껑처럼 8품목이 아닌 물체 → 박스를 안 씀(모델이 "물체 아님"으로 배움)
     - 제외(DROP): "Unlabeled litter"·"Other plastic"처럼 우리 품목일 수도 있는 애매한 박스 → 그 사진을 통째로 뺌
       (넣으면 같은 물체가 어떤 사진에선 품목, 어떤 사진에선 배경으로 학습돼 꼬임)
  2. 사진을 EXIF 회전을 반영해 똑바로 세우고(박스 좌표가 annotations.json의 width/height 기준이라 맞춰야 함),
     긴 변 --max-side(기본 1280)로 줄여 저장 (학습은 640으로 하므로 화질 손실 없음, Drive 업로드·Colab 다운로드 시간 절약).
  3. train/val 을 사진 단위로 나누고(기본 8:2, 시드 고정) train_yolo.py가 읽는 구조로 저장:
       <out>/data.yaml, images/{train,val}/, labels/{train,val}/
  4. 우리 품목 박스가 하나도 없는 사진(배경만 있는 사진)은 전체의 --background-ratio(기본 10%)까지만 넣음
     (Ultralytics 학습 팁: 배경 사진 0~10%가 오탐을 줄임).

사용법 (recycle-ai-server 폴더에서):
  python training/convert_taco.py --taco-dir ~/datasets/TACO/data --out ~/datasets/taco_v1
  # 종이컵 분류를 바꾸려면 --paper-cup PAPER_PACK 또는 --paper-cup DROP
  # Drive recycle_training_data/ 기존 구조(can_v1 …)처럼 품목별 폴더로: --split-by-item → can_taco_v1/ … background_taco_v1/
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import yaml
from PIL import Image, ImageOps

# 서버(yolo_service.CLASS_TO_ITEM)가 소문자로 바꿔 조회하므로 대소문자는 상관없지만, 가이드 0단계 표와 똑같이 씀.
# 순서는 상관없음 - train_yolo.py가 이름 기준으로 전역 번호를 다시 매김.
OUR_CLASSES = ["CLEAR_PET_BOTTLE", "PLASTIC", "CAN", "GLASS_BOTTLE", "VINYL", "PAPER", "PAPER_PACK", "STYROFOAM"]
BACKGROUND = "BACKGROUND"
DROP = "DROP"

# TACO 클래스 이름(category name) → 우리 품목 / BACKGROUND / DROP.
# 기준: 2026-09-15 "규칙 수집" 8품목 정의 + 가이드 0단계 표(포함·제외 물체).
TACO_TO_OURS: dict[str, str] = {
    # CLEAR_PET_BOTTLE: 무색 투명 페트병
    "Clear plastic bottle": "CLEAR_PET_BOTTLE",
    # PLASTIC: 유색 페트병, 플라스틱 컵·용기
    "Other plastic bottle": "PLASTIC",
    "Disposable plastic cup": "PLASTIC",
    "Other plastic cup": "PLASTIC",
    "Disposable food container": "PLASTIC",
    "Other plastic container": "PLASTIC",
    "Spread tub": "PLASTIC",
    "Tupperware": "PLASTIC",
    # CAN: 음료캔·통조림캔
    "Drink can": "CAN",
    "Food Can": "CAN",
    # GLASS_BOTTLE: 유리병 (잼 병 같은 유리 용기 포함, 깨진 유리·유리컵 제외)
    "Glass bottle": "GLASS_BOTTLE",
    "Glass jar": "GLASS_BOTTLE",
    # VINYL: 비닐 봉투·과자 봉지·포장 필름
    "Plastic film": "VINYL",
    "Other plastic wrapper": "VINYL",
    "Single-use carrier bag": "VINYL",
    "Crisp packet": "VINYL",
    "Garbage bag": "VINYL",
    "Polypropylene bag": "VINYL",
    # PAPER: 종이·상자·종이봉투 (종이컵은 --paper-cup 옵션으로 결정, 기본 PAPER)
    "Normal paper": "PAPER",
    "Magazine paper": "PAPER",
    "Wrapping paper": "PAPER",
    "Paper bag": "PAPER",
    "Corrugated carton": "PAPER",
    "Meal carton": "PAPER",
    "Egg carton": "PAPER",
    "Pizza box": "PAPER",
    "Toilet tube": "PAPER",
    "Paper cup": "PAPER",
    # PAPER_PACK: 우유팩·주스팩 같은 음료 종이팩
    "Drink carton": "PAPER_PACK",
    # STYROFOAM
    "Styrofoam piece": "STYROFOAM",
    "Foam food container": "STYROFOAM",
    "Foam cup": "STYROFOAM",
    # 애매해서 사진째 제외: 우리 품목일 수도 있는 물체
    "Unlabeled litter": DROP,      # 무엇인지 표시 안 된 쓰레기 (캔·병일 수도 있음)
    "Other plastic": DROP,         # 플라스틱 조각·용기 구분 불가
    "Other carton": DROP,          # 종이상자인지 종이팩인지 구분 불가
    "Aerosol": DROP,               # 스프레이캔 - 국내 기준 캔류지만 모양이 달라 이번엔 제외
    # 나머지(담배꽁초, 병뚜껑, 빨대, 깨진 유리, 휴지, 은박지, 일회용 수저 등)는 아래 기본값으로 BACKGROUND
}


def resolve_class(taco_name: str, paper_cup: str) -> str:
    if taco_name == "Paper cup":
        return paper_cup  # 팀 결정에 따라 PAPER / PAPER_PACK / DROP
    return TACO_TO_OURS.get(taco_name, BACKGROUND)


def load_upright(image_path: Path, expected_w: int, expected_h: int) -> Image.Image:
    """박스 좌표가 기준으로 삼은 방향(annotations.json의 width/height)과 같은 방향으로 사진을 연다."""
    raw = Image.open(image_path)
    upright = ImageOps.exif_transpose(raw)  # 휴대폰 세로 사진 EXIF 회전 반영
    if upright.size == (expected_w, expected_h):
        return upright.convert("RGB")
    if raw.size == (expected_w, expected_h):
        return raw.convert("RGB")  # 어노테이션이 회전 전 기준인 사진
    raise ValueError(f"size mismatch json={expected_w}x{expected_h} raw={raw.size} upright={upright.size}")


def to_yolo_line(class_index: int, bbox: list[float], width: int, height: int) -> str | None:
    """COCO [x, y, w, h](픽셀) → YOLO 'class cx cy w h'(0~1). 넓이가 없으면 None."""
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)  # 사진 밖으로 나간 박스는 잘라냄
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"{class_index} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def primary_item(boxes: list[tuple[str, list[float]]]) -> str:
    """사진을 넣을 품목 폴더: 박스가 가장 많은 품목, 같으면 박스 넓이 합이 큰 품목 (폴더명은 기존처럼 소문자)."""
    score: dict[str, tuple[int, float]] = {}
    for cls, (_, _, w, h) in boxes:
        count, area = score.get(cls, (0, 0.0))
        score[cls] = (count + 1, area + w * h)
    return max(score, key=lambda cls: score[cls]).lower()


def write_data_yaml(dest: Path) -> None:
    # 품목 폴더여도 names는 8품목 전부 - 라벨에 다른 품목 박스가 같이 들어 있기 때문
    data_yaml = {"path": ".", "train": "images/train", "val": "images/val", "nc": len(OUR_CLASSES), "names": OUR_CLASSES}
    (dest / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="TACO → 8품목 YOLO 폴더 변환")
    parser.add_argument("--taco-dir", required=True, type=Path, help="annotations.json 과 batch_* 폴더가 있는 곳")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--paper-cup", default="PAPER", choices=["PAPER", "PAPER_PACK", DROP])
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--background-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-item", action="store_true",
                        help="Drive 기존 폴더처럼 <품목>_taco_v1/ 폴더로 나눠 저장 (배경 사진은 background_taco_v1/)")
    args = parser.parse_args()

    coco = json.loads((args.taco_dir / "annotations.json").read_text(encoding="utf-8"))
    category_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # 1) 사진마다 사용/제외 판단
    labeled, background_only, dropped = [], [], collections.Counter()
    for image in coco["images"]:
        targets = [(resolve_class(category_name[a["category_id"]], args.paper_cup), a) for a in anns_by_image[image["id"]]]
        drop_reasons = {category_name[a["category_id"]] for cls, a in targets if cls == DROP}
        if drop_reasons:
            dropped.update(drop_reasons)
            continue
        boxes = [(cls, a["bbox"]) for cls, a in targets if cls in OUR_CLASSES]
        (labeled if boxes else background_only).append((image, boxes))

    rng = random.Random(args.seed)
    rng.shuffle(labeled)
    rng.shuffle(background_only)
    # 배경 사진은 전체(라벨 사진 + 배경 사진)의 background_ratio 까지만
    max_background = int(len(labeled) * args.background_ratio / (1 - args.background_ratio))
    background_only = background_only[:max_background]

    # 2) train/val 나누기 - 라벨 사진과 배경 사진을 각각 같은 비율로
    def split(items: list) -> tuple[list, list]:
        n_val = round(len(items) * args.val_ratio)
        return items[n_val:], items[:n_val]

    # 2026-11-10 --split-by-item: Drive recycle_training_data/ 의 기존 품목 폴더(can_v1 …)처럼 품목별 폴더로 나눔.
    # 사진은 "가장 많이 찍힌 품목" 폴더에 넣고 라벨은 사진 속 8품목 박스를 전부 유지한다 - 다른 품목 박스를 지우면
    # 그 물체가 배경으로 학습됨. train_yolo.py는 폴더별 data.yaml 이름 기준으로 합치므로 같은 품목끼리 다시 모임.
    groups: dict[str, list] = collections.defaultdict(list)
    for item in labeled:
        groups[primary_item(item[1]) if args.split_by_item else ""].append(item)
    groups["background" if args.split_by_item else ""].extend(background_only)

    plan = []  # (출력 폴더, split, (image, boxes))
    for group_name, items in groups.items():
        dest = args.out / f"{group_name}_taco_v1" if group_name else args.out
        train, val = split(items)  # 품목 폴더마다 같은 비율로 나눠야 폴더별 val이 비지 않음
        plan += [(dest, "train", it) for it in train] + [(dest, "val", it) for it in val]

    # 3) 저장
    for dest in {p[0] for p in plan}:
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            (dest / sub).mkdir(parents=True, exist_ok=True)
        write_data_yaml(dest)

    box_count = {s: collections.Counter() for s in ("train", "val")}
    image_count = collections.Counter()
    folder_count = collections.Counter()
    errors = []
    for dest, split_name, (image, boxes) in plan:
        src = args.taco_dir / image["file_name"]
        stem = image["file_name"].replace("/", "_").rsplit(".", 1)[0]  # batch_1/000006.jpg → batch_1_000006
        try:
            picture = load_upright(src, image["width"], image["height"])
        except Exception as exc:  # 파일 없음·깨짐·크기 불일치는 건너뛰고 보고
            errors.append(f"{image['file_name']}: {exc}")
            continue
        lines = [to_yolo_line(OUR_CLASSES.index(cls), bbox, image["width"], image["height"]) for cls, bbox in boxes]
        lines = [line for line in lines if line]
        if boxes and not lines:
            errors.append(f"{image['file_name']}: all boxes empty after clipping")
            continue

        picture.thumbnail((args.max_side, args.max_side))  # 비율 유지 축소 - YOLO 라벨은 0~1 비율이라 그대로 유효
        picture.save(dest / "images" / split_name / f"{stem}.jpg", quality=92)
        # 배경 사진도 빈 라벨 파일을 만들어 둠 (Ultralytics는 라벨 없는 사진을 배경으로 학습)
        (dest / "labels" / split_name / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        image_count[(split_name, "labeled" if lines else "background")] += 1
        folder_count[f"{dest.name}/{split_name}"] += 1
        for line in lines:
            box_count[split_name][OUR_CLASSES[int(line.split()[0])]] += 1

    report = {
        "source": "TACO (Zenodo 10.5281/zenodo.3587843, CC BY 4.0)",
        "paperCup": args.paper_cup,
        "folders": dict(sorted(folder_count.items())),
        "images": {f"{k[0]}/{k[1]}": v for k, v in sorted(image_count.items())},
        "boxes": {s: dict(box_count[s]) for s in box_count},
        "droppedImagesByReason": dict(dropped),
        "backgroundImagesAvailable": len(background_only),
        "errors": errors,
    }
    (args.out / "convert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, ensure_ascii=False, indent=2))
    print(f"errors: {len(errors)}", *errors[:10], sep="\n  ")


if __name__ == "__main__":
    main()
