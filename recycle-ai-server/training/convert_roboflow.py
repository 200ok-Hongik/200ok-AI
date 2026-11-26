"""
(2026-11-24 수정) ① 같은 물체에 박스 줄과 폴리곤 줄이 둘 다 있는 데이터(AIHub 출처 스티로폼·종이팩 등)는
  박스가 2개씩 생겨서, 같은 품목·IoU 0.9 이상 박스는 하나로 합침(dedupe_targets). 생활 val 의 종이팩 186장
  라벨이 전부 2중이었던 원인. ② Roboflow 증강 복사본(같은 원본의 회전·밝기 변형, 파일명 "<원본>_jpg.rf.<해시>")이
  train/val 에 나뉘어 들어가면 val 점수가 부풀려지므로 원본 단위로 묶어서 나눔(source_key).
  (라벨 빠진 물체가 찍힌 사진 제외는 변환 뒤 training/find_unlabeled_objects.py 로 따로 함)

── "우리 형식" 이란 (2026-11-24 설명 추가) ──
  YOLO 학습 폴더 구조 + 우리 8품목 클래스 목록으로 통일한 것.
    <폴더>/data.yaml            : 클래스 목록 names = OUR_CLASSES(아래 8개), 학습·검증 사진 위치
    <폴더>/images/train|val/    : 사진 (긴 변 --max-side 이하로 줄여 저장)
    <폴더>/labels/train|val/    : 사진과 같은 이름의 .txt, 물체 하나당 한 줄
       "클래스번호 중심x 중심y 너비 높이"  (전부 사진 크기 대비 0~1 비율, 사각형 박스만)
       예) 7 0.52 0.54 0.50 0.56  → OUR_CLASSES[7]=STYROFOAM, 사진 가운데쯤 절반 크기 박스
  공개 데이터는 데이터마다 다름 - 클래스 이름("glassbottle", "styrofoam - v3 2023-06-27..." 등)과 번호가 제각각이고,
  사각형 대신 윤곽선(폴리곤: "클래스 x1 y1 x2 y2 x3 y3 ...") 으로 적힌 데이터도 있어서 그대로는 합쳐 학습할 수 없다.
  train_yolo.py 는 여러 폴더를 data.yaml 의 이름 기준으로 번호를 다시 매겨 합치므로, 이름만 8품목으로 맞추면 됨.

── 중복 라벨 제거란 ──
  일부 데이터는 같은 물체를 사각형 한 줄 + 윤곽선 한 줄로 **두 번** 적어 둠. 윤곽선을 감싸는 사각형으로 바꾸면
  거의 같은 박스가 2개가 됨 → 모델이 "물체 하나에 박스 2개가 정답" 으로 배우고, 채점할 때는 1개를 잡아도 하나를
  놓친 것으로 계산됨. 같은 품목·IoU(겹친 면적/합친 면적) 0.9 이상이면 같은 물체로 보고 하나만 남긴다.
  예) 스티로폼 데이터: 라벨 약 2,600줄 → 실제 물체 박스 1,313개

Roboflow Universe에서 받은 YOLO 형식 ZIP → 우리 8품목 학습 폴더 변환 스크립트.

2026-11-10 작성. 종이팩·종이처럼 기존 학습 데이터(AIHub 선별장 사진)에 생활 환경 사진이 부족한 품목을
Universe 공개 데이터(CC BY 4.0 등)로 보충하기 위함. 출처 URL·라이선스는 ZIP 안 README.dataset.txt 에 있고,
변환 결과 폴더의 source.txt 로 같이 남긴다 (보고서 출처 표기용).

하는 일:
  1. ZIP 안의 train/valid/test 를 모두 모은 뒤, 원본 클래스 이름을 --map 으로 우리 품목에 연결한다.
     예: --map CARDBOARD=PAPER --map PAPER=PAPER --map BIODEGRADABLE=DROP
       - DROP: 그 클래스 박스가 있는 사진을 통째로 뺌 (우리 품목인지 애매한 물체)
       - BACKGROUND: 박스만 지우고 사진은 씀 (8품목이 아닌 물체)
       - --map 에 없는 클래스는 DROP 으로 취급 (모르는 클래스를 배경으로 학습하지 않게)
  2. 폴리곤(세그멘테이션) 라벨은 감싸는 사각형 박스로 바꾼다 (paper-pack 데이터가 폴리곤 형식).
  3. 품질 필터 (선택):
       --max-boxes N        : 박스가 N개 넘는 사진 제외 (신문 더미·상자 더미처럼 박스 수십 개 겹친 사진)
       --min-area / --max-area : 박스 넓이가 사진의 이 비율 밖이면 그 사진 제외 (사진 전체를 덮는 질감 사진 등)
       --skip-rotated       : 모서리가 검게 채워진 사진(회전 증강이 미리 들어간 사진) 제외
       --sample N           : 필터 통과 사진 중 N장만 무작위로 (공개 데이터가 직접 찍은 사진보다 많아지지 않게)
  4. 사진 단위로 train/val 을 다시 나눠(기본 8:2, 시드 고정) train_yolo.py 가 읽는 구조로 저장:
       <out>/data.yaml, images/{train,val}/, labels/{train,val}/, source.txt, convert_report.json

사용법 (recycle-ai-server 폴더에서):
  python training/convert_roboflow.py --zip paper-pack.v1i.yolov8.zip --out .../paper_pack_roboflow_v1 --map paper-packs=PAPER_PACK
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import re
import zipfile
from pathlib import Path

import yaml
from PIL import Image

OUR_CLASSES = ["CLEAR_PET_BOTTLE", "PLASTIC", "CAN", "GLASS_BOTTLE", "VINYL", "PAPER", "PAPER_PACK", "STYROFOAM"]
DROP, BACKGROUND = "DROP", "BACKGROUND"
IMAGE_RE = re.compile(r"\.(jpe?g|png|bmp|webp)$", re.I)


def label_to_box(values: list[float]) -> tuple[float, float, float, float] | None:
    """YOLO 라벨 한 줄의 숫자들(클래스 제외) → (x1, y1, x2, y2) 0~1. 박스(4개)와 폴리곤(짝수 6개 이상) 둘 다 처리."""
    if len(values) == 4:
        cx, cy, w, h = values
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]  # 폴리곤 꼭짓점 → 감싸는 사각형
        box = (min(xs), min(ys), max(xs), max(ys))
    else:
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in box)  # 사진 밖으로 나간 부분 잘라냄
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """(11-24 추가) 두 박스(x1,y1,x2,y2) 겹침 비율."""
    iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_targets(targets: list[tuple[str, tuple]], iou: float = 0.9) -> list[tuple[str, tuple]]:
    """(11-24 추가) 같은 품목이고 IoU 가 iou 이상인 박스는 처음 것 하나만 남김 (박스 줄 + 폴리곤 줄 중복 제거)."""
    kept: list[tuple[str, tuple]] = []
    for cls, box in targets:
        if not any(cls == k_cls and _box_iou(box, k_box) >= iou for k_cls, k_box in kept):
            kept.append((cls, box))
    return kept


def source_key(image_path: str) -> str:
    """(11-24 추가) Roboflow 파일명 "<원본>_jpg.rf.<해시>.jpg" 에서 원본 이름만 (증강 복사본끼리 같은 값)."""
    return re.split(r"\.rf\.", Path(image_path).name)[0]


def box_to_line(class_index: int, box: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    return f"{class_index} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def looks_pre_rotated(picture: Image.Image, threshold: int = 12) -> bool:
    """모서리 4곳 중 2곳 이상이 거의 완전한 검정이면 회전 증강으로 생긴 빈 공간으로 본다."""
    gray = picture.convert("L")
    w, h = gray.size
    corners = [(1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2)]
    return sum(gray.getpixel(c) <= threshold for c in corners) >= 2


def parse_map(pairs: list[str]) -> dict[str, str]:
    mapping = {}
    for pair in pairs:
        source, _, target = pair.partition("=")
        if target not in OUR_CLASSES + [DROP, BACKGROUND]:
            raise SystemExit(f"--map {pair}: 대상은 8품목 이름 또는 DROP/BACKGROUND 여야 함")
        mapping[source] = target
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboflow YOLO ZIP → 8품목 학습 폴더")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--map", action="append", default=[], help="원본클래스=우리품목|DROP|BACKGROUND (여러 번)")
    parser.add_argument("--max-boxes", type=int)
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--max-area", type=float, default=1.0)
    parser.add_argument("--skip-rotated", action="store_true")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mapping = parse_map(args.map)

    archive = zipfile.ZipFile(args.zip)
    names = archive.namelist()
    yaml_path = next(n for n in names if n.endswith("data.yaml") and "__MACOSX" not in n)
    source_names = yaml.safe_load(archive.read(yaml_path))["names"]
    if isinstance(source_names, dict):  # names 가 {0: a, 1: b} 형식인 데이터도 있음
        source_names = [source_names[k] for k in sorted(source_names)]

    # 1) 사진마다 라벨 읽고 필터
    skipped = collections.Counter()
    candidates = []  # (zip 안 사진 경로, [(우리 품목, box)])
    for image_path in sorted(n for n in names if "/images/" in n and IMAGE_RE.search(n) and "__MACOSX" not in n):
        label_path = IMAGE_RE.sub(".txt", image_path.replace("/images/", "/labels/"))
        lines = archive.read(label_path).decode().splitlines() if label_path in names else []
        targets, reason = [], None
        for line in filter(str.strip, lines):
            parts = line.split()
            our = mapping.get(source_names[int(parts[0])], DROP)
            box = label_to_box(list(map(float, parts[1:])))
            if our == DROP:
                reason = f"drop:{source_names[int(parts[0])]}"
                break
            if our != BACKGROUND and box:
                targets.append((our, box))
        targets = dedupe_targets(targets)  # (11-24 추가) 필터 전에 중복부터 지워야 --max-boxes 가 맞게 셈
        if reason is None and not targets:
            reason = "no_target_box"  # 우리 품목 박스가 없는 사진은 이 데이터에선 쓰지 않음
        if reason is None and args.max_boxes and len(targets) > args.max_boxes:
            reason = "too_many_boxes"
        if reason is None and any(not (args.min_area <= (b[2] - b[0]) * (b[3] - b[1]) <= args.max_area) for _, b in targets):
            reason = "box_area_out_of_range"
        if reason:
            skipped[reason] += 1
            continue
        candidates.append((image_path, targets))

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    # 2) 회전 증강 사진 거르기 (사진을 열어야 알 수 있어서 --sample 개수를 채울 때까지 순서대로 확인)
    chosen = []
    for image_path, targets in candidates:
        if args.sample and len(chosen) >= args.sample:
            break
        picture = Image.open(io.BytesIO(archive.read(image_path))).convert("RGB")
        if args.skip_rotated and looks_pre_rotated(picture):
            skipped["pre_rotated"] += 1
            continue
        chosen.append((image_path, targets, picture))

    # 3) 저장 - (11-24 수정) 원본 단위로 val 을 정함: 원본 이름을 섞어 앞에서부터 val_ratio 만큼을 val 로
    keys = sorted({source_key(p) for p, _, _ in chosen})
    random.Random(args.seed).shuffle(keys)
    val_keys = set(keys[: round(len(keys) * args.val_ratio)])
    n_val = sum(source_key(p) in val_keys for p, _, _ in chosen)
    box_count = collections.Counter()
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    for index, (image_path, targets, picture) in enumerate(chosen):
        split = "val" if source_key(image_path) in val_keys else "train"  # (11-24 수정) 순서 대신 원본 기준
        stem = re.sub(r"[^\w.-]", "_", Path(image_path).stem)[:120]  # Roboflow 파일명의 긴 해시·특수문자 정리
        picture.thumbnail((args.max_side, args.max_side))
        picture.save(args.out / "images" / split / f"{stem}.jpg", quality=92)
        lines = [box_to_line(OUR_CLASSES.index(cls), box) for cls, box in targets]
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        box_count.update(cls for cls, _ in targets)

    data_yaml = {"path": ".", "train": "images/train", "val": "images/val", "nc": len(OUR_CLASSES), "names": OUR_CLASSES}
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False))
    readme = next((n for n in names if n.endswith("README.dataset.txt")), None)
    source = archive.read(readme).decode(errors="ignore") if readme else f"(README 없음) {args.zip.name}"
    (args.out / "source.txt").write_text(source)

    report = {
        "zip": args.zip.name,
        "map": mapping,
        "filters": {k: getattr(args, k) for k in ("max_boxes", "min_area", "max_area", "skip_rotated", "sample")},
        "images": {"train": len(chosen) - n_val, "val": n_val},
        "sourceImages": len(keys),  # (11-24 추가) 증강 복사본을 묶은 원본 수
        "boxes": dict(box_count),
        "candidatesAfterLabelFilter": len(candidates),
        "skipped": dict(skipped),
    }
    (args.out / "convert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
