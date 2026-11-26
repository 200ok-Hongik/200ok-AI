# (2026-11-21 추가) bench/benchmark.py --set 옵션 테스트
# - 세트 이름을 주면 사진·정답 경로가 bench/sets/<이름>/ 으로 바뀌는지
# - 없는 세트는 바로 종료하는지
# - --compare 때 서로 다른 세트 결과를 비교하지 못하게 막는지
import importlib.util
import json
from pathlib import Path

import pytest

BENCH_PATH = Path(__file__).resolve().parent.parent / "bench" / "benchmark.py"


def _load_bench(tmp_path):
    """매 테스트마다 benchmark 모듈을 새로 읽고 폴더 경로를 임시 폴더로 바꿈 (전역 경로가 테스트끼리 섞이지 않게)."""
    spec = importlib.util.spec_from_file_location("bench_under_test", BENCH_PATH)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.SETS_DIR = tmp_path / "sets"
    bench.RESULTS_DIR = tmp_path / "results"
    return bench


def test_use_set_switches_paths(tmp_path):
    bench = _load_bench(tmp_path)
    set_dir = tmp_path / "sets" / "life_hard"
    (set_dir / "images").mkdir(parents=True)
    (set_dir / "labels.json").write_text(json.dumps({"a.jpg": ["CAN"]}), encoding="utf-8")

    bench._use_set("life_hard")

    assert bench.IMAGES_DIR == set_dir / "images"
    assert bench.CURRENT_SET == "life_hard"
    assert bench._load_labels() == {"a.jpg": ["CAN"]}  # 세트의 정답 파일을 읽어야 함


def test_use_set_none_keeps_default(tmp_path):
    bench = _load_bench(tmp_path)
    default_images = bench.IMAGES_DIR
    bench._use_set(None)
    assert bench.IMAGES_DIR == default_images and bench.CURRENT_SET is None


def test_use_set_missing_exits(tmp_path):
    bench = _load_bench(tmp_path)
    with pytest.raises(SystemExit):
        bench._use_set("nope")


def test_compare_rejects_different_sets(tmp_path):
    bench = _load_bench(tmp_path)
    bench.RESULTS_DIR.mkdir()
    for label, set_name in (("a", "life_hard"), ("b", None)):
        (bench.RESULTS_DIR / f"{label}.json").write_text(json.dumps({"label": label, "set": set_name, "runs": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="사진 세트가 다릅니다"):
        bench.compare("a", "b")
