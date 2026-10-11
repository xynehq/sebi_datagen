"""Region detection groups ink into blocks and does not read text."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.detection.regions import (
    DetectionConflictError,
    detect_pages,
    detect_regions_on_array,
)
from indic_ocr_pipeline.ids import make_page_id, make_region_id
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.schemas import PageRecord, RegionRecord, RegionType, StageStatus
from indic_ocr_pipeline.utils.hashing import sha256_file


def _settings(repo_root: Path):
    return load_settings(repo_root, environ={}).detection


def _canvas(width: int = 400, height: int = 480) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _fill(image: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    image[y : y + height, x : x + width] = 0


def _lines(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    count: int,
    *,
    thickness: int = 8,
    gap: int = 8,
) -> None:
    for index in range(count):
        _fill(image, x, y + index * (thickness + gap), width, thickness)


def _save(path: Path, image: np.ndarray, mode: str = "L") -> None:
    Image.fromarray(image, mode=mode).save(path)


def _page_row(path: Path, *, document_id: str = "MHGR_000001", page_number: int = 1) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    page_id = make_page_id(document_id, page_number)
    record = PageRecord(
        document_id=document_id,
        page_id=page_id,
        page_number=page_number,
        image_path=str(path),
        width=width,
        height=height,
        dpi=200,
        sha256=sha256_file(path),
        status=StageStatus.COMPLETE,
    )
    return record.to_dict()


def test_blank_page_has_no_regions(repo_root: Path) -> None:
    boxes = detect_regions_on_array(_canvas(), _settings(repo_root))
    assert boxes == []


def test_multiline_block_is_one_region(repo_root: Path) -> None:
    image = _canvas()
    _lines(image, 40, 50, 180, 4)
    boxes = detect_regions_on_array(image, _settings(repo_root))
    assert len(boxes) == 1
    box = boxes[0]
    assert box.region_type is RegionType.TEXT_BLOCK
    assert box.x0 <= 40 and box.x1 >= 220
    assert box.y0 <= 50 and box.y1 >= 50 + 3 * 16 + 8
    assert box.y1 - box.y0 < 200
    assert box.x1 <= image.shape[1] and box.y1 <= image.shape[0]


def test_separated_blocks_stay_separate(repo_root: Path) -> None:
    image = _canvas()
    _lines(image, 40, 40, 180, 3)
    _lines(image, 40, 220, 180, 3)
    boxes = detect_regions_on_array(image, _settings(repo_root))
    assert len(boxes) == 2
    assert boxes[0].y1 < boxes[1].y0
    assert boxes[0].y0 < 80
    assert boxes[1].y0 > 180


def test_columns_are_not_one_page_region(repo_root: Path) -> None:
    image = _canvas()
    _lines(image, 20, 60, 80, 4)
    _lines(image, 240, 60, 80, 4)
    boxes = detect_regions_on_array(image, _settings(repo_root))
    width = image.shape[1]
    assert len(boxes) == 2
    assert all(box.x1 - box.x0 < int(width * 0.7) for box in boxes)
    left, right = sorted(boxes, key=lambda box: box.x0)
    assert left.x1 < right.x0


def test_small_text_block_is_kept(repo_root: Path) -> None:
    image = _canvas()
    _fill(image, 30, 40, 22, 14)
    _fill(image, 250, 250, 2, 2)
    boxes = detect_regions_on_array(image, _settings(repo_root))
    assert len(boxes) == 1
    assert boxes[0].x1 - boxes[0].x0 < 40
    assert boxes[0].y1 - boxes[0].y0 < 40


def test_ruled_grid_is_one_table_region(repo_root: Path) -> None:
    image = _canvas(height=520)
    for y in (80, 180, 280):
        _fill(image, 50, y, 300, 3)
    for x in (50, 190, 347):
        _fill(image, x, 80, 3, 203)
    boxes = detect_regions_on_array(image, _settings(repo_root))
    assert len(boxes) == 1
    assert boxes[0].region_type is RegionType.TABLE
    assert boxes[0].x0 <= 50 and boxes[0].x1 >= 350
    assert boxes[0].y0 <= 80 and boxes[0].y1 >= 283


def test_region_ids_are_deterministic_and_ordered(repo_root: Path, tmp_path: Path) -> None:
    image = _canvas()
    _lines(image, 40, 40, 160, 2)
    _lines(image, 40, 240, 160, 2)
    settings = _settings(repo_root)
    assert detect_regions_on_array(image, settings) == detect_regions_on_array(image, settings)
    path = tmp_path / "page.png"
    _save(path, image)
    manifest = JsonlManifest(tmp_path / "regions.jsonl", "region_id")
    report = detect_pages([_page_row(path)], manifest, repo_root=repo_root, settings=settings)
    rows = [json.loads(line) for line in manifest.path.read_text().splitlines()]
    assert report.errors == 0
    assert [row["region_id"] for row in rows] == [
        make_region_id("MHGR_000001_P0001", 1),
        make_region_id("MHGR_000001_P0001", 2),
    ]
    assert rows[0]["bbox"][1] < rows[1]["bbox"][1]
    assert all(row["line_id"] is None and row["confidence"] is None for row in rows)
    assert all("text" not in row and "gold_text" not in row for row in rows)


def test_rerun_skips_and_conflict_keeps_the_manifest(repo_root: Path, tmp_path: Path) -> None:
    image = _canvas()
    _lines(image, 40, 40, 160, 3)
    path = tmp_path / "page.png"
    _save(path, image)
    before = path.read_bytes()
    manifest = JsonlManifest(tmp_path / "regions.jsonl", "region_id")
    settings = _settings(repo_root)
    page = _page_row(path)
    first = detect_pages([page], manifest, repo_root=repo_root, settings=settings)
    stored = manifest.path.read_bytes()
    second = detect_pages([page], manifest, repo_root=repo_root, settings=settings)
    assert first.regions == 1
    assert first.regions_skipped == 0
    assert second.regions_skipped == 1
    assert second.regions == 1
    assert manifest.path.read_bytes() == stored
    assert path.read_bytes() == before

    rows = [json.loads(line) for line in stored.decode().splitlines()]
    rows[0]["bbox"] = [1, 2, 40, 50]
    corrupted = (json.dumps(rows[0], sort_keys=True) + "\n").encode()
    manifest.path.write_bytes(corrupted)
    conflict = detect_pages([page], manifest, repo_root=repo_root, settings=settings)
    assert conflict.errors == 1
    assert "different payload" in conflict.details[0].message
    assert manifest.path.read_bytes() == corrupted
    assert path.read_bytes() == before


def test_missing_corrupt_and_stale_pages(repo_root: Path, tmp_path: Path) -> None:
    settings = _settings(repo_root)
    missing = JsonlManifest(tmp_path / "missing.jsonl", "region_id")
    missing_page = {
        "document_id": "MHGR_000001",
        "page_id": "MHGR_000001_P0001",
        "page_number": 1,
        "image_path": str(tmp_path / "absent.png"),
        "width": 20,
        "height": 20,
        "sha256": "a" * 64,
    }
    missing_report = detect_pages(
        [missing_page], missing, repo_root=repo_root, settings=settings
    )
    assert missing_report.errors == 1
    assert "missing" in missing_report.details[0].message
    assert not missing.path.exists()

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not-a-png")
    corrupt_page = {
        "document_id": "MHGR_000001",
        "page_id": "MHGR_000001_P0001",
        "page_number": 1,
        "image_path": str(corrupt_path),
        "width": 20,
        "height": 20,
        "sha256": sha256_file(corrupt_path),
    }
    corrupt = JsonlManifest(tmp_path / "corrupt.jsonl", "region_id")
    corrupt_report = detect_pages(
        [corrupt_page], corrupt, repo_root=repo_root, settings=settings
    )
    assert corrupt_report.errors == 1
    assert "not a readable" in corrupt_report.details[0].message
    assert not corrupt.path.exists()

    image = _canvas(width=80, height=80)
    _fill(image, 10, 10, 30, 16)
    stale_path = tmp_path / "stale.png"
    _save(stale_path, image)
    stale_page = _page_row(stale_path)
    stale_page["sha256"] = "b" * 64
    stale = JsonlManifest(tmp_path / "stale.jsonl", "region_id")
    stale_report = detect_pages([stale_page], stale, repo_root=repo_root, settings=settings)
    assert stale_report.errors == 1
    assert "SHA-256" in stale_report.details[0].message
    assert not stale.path.exists()


def test_failed_append_does_not_record_the_failed_region(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _canvas()
    _lines(image, 30, 30, 140, 2)
    _lines(image, 30, 240, 140, 2)
    path = tmp_path / "page.png"
    _save(path, image)
    manifest = JsonlManifest(tmp_path / "regions.jsonl", "region_id")
    original = manifest.append

    def fail_second(record: dict) -> str:
        if record["region_index"] == 2:
            raise DetectionConflictError("append failed")
        return original(record)

    monkeypatch.setattr(manifest, "append", fail_second)
    report = detect_pages(
        [_page_row(path)], manifest, repo_root=repo_root, settings=_settings(repo_root)
    )
    assert report.errors == 1
    rows = [json.loads(line) for line in manifest.path.read_text().splitlines()]
    assert [row["region_index"] for row in rows] == [1]


def test_manifest_is_jsonl_and_blank_page_does_not_crash(repo_root: Path, tmp_path: Path) -> None:
    image = _canvas()
    _lines(image, 40, 40, 150, 2)
    path = tmp_path / "page.png"
    _save(path, image)
    blank = tmp_path / "blank.png"
    _save(blank, _canvas())
    manifest = JsonlManifest(tmp_path / "regions.jsonl", "region_id")
    pages = [
        _page_row(path, page_number=1),
        _page_row(blank, page_number=2),
    ]
    report = detect_pages(pages, manifest, repo_root=repo_root, settings=_settings(repo_root))
    assert report.errors == 0
    assert report.zero_pages == 1
    rows = []
    for line in manifest.path.read_text().splitlines():
        payload = json.loads(line)
        record = RegionRecord.from_dict(payload)
        assert record.region_type is not RegionType.LINE
        rows.append(record)
    assert len(rows) == 1
    again = detect_pages(pages, manifest, repo_root=repo_root, settings=_settings(repo_root))
    assert again.regions_skipped == 1
    assert again.errors == 0


def test_debug_copy_does_not_change_the_page(repo_root: Path, tmp_path: Path) -> None:
    image = _canvas()
    _lines(image, 40, 40, 120, 2)
    path = tmp_path / "page.png"
    _save(path, image)
    before = sha256_file(path)
    manifest = JsonlManifest(tmp_path / "regions.jsonl", "region_id")
    detect_pages(
        [_page_row(path)],
        manifest,
        repo_root=repo_root,
        settings=_settings(repo_root),
        debug_dir=tmp_path / "debug",
    )
    assert sha256_file(path) == before
    debug = tmp_path / "debug" / "MHGR_000001" / "page_0001.png"
    assert debug.is_file()
    assert debug.read_bytes() != path.read_bytes()


def test_rgb_and_grayscale_detect_the_same_boxes(repo_root: Path, tmp_path: Path) -> None:
    gray = _canvas(width=240, height=180)
    _lines(gray, 20, 30, 100, 3)
    gray_path = tmp_path / "gray.png"
    rgb_path = tmp_path / "rgb.png"
    _save(gray_path, gray, "L")
    _save(rgb_path, np.stack([gray, gray, gray], axis=-1), "RGB")
    from indic_ocr_pipeline.detection.regions import _load_gray

    settings = _settings(repo_root)
    left = detect_regions_on_array(_load_gray(gray_path, "gray"), settings)
    right = detect_regions_on_array(_load_gray(rgb_path, "rgb"), settings)
    assert left == right
    assert left


def test_detector_source_does_not_reference_ocr_or_network() -> None:
    source = Path(__file__).resolve().parents[1] / "src/indic_ocr_pipeline/detection/regions.py"
    text = source.read_text(encoding="utf-8").lower()
    for banned in (
        "tesseract",
        "indicocr",
        "chitrapathak",
        "easyocr",
        "paddleocr",
        "http://",
        "https://",
    ):
        assert banned not in text
