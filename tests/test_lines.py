"""Physical line detection crops ink bands and does not read text."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.detection.lines import (
    LineConflictError,
    detect_lines_in_region,
    detect_physical_lines,
)
from indic_ocr_pipeline.ids import make_line_id, make_page_id
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.schemas import BBox, CropRecord, PageRecord, RegionRecord, StageStatus
from indic_ocr_pipeline.utils.hashing import sha256_file


def _settings(repo_root: Path):
    return load_settings(repo_root, environ={}).detection


def _canvas(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _fill(image: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    image[y : y + height, x : x + width] = 0


def _region(page_id: str, index: int, bbox: list[int], *, document_id: str = "MHGR_000001") -> dict:
    record = RegionRecord(
        region_id=f"{page_id}_R{index:04d}",
        document_id=document_id,
        page_id=page_id,
        page_number=1,
        region_index=index,
        region_type="text_block",
        bbox=BBox.from_list(bbox),
        detection_method="test",
        status=StageStatus.COMPLETE,
    )
    return record.to_dict()


def _save(path: Path, image: np.ndarray, mode: str = "L") -> None:
    Image.fromarray(image, mode=mode).save(path)


def _page_row(path: Path, *, document_id: str = "MHGR_000001", page_number: int = 1) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    record = PageRecord(
        document_id=document_id,
        page_id=make_page_id(document_id, page_number),
        page_number=page_number,
        image_path=str(path),
        width=width,
        height=height,
        dpi=200,
        sha256=sha256_file(path),
        status=StageStatus.COMPLETE,
    )
    return record.to_dict()


def _run(
    repo_root: Path,
    tmp_path: Path,
    image: np.ndarray,
    regions: list[dict],
    *,
    mode: str = "L",
):
    path = tmp_path / "page.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(path, image, mode)
    lines = JsonlManifest(tmp_path / "lines.jsonl", "line_id")
    crops = JsonlManifest(tmp_path / "crops.jsonl", "crop_id")
    report = detect_physical_lines(
        [_page_row(path)],
        regions,
        lines,
        crops,
        repo_root=repo_root,
        crops_dir=tmp_path / "crops",
        settings=_settings(repo_root),
    )
    return path, lines, crops, report


def test_blank_region_has_no_lines(repo_root: Path) -> None:
    image = _canvas()
    box = BBox(10, 10, 200, 120)
    assert detect_lines_in_region(image, box, "MHGR_000001_P0001_R0001", _settings(repo_root)) == []


def test_one_band_and_three_bands(repo_root: Path) -> None:
    settings = _settings(repo_root)
    single = _canvas()
    _fill(single, 30, 40, 140, 16)
    one = detect_lines_in_region(single, BBox(0, 0, 320, 240), "MHGR_000001_P0001_R0001", settings)
    assert len(one) == 1
    assert one[0].y0 <= 40 and one[0].y1 >= 56
    assert one[0].x0 <= 30 and one[0].x1 >= 170

    triple = _canvas()
    for y in (20, 60, 100):
        _fill(triple, 24, y, 160, 14)
    three = detect_lines_in_region(
        triple, BBox(0, 0, 320, 240), "MHGR_000001_P0001_R0001", settings
    )
    assert len(three) == 3
    assert [line.reading_order for line in three] == [1, 2, 3]
    assert three[0].y1 < three[1].y0
    assert three[1].y1 < three[2].y0


def test_word_gaps_and_connected_glyphs_stay_one_line(repo_root: Path) -> None:
    settings = _settings(repo_root)
    words = _canvas()
    for x in (20, 80, 150):
        _fill(words, x, 50, 36, 14)
    found = detect_lines_in_region(words, BBox(0, 0, 320, 200), "MHGR_000001_P0001_R0001", settings)
    assert len(found) == 1
    assert found[0].x1 - found[0].x0 > 140

    linked = _canvas()
    _fill(linked, 20, 48, 180, 3)
    for x in (20, 60, 100, 140):
        _fill(linked, x, 40, 16, 16)
    joined = detect_lines_in_region(
        linked, BBox(0, 0, 320, 200), "MHGR_000001_P0001_R0001", settings
    )
    assert len(joined) == 1


def test_symbol_row_and_upper_fragment_stay_with_their_line(repo_root: Path) -> None:
    settings = _settings(repo_root)
    symbols = _canvas(width=160, height=90)
    for x in (12, 36, 60, 84, 108):
        _fill(symbols, x, 30, 22, 14)
    kept = detect_lines_in_region(
        symbols, BBox(10, 18, 140, 60), "MHGR_000001_P0001_R0001", settings
    )
    assert len(kept) == 1

    marks = _canvas()
    _fill(marks, 40, 30, 40, 4)
    _fill(marks, 20, 42, 140, 16)
    joined = detect_lines_in_region(
        marks, BBox(0, 0, 320, 200), "MHGR_000001_P0001_R0001", settings
    )
    assert len(joined) == 1
    assert joined[0].y0 <= 30 and joined[0].y1 >= 58


def test_small_short_and_both_font_sizes(repo_root: Path) -> None:
    settings = _settings(repo_root)
    small = _canvas()
    _fill(small, 20, 30, 24, 12)
    short = detect_lines_in_region(small, BBox(0, 0, 200, 120), "MHGR_000001_P0001_R0001", settings)
    assert len(short) == 1
    assert short[0].x1 - short[0].x0 < 40

    large = _canvas(height=180)
    _fill(large, 20, 30, 140, 42)
    _fill(large, 20, 100, 80, 10)
    sizes = detect_lines_in_region(
        large, BBox(0, 0, 320, 180), "MHGR_000001_P0001_R0001", settings
    )
    assert len(sizes) == 2
    assert sizes[0].y1 - sizes[0].y0 > sizes[1].y1 - sizes[1].y0


def test_separate_regions_do_not_merge_and_rules_are_not_lines(
    repo_root: Path, tmp_path: Path
) -> None:
    image = _canvas(width=360, height=220)
    _fill(image, 20, 40, 80, 14)
    _fill(image, 220, 40, 80, 14)
    page_id = "MHGR_000001_P0001"
    regions = [
        _region(page_id, 1, [10, 20, 140, 90]),
        _region(page_id, 2, [200, 20, 340, 90]),
    ]
    _path, lines, _crops, report = _run(repo_root, tmp_path, image, regions)
    assert report.errors == 0
    rows = [json.loads(line) for line in lines.path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["parent_region_id"] for row in rows} == {
        f"{page_id}_R0001",
        f"{page_id}_R0002",
    }
    assert rows[0]["bbox"][2] < rows[1]["bbox"][0]

    table = _canvas(width=300, height=220)
    _fill(table, 20, 20, 240, 3)
    _fill(table, 20, 180, 240, 3)
    _fill(table, 20, 20, 3, 163)
    _fill(table, 257, 20, 3, 163)
    _fill(table, 40, 50, 180, 14)
    _fill(table, 40, 110, 180, 14)
    found = detect_lines_in_region(
        table, BBox(10, 10, 280, 200), "MHGR_000001_P0001_R0001", _settings(repo_root)
    )
    assert len(found) == 2
    assert all(line.y1 - line.y0 > 6 for line in found)


def test_line_boxes_stay_inside_the_region_and_page(repo_root: Path) -> None:
    image = _canvas()
    _fill(image, 4, 8, 40, 14)
    region = BBox(0, 0, 80, 40)
    found = detect_lines_in_region(image, region, "MHGR_000001_P0001_R0001", _settings(repo_root))
    assert len(found) == 1
    box = BBox(found[0].x0, found[0].y0, found[0].x1, found[0].y1)
    assert region.contains(box)
    assert box.x0 >= 0 and box.y0 >= 0
    assert box.x1 <= image.shape[1] and box.y1 <= image.shape[0]


def test_ids_are_stable_and_rerun_skips(repo_root: Path, tmp_path: Path) -> None:
    image = _canvas()
    _fill(image, 20, 30, 120, 14)
    _fill(image, 20, 70, 120, 14)
    page_id = "MHGR_000001_P0001"
    regions = [_region(page_id, 1, [0, 0, 320, 240])]
    path, lines, crops, first = _run(repo_root, tmp_path, image, regions)
    before_page = path.read_bytes()
    stored_lines = lines.path.read_bytes()
    stored_crops = crops.path.read_bytes()
    rows = [json.loads(line) for line in stored_lines.decode().splitlines()]
    assert [row["line_id"] for row in rows] == [
        make_line_id(page_id, 1),
        make_line_id(page_id, 2),
    ]
    assert rows[0]["reading_order"] == 1
    assert rows[1]["reading_order"] == 2
    assert all(row["line_index"] == index for index, row in enumerate(rows, start=1))
    assert all("text" not in row and "gold_text" not in row for row in rows)
    crop_rows = [
        CropRecord.from_dict(json.loads(line)) for line in stored_crops.decode().splitlines()
    ]
    assert len(crop_rows) == 2
    assert all(row.sha256 and (tmp_path / "crops").joinpath(
        "MHGR_000001", page_id, f"{row.parent_line_id}.png"
    ).is_file() for row in crop_rows)

    second = detect_physical_lines(
        [_page_row(path)],
        regions,
        lines,
        crops,
        repo_root=repo_root,
        crops_dir=tmp_path / "crops",
        settings=_settings(repo_root),
    )
    assert first.lines == 2
    assert second.skipped == 2
    assert lines.path.read_bytes() == stored_lines
    assert crops.path.read_bytes() == stored_crops
    assert path.read_bytes() == before_page

    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 unchanged")
    pdf_before = pdf.read_bytes()
    detect_physical_lines(
        [_page_row(path)],
        regions,
        lines,
        crops,
        repo_root=repo_root,
        crops_dir=tmp_path / "crops",
        settings=_settings(repo_root),
    )
    assert pdf.read_bytes() == pdf_before


def test_payload_conflict_and_crop_recovery(repo_root: Path, tmp_path: Path) -> None:
    image = _canvas()
    _fill(image, 30, 40, 100, 16)
    page_id = "MHGR_000001_P0001"
    regions = [_region(page_id, 1, [0, 0, 320, 200])]
    path, lines, crops, _report = _run(repo_root, tmp_path, image, regions)
    crop_path = tmp_path / "crops" / "MHGR_000001" / page_id / f"{make_line_id(page_id, 1)}.png"
    original_crop = crop_path.read_bytes()
    original_lines = lines.path.read_bytes()

    crop_path.unlink()
    restored = detect_physical_lines(
        [_page_row(path)],
        regions,
        lines,
        crops,
        repo_root=repo_root,
        crops_dir=tmp_path / "crops",
        settings=_settings(repo_root),
    )
    assert restored.errors == 0
    assert crop_path.read_bytes() == original_crop
    assert lines.path.read_bytes() == original_lines

    crop_path.write_bytes(b"not-the-crop")
    corrupt = detect_physical_lines(
        [_page_row(path)],
        regions,
        lines,
        crops,
        repo_root=repo_root,
        crops_dir=tmp_path / "crops",
        settings=_settings(repo_root),
    )
    assert corrupt.errors == 1
    assert "SHA-256" in corrupt.details[0].message
    assert crop_path.read_bytes() == b"not-the-crop"

    crop_path.write_bytes(original_crop)
    row = json.loads(original_lines.decode().splitlines()[0])
    row["bbox"] = [1, 2, 30, 40]
    corrupted = (json.dumps(row, sort_keys=True) + "\n").encode()
    lines.path.write_bytes(corrupted)
    conflict = detect_physical_lines(
        [_page_row(path)],
        regions,
        lines,
        crops,
        repo_root=repo_root,
        crops_dir=tmp_path / "crops",
        settings=_settings(repo_root),
    )
    assert conflict.errors == 1
    assert "different payload" in conflict.details[0].message
    assert lines.path.read_bytes() == corrupted
    assert crop_path.read_bytes() == original_crop


def test_failed_append_missing_page_and_corrupt_page(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _canvas()
    _fill(image, 20, 20, 80, 12)
    _fill(image, 20, 70, 80, 12)
    page_id = "MHGR_000001_P0001"
    regions = [_region(page_id, 1, [0, 0, 320, 200])]
    path = tmp_path / "page.png"
    _save(path, image)
    lines = JsonlManifest(tmp_path / "lines.jsonl", "line_id")
    crops = JsonlManifest(tmp_path / "crops.jsonl", "crop_id")
    original = lines.append

    def fail_second(record: dict) -> str:
        if record.get("line_index") == 2:
            raise LineConflictError("append failed")
        return original(record)

    monkeypatch.setattr(lines, "append", fail_second)
    failed = detect_physical_lines(
        [_page_row(path)],
        regions,
        lines,
        crops,
        repo_root=repo_root,
        crops_dir=tmp_path / "crops",
        settings=_settings(repo_root),
    )
    assert failed.errors == 1
    written = [json.loads(line)["line_index"] for line in lines.path.read_text().splitlines()]
    assert written == [1]
    second_crop = tmp_path / "crops" / "MHGR_000001" / page_id / f"{make_line_id(page_id, 2)}.png"
    assert not second_crop.exists()

    missing = detect_physical_lines(
        [
            {
                "document_id": "MHGR_000001",
                "page_id": page_id,
                "page_number": 1,
                "image_path": str(tmp_path / "absent.png"),
                "width": 10,
                "height": 10,
                "sha256": "a" * 64,
            }
        ],
        regions,
        JsonlManifest(tmp_path / "missing-lines.jsonl", "line_id"),
        JsonlManifest(tmp_path / "missing-crops.jsonl", "crop_id"),
        repo_root=repo_root,
        crops_dir=tmp_path / "missing-crops",
        settings=_settings(repo_root),
    )
    assert missing.errors == 1
    assert "missing" in missing.details[0].message

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not-a-png")
    corrupt = detect_physical_lines(
        [
            {
                "document_id": "MHGR_000001",
                "page_id": page_id,
                "page_number": 1,
                "image_path": str(corrupt_path),
                "width": 10,
                "height": 10,
                "sha256": sha256_file(corrupt_path),
            }
        ],
        regions,
        JsonlManifest(tmp_path / "corrupt-lines.jsonl", "line_id"),
        JsonlManifest(tmp_path / "corrupt-crops.jsonl", "crop_id"),
        repo_root=repo_root,
        crops_dir=tmp_path / "corrupt-out",
        settings=_settings(repo_root),
    )
    assert corrupt.errors == 1
    assert "not a readable" in corrupt.details[0].message


def test_rgb_page_matches_grayscale_and_source_has_no_ocr(repo_root: Path, tmp_path: Path) -> None:
    gray = _canvas(width=200, height=120)
    _fill(gray, 16, 30, 90, 14)
    page_id = "MHGR_000001_P0001"
    regions = [_region(page_id, 1, [0, 0, 200, 120])]
    gray_path, gray_lines, _gray_crops, gray_report = _run(
        repo_root, tmp_path / "gray", gray, regions, mode="L"
    )
    rgb = np.stack([gray, gray, gray], axis=-1)
    _rgb_path, rgb_lines, _rgb_crops, rgb_report = _run(
        repo_root, tmp_path / "rgb", rgb, regions, mode="RGB"
    )
    assert gray_report.errors == 0 and rgb_report.errors == 0
    gray_rows = [json.loads(line)["bbox"] for line in gray_lines.path.read_text().splitlines()]
    rgb_rows = [json.loads(line)["bbox"] for line in rgb_lines.path.read_text().splitlines()]
    assert gray_rows == rgb_rows
    assert len(gray_rows) == 1
    assert sha256_file(gray_path) == sha256_file(gray_path)
    source = (
        Path(__file__).resolve().parents[1] / "src/indic_ocr_pipeline/detection/lines.py"
    )
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
