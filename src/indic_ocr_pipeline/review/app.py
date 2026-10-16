"""Local offline review server. No OCR, no network assets, localhost only."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.errors import ManifestConflictError
from indic_ocr_pipeline.ids import make_review_id
from indic_ocr_pipeline.manifests import open_manifest
from indic_ocr_pipeline.review.context import (
    ReviewContextError,
    line_bbox_from_records,
    load_png_bytes,
    page_context_png,
)
from indic_ocr_pipeline.review.decisions import ReviewDecisionError, commit_decision
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger("review.app")


def _index_html() -> str:
    return (Path(__file__).with_name("static") / "index.html").read_text(encoding="utf-8")


class ReviewApp:
    """In-memory index over manifests for one local reviewer."""

    def __init__(self, repo_root: Path, manifests_dir: Path, reviewer_id: str) -> None:
        self.repo_root = Path(repo_root)
        self.manifests_dir = Path(manifests_dir)
        self.reviewer_id = reviewer_id
        self.reload()

    def reload(self) -> None:
        self.reviews = {
            str(row["crop_id"]): row for row in open_manifest(self.manifests_dir, "review").load()
        }
        self.crops = {
            str(row["crop_id"]): row for row in open_manifest(self.manifests_dir, "crops").load()
        }
        self.pages = {
            str(row["page_id"]): row for row in open_manifest(self.manifests_dir, "pages").load()
        }
        self.lines = {
            str(row["line_id"]): row for row in open_manifest(self.manifests_dir, "lines").load()
        }
        self.regions = {
            str(row["region_id"]): row
            for row in open_manifest(self.manifests_dir, "regions").load()
        }
        self.verified = {
            str(row["crop_id"]): row for row in open_manifest(self.manifests_dir, "verified").load()
        }
        batch = open_manifest(self.manifests_dir, "review_batch").load()
        self.batch = sorted(batch, key=lambda row: int(row["sample_index"]))

    def queue(self) -> dict[str, Any]:
        items = []
        next_index = 0
        found_pending = False
        for index, item in enumerate(self.batch):
            crop_id = str(item["crop_id"])
            done = crop_id in self.verified
            items.append(
                {
                    "sample_index": item["sample_index"],
                    "crop_id": crop_id,
                    "review_priority": item.get("review_priority"),
                    "reviewed": done,
                }
            )
            if not done and not found_pending:
                next_index = index
                found_pending = True
        if not found_pending:
            next_index = len(items)
        return {
            "items": items,
            "next_index": next_index,
            "reviewed_count": sum(1 for item in items if item["reviewed"]),
            "pending_count": sum(1 for item in items if not item["reviewed"]),
            "reviewer_id": self.reviewer_id,
            "auto_approval_enabled": False,
            "gold_export": False,
        }

    def sample(self, crop_id: str) -> dict[str, Any]:
        review = self.reviews.get(crop_id)
        crop = self.crops.get(crop_id)
        if review is None:
            raise ReviewDecisionError(f"missing review row for {crop_id}")
        if crop is None:
            raise ReviewDecisionError(f"missing crop {crop_id}")
        if isinstance(review.get("diagnostics"), dict):
            diagnostics = review["diagnostics"]
        else:
            diagnostics = {}
        verified = self.verified.get(crop_id)
        texts = {
            "indicocr_text": diagnostics.get("indicocr_text"),
            "chitrapathak_text": diagnostics.get("chitrapathak_text"),
            "tesseract_text": diagnostics.get("tesseract_text"),
        }
        for key, value in texts.items():
            if value is not None and not isinstance(value, str):
                raise ReviewDecisionError(f"{key} is not a string for {crop_id}")
        queue = self.queue()
        region_id = diagnostics.get("region_id") or crop.get("region_id")
        region = self.regions.get(str(region_id)) if region_id else None
        return {
            "crop_id": crop_id,
            "review_id": make_review_id(crop_id),
            "document_id": review["document_id"],
            "page_id": review["page_id"],
            "region_id": region_id,
            "region_type": None if region is None else region.get("region_type"),
            "line_id": review["parent_line_id"],
            "priority_rank": review["priority_rank"],
            "review_priority": diagnostics.get("review_priority"),
            "review_risk_score": review.get("risk_score"),
            "agreement_category": diagnostics.get("agreement_category"),
            "flags": diagnostics.get("flags") or [],
            "proposed_consensus": diagnostics.get("proposed_consensus"),
            "predictions": diagnostics.get("predictions")
            or {
                "indicocr": texts["indicocr_text"],
                "chitrapathak": texts["chitrapathak_text"],
                "tesseract": texts["tesseract_text"],
            },
            "suggestion": diagnostics.get("proposed_consensus"),
            "data_status": (
                ((verified or {}).get("metadata") or {}).get("data_status")
                or diagnostics.get("data_status")
                or "needs_human_review"
            ),
            "models_configured": diagnostics.get("models_configured"),
            "models_agreed": diagnostics.get("models_agreed"),
            "models_differed": diagnostics.get("models_differed"),
            "decision_reason": diagnostics.get("decision_reason"),
            "gold": ((verified or {}).get("metadata") or {}).get("gold") is True,
            "indicocr_text": texts["indicocr_text"],
            "chitrapathak_text": texts["chitrapathak_text"],
            "tesseract_text": texts["tesseract_text"],
            "indicocr_confidence": diagnostics.get("indicocr_confidence"),
            "chitrapathak_confidence": diagnostics.get("chitrapathak_confidence"),
            "tesseract_confidence": diagnostics.get("tesseract_confidence"),
            "tesseract_confidence_scale": diagnostics.get("tesseract_confidence_scale"),
            "crop_url": f"/api/crop/{crop_id}",
            "context_url": f"/api/context/{crop_id}",
            "already_reviewed": verified is not None,
            "decision_label": None
            if verified is None
            else (verified.get("metadata") or {}).get("decision_label"),
            "draft_text": None if verified is None else verified.get("final_text"),
            "notes": None if verified is None else verified.get("notes"),
            "error_tags": []
            if verified is None
            else (verified.get("metadata") or {}).get("error_tags") or [],
            "reviewed_count": queue["reviewed_count"],
            "pending_count": queue["pending_count"],
            "score_meaning": "review prioritization heuristic, not probability of correctness",
        }

    def crop_bytes(self, crop_id: str) -> bytes:
        crop = self.crops.get(crop_id)
        review = self.reviews.get(crop_id)
        if crop is None:
            raise ReviewDecisionError(f"missing crop {crop_id}")
        diagnostics = {}
        if review and isinstance(review.get("diagnostics"), dict):
            diagnostics = review["diagnostics"]
        rel = crop.get("image_path") or diagnostics.get("image_path")
        if not isinstance(rel, str):
            raise ReviewDecisionError(f"crop {crop_id} has no image_path")
        path = Path(rel)
        if not path.is_absolute():
            path = self.repo_root / path
        return load_png_bytes(path)

    def context_bytes(self, crop_id: str) -> bytes:
        review = self.reviews.get(crop_id)
        crop = self.crops.get(crop_id)
        if review is None or crop is None:
            raise ReviewDecisionError(f"missing crop/review for {crop_id}")
        page = self.pages.get(str(review["page_id"]))
        if page is None:
            raise ReviewContextError(f"missing page {review['page_id']}")
        rel = page.get("image_path")
        if not isinstance(rel, str):
            raise ReviewContextError(f"page {review['page_id']} has no image_path")
        page_path = Path(rel)
        if not page_path.is_absolute():
            page_path = self.repo_root / page_path
        line = self.lines.get(str(review["parent_line_id"]))
        bbox = line_bbox_from_records(line, crop)
        return page_context_png(page_path, bbox)

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        crop_id = payload.get("crop_id")
        if not isinstance(crop_id, str) or not crop_id:
            raise ReviewDecisionError("crop_id is required")
        result = commit_decision(
            self.manifests_dir,
            crop_id=crop_id,
            decision_label=str(payload.get("decision")),
            gold_text=payload.get("gold_text"),
            selected_candidate_text=payload.get("selected_candidate_text"),
            reviewer_id=self.reviewer_id,
            notes=payload.get("notes"),
            error_tags=list(payload.get("error_tags") or []),
            repo_root=self.repo_root,
        )
        self.reload()
        return result


def make_handler(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path in {"/", "/index.html"}:
                    self._send(200, _index_html().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/api/queue":
                    self._json(200, app.queue())
                    return
                if path.startswith("/api/sample/"):
                    crop_id = path.removeprefix("/api/sample/")
                    self._json(200, app.sample(crop_id))
                    return
                if path.startswith("/api/crop/"):
                    crop_id = path.removeprefix("/api/crop/")
                    self._send(200, app.crop_bytes(crop_id), "image/png")
                    return
                if path.startswith("/api/context/"):
                    crop_id = path.removeprefix("/api/context/")
                    self._send(200, app.context_bytes(crop_id), "image/png")
                    return
                self._json(404, {"error": "not found"})
            except (ReviewDecisionError, ReviewContextError, FileNotFoundError) as exc:
                self._json(400, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/decision":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ReviewDecisionError("decision payload must be an object")
                result = app.decide(payload)
                self._json(200, {"ok": True, "outcome": result["outcome"]})
            except ManifestConflictError as exc:
                self._json(409, {"error": str(exc)})
            except (ReviewDecisionError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._json(400, {"error": str(exc)})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Start the offline localhost review UI. "
            "Does not run OCR, download assets, or export gold."
        )
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    settings = load_settings()
    host = args.host or settings.review.host
    port = args.port or settings.review.port
    if host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("review app may bind only to 127.0.0.1 or localhost")
    if not (settings.manifests_dir / "review_batch.jsonl").exists():
        raise SystemExit(
            "missing data/output/review_batch.jsonl; run scripts/sample_review.py first"
        )
    app = ReviewApp(settings.repo_root, settings.manifests_dir, settings.review.reviewer_id)
    handler = make_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"review_app: http://{host}:{port}/", flush=True)
    print(f"reviewer_id: {settings.review.reviewer_id}", flush=True)
    print("auto_approval_enabled: false", flush=True)
    print("gold_export: false", flush=True)
    print("network_assets: none", flush=True)
    logger.info("review app listening on %s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nreview app stopped", flush=True)
        return 0
    return 0
