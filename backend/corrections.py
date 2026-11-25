"""Human corrections: edit, add and delete components on a processed document.

Every change
  * updates the stored JSON record (and the components table via save_result)
  * re-crops the component's PNG (+ transparent PNG) when its box or type changes
  * recomputes reading order for the page and the extracted fields
  * is appended to record["corrections"] with before/after values, so nothing is
    silently lost and corrected documents can be used as an evaluation set
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import Settings
from .extractor import (CROP_TYPES, TRANSPARENT_TYPES, TYPE_LIST, Box, OcrEngine, ink_masks, layer_of,
                        make_transparent, reading_order, save_png)
from .postprocess import clean_text, detect_script, extract_fields, normalize_for_search

MIN_SIDE = 6


class CorrectionError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _page(record: dict, page: int) -> dict:
    for p in record["pages"]:
        if p["page"] == page:
            return p
    raise CorrectionError(f"Page {page} does not exist")


def _component(record: dict, cid: str) -> dict:
    for c in record["components"]:
        if c["id"] == cid:
            return c
    raise CorrectionError(f"Component {cid} not found")


def _clean_bbox(bbox: dict, page: dict) -> dict:
    try:
        x0, y0, x1, y1 = (float(bbox[k]) for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError):
        raise CorrectionError("bbox needs numeric x0, y0, x1, y1") from None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    w, h = page["width"], page["height"]
    x0, x1 = max(0.0, min(x0, w)), max(0.0, min(x1, w))
    y0, y1 = max(0.0, min(y0, h)), max(0.0, min(y1, h))
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        raise CorrectionError("Box is too small or outside the page")
    return {"x0": round(x0), "y0": round(y0), "x1": round(x1), "y1": round(y1),
            "width": round(x1 - x0), "height": round(y1 - y0)}


def _norm(bbox: dict, page: dict) -> dict:
    return {"x0": round(bbox["x0"] / page["width"], 5), "y0": round(bbox["y0"] / page["height"], 5),
            "x1": round(bbox["x1"] / page["width"], 5), "y1": round(bbox["y1"] / page["height"], 5)}


def _page_image(settings: Settings, record: dict, page: int) -> Image.Image:
    path = settings.data_dir / _page(record, page)["image_path"]
    return Image.open(path).convert("RGB")


def _set_text(comp: dict, text: str | None, confidence: float | None, source: str) -> None:
    text = clean_text(text or "")
    comp["text"] = text or None
    comp["text_normalized"] = normalize_for_search(text) or None
    comp["script"] = detect_script(text) if text else None
    comp["text_confidence"] = confidence
    comp["text_source"] = source


def _recrop(settings: Settings, record: dict, comp: dict, transparent: bool) -> None:
    """(Re)write the component's PNGs, or remove them if the type has no crop."""
    doc_id = record["document_id"]
    folder = settings.assets_dir / doc_id
    for suffix in ("", "-alpha"):
        (folder / f"{comp['id']}{suffix}.png").unlink(missing_ok=True)
    if comp["type"] not in CROP_TYPES:
        comp["asset"] = None
        return
    rgb = np.asarray(_page_image(settings, record, comp["page"]))
    b = comp["bbox"]
    crop = rgb[b["y0"]:b["y1"], b["x0"]:b["x1"]]
    rel = Path("assets") / doc_id / f"{comp['id']}.png"
    save_png(crop, settings.data_dir / rel)
    asset = {"png": rel.as_posix(), "transparent_png": None}
    if transparent and comp["type"] in TRANSPARENT_TYPES:
        ink = None
        if comp["type"] == "stamp":
            ink = ink_masks(crop)["any"]
            if cv2.countNonZero(ink) < 50:
                ink = None
        rel_a = Path("assets") / doc_id / f"{comp['id']}-alpha.png"
        save_png(make_transparent(crop, ink), settings.data_dir / rel_a)
        asset["transparent_png"] = rel_a.as_posix()
    comp["asset"] = asset


def _reorder(record: dict) -> None:
    """Recompute reading order per page; IDs stay stable so links keep working."""
    comps = record["components"]
    ordered: list[dict] = []
    for p in sorted(record["pages"], key=lambda p: p["page"]):
        on_page = [c for c in comps if c["page"] == p["page"]]
        boxes = [Box(c["type"], c["confidence"], c["bbox"]["x0"], c["bbox"]["y0"], c["bbox"]["x1"],
                     c["bbox"]["y1"], c["source"]) for c in on_page]
        ordered += [on_page[i] for i in reading_order(boxes, p["width"], p["height"])]
    for n, c in enumerate(ordered, start=1):
        c["reading_order"] = n
    record["components"] = ordered


def _refresh(record: dict, entry: dict) -> dict:
    _reorder(record)
    record["fields"] = extract_fields(record["components"])
    record["text"] = "\n\n".join(c["text"] for c in record["components"] if c.get("text"))
    by_type: dict[str, int] = {}
    for c in record["components"]:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    record["stats"]["components"] = len(record["components"])
    record["stats"]["by_type"] = by_type
    record.setdefault("corrections", []).append({**entry, "at": _now()})
    review = record.setdefault("review", {"verified": False})
    review["edited"], review["edited_at"] = True, _now()
    return record


def _snapshot(comp: dict) -> dict:
    return {k: comp.get(k) for k in ("type", "bbox", "text")}


def ocr_region(ocr: OcrEngine | None, image: Image.Image, comp: dict) -> tuple[str | None, float | None, str | None]:
    if ocr is None:
        raise CorrectionError("No OCR engine is available")
    b = comp["bbox"]
    box = Box(comp["type"] if comp["type"] in {"text", "title", "table", "handwriting", "caption",
                                                "footnote", "header_footer"} else "text",
              1.0, b["x0"], b["y0"], b["x1"], b["y1"], "manual")
    result = ocr.recognize(image, [(comp["id"], box)]).get(comp["id"], {})
    return result.get("text"), result.get("confidence"), result.get("html")


# ---- public operations ---------------------------------------------------------------

def update_component(settings: Settings, record: dict, cid: str, changes: dict,
                     ocr: OcrEngine | None = None) -> dict:
    comp = _component(record, cid)
    before = _snapshot(comp)
    page = _page(record, comp["page"])
    recrop = False
    if "type" in changes and changes["type"] != comp["type"]:
        if changes["type"] not in TYPE_LIST:
            raise CorrectionError(f"Unknown type '{changes['type']}'")
        comp["type"], comp["layer"] = changes["type"], layer_of(changes["type"])
        recrop = True
    if "bbox" in changes:
        new = _clean_bbox(changes["bbox"], page)
        if new != comp["bbox"]:
            comp["bbox"], comp["bbox_norm"] = new, _norm(new, page)
            recrop = True
    if "text" in changes:
        _set_text(comp, changes["text"], 1.0, "manual")
    elif changes.get("reocr"):
        text, conf, html = ocr_region(ocr, _page_image(settings, record, comp["page"]), comp)
        _set_text(comp, text, conf, "ocr")
        comp["html"] = html
    comp["source"] = "manual" if comp["source"] != "manual" else comp["source"]
    comp["confidence"] = 1.0
    comp.setdefault("attributes", {})["corrected"] = True
    if recrop:
        _recrop(settings, record, comp, record.get("options", {}).get("transparent", True))
    return _refresh(record, {"action": "update", "component_id": cid, "before": before, "after": _snapshot(comp)})


def add_component(settings: Settings, record: dict, page_no: int, bbox: dict, kind: str,
                  text: str | None = None, ocr: OcrEngine | None = None, run_ocr: bool = True) -> tuple[dict, str]:
    if kind not in TYPE_LIST:
        raise CorrectionError(f"Unknown type '{kind}'")
    page = _page(record, page_no)
    clean = _clean_bbox(bbox, page)
    used = [int(c["id"].split("-c")[1]) for c in record["components"] if c["page"] == page_no]
    cid = f"p{page_no}-c{(max(used, default=0) + 1):03d}"
    comp = {"id": cid, "page": page_no, "reading_order": 0, "type": kind, "layer": layer_of(kind),
            "source": "manual", "confidence": 1.0, "bbox": clean, "bbox_norm": _norm(clean, page),
            "text": None, "text_normalized": None, "text_confidence": None, "script": None, "html": None,
            "asset": None, "attributes": {"corrected": True, "added_by_user": True}}
    if text:
        _set_text(comp, text, 1.0, "manual")
    elif run_ocr and layer_of(kind) in {"text", "table"}:
        t, conf, html = ocr_region(ocr, _page_image(settings, record, page_no), comp)
        _set_text(comp, t, conf, "ocr")
        comp["html"] = html
    record["components"].append(comp)
    _recrop(settings, record, comp, record.get("options", {}).get("transparent", True))
    return _refresh(record, {"action": "add", "component_id": cid, "before": None, "after": _snapshot(comp)}), cid


def delete_component(settings: Settings, record: dict, cid: str) -> dict:
    comp = _component(record, cid)
    record["components"] = [c for c in record["components"] if c["id"] != cid]
    for suffix in ("", "-alpha"):
        (settings.assets_dir / record["document_id"] / f"{cid}{suffix}.png").unlink(missing_ok=True)
    return _refresh(record, {"action": "delete", "component_id": cid, "before": _snapshot(comp), "after": None})


def set_review(record: dict, verified: bool) -> dict:
    review = record.setdefault("review", {})
    review["verified"] = bool(verified)
    review["verified_at"] = _now() if verified else None
    record.setdefault("corrections", []).append({"action": "review", "verified": bool(verified), "at": _now()})
    return record
