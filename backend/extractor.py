"""Document extraction engine — 100% local.

Pipeline per page
    image ─► layout boxes (DocLayout-YOLO on MPS/CPU, or OpenCV fallback)
          ─► ink detector (OpenCV HSV): stamps, signatures, emblems, handwriting
          ─► merge + classify (YOLO has no stamp/signature/emblem classes)
          ─► crops → PNG (+ optional transparent PNG)
          ─► stamp ink suppressed on an OCR copy of the page (text under stamps)
          ─► batch OCR of text-bearing regions (Surya via local llama.cpp, or Tesseract)
          ─► reading order (XY-cut with form-row awareness) ─► JSON record

Model notes (verified against surya-ocr 0.24 / doclayout-yolo 0.0.4):
  * Surya 2.x is a vision-language model served by a local llama.cpp server
    (Metal on Apple Silicon). It is not a PyTorch/MPS model any more. We feed it
    our own regions as a `LayoutResult` so every text region is OCR'd in one batch.
  * DocLayout-YOLO (DocStructBench) classes: title, plain_text, abandon, figure,
    figure_caption, table, table_caption, table_footnote, isolate_formula,
    formula_caption. Stamps/signatures/emblems come from the ink detector.
"""
from __future__ import annotations

import gc
import io
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .config import Settings, get_settings
from .postprocess import clean_text, detect_script, extract_fields, normalize_for_search

log = logging.getLogger("nepdoc.extractor")

SCHEMA_VERSION = "1.0"

# Layer types produced by this module
TEXT_TYPES = {"title", "text", "header_footer", "caption", "footnote", "table", "handwriting"}
VISUAL_TYPES = {"emblem", "logo", "figure", "stamp", "signature", "table"}
CROP_TYPES = {"emblem", "logo", "figure", "stamp", "signature", "table"}
TRANSPARENT_TYPES = {"emblem", "logo", "stamp", "signature"}
TYPE_LIST = sorted(TEXT_TYPES | VISUAL_TYPES | {"formula"})

YOLO_TO_TYPE = {
    "title": "title", "plain_text": "text", "plain text": "text", "abandon": "header_footer",
    "figure": "figure", "figure_caption": "caption", "table": "table", "table_caption": "caption",
    "table_footnote": "footnote", "isolate_formula": "formula", "formula_caption": "caption",
}
SURYA_LABEL = {
    "text": "Text", "title": "SectionHeader", "header_footer": "PageHeader", "caption": "Caption",
    "footnote": "Footnote", "table": "Table", "handwriting": "Text",
}


# ---------------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------------

@dataclass
class Box:
    type: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float
    source: str  # "doclayout-yolo" | "heuristic" | "ink-detector"
    attributes: dict = field(default_factory=dict)

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def clip(self, width: int, height: int) -> "Box":
        self.x0, self.x1 = max(0.0, min(self.x0, width)), max(0.0, min(self.x1, width))
        self.y0, self.y1 = max(0.0, min(self.y0, height)), max(0.0, min(self.y1, height))
        return self

    def as_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1))


def layer_of(kind: str) -> str:
    """Coarse layer for UI filtering: visual | text | table | other."""
    if kind == "table":
        return "table"
    if kind in VISUAL_TYPES:
        return "visual"
    return "text" if kind in TEXT_TYPES else "other"


def intersection(a: Box, b: Box) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    return inter / (a.area + b.area - inter) if inter else 0.0


def coverage(inner: Box, outer: Box) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    return intersection(inner, outer) / inner.area if inner.area else 0.0


# ---------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------

def select_device(preference: str = "auto") -> str:
    """'mps' on Apple Silicon when usable, otherwise 'cpu'. Never raises."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            torch.zeros(1, device="mps")  # catches broken MPS installs early
            return "mps"
    except Exception as exc:  # noqa: BLE001 - torch missing or MPS unusable
        log.info("MPS not usable (%s); using CPU", exc)
    return "cpu"


def free_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------------

class LayoutDetector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> list[Box]: ...


class YoloLayout:
    """DocLayout-YOLO (AGPL-3.0 package; weights Apache-2.0)."""

    def __init__(self, settings: Settings):
        from doclayout_yolo import YOLOv10  # heavy import, deferred

        weights = settings.yolo_weights_path
        if not weights.exists():
            raise FileNotFoundError(f"DocLayout-YOLO weights not found at {weights}. "
                                    "Run: python scripts/download_models.py")
        self.model = YOLOv10(str(weights))
        self.device = select_device(settings.device)
        self.imgsz, self.conf = settings.yolo_imgsz, settings.yolo_conf
        self.name = f"doclayout-yolo@{self.device}"

    def _predict(self, image: Image.Image, device: str):
        return self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device=device, verbose=False)

    def detect(self, image: Image.Image) -> list[Box]:
        try:
            results = self._predict(image, self.device)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device == "cpu":
                raise
            log.warning("YOLO on %s failed (%s); retrying on CPU", self.device, exc)
            self.device, self.name = "cpu", "doclayout-yolo@cpu"
            results = self._predict(image, "cpu")
        res = results[0]
        names = res.names
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        boxes = []
        for (x0, y0, x1, y1), c, p in zip(xyxy, cls, conf):
            label = str(names.get(int(c), c)).lower()
            boxes.append(Box(YOLO_TO_TYPE.get(label, "text"), float(p), float(x0), float(y0), float(x1),
                             float(y1), "doclayout-yolo", {"model_label": label}))
        del results
        return boxes


class HeuristicLayout:
    """OpenCV fallback when DocLayout-YOLO isn't installed: ruled tables + text blocks.

    Good enough for clean forms and letters; use YOLO for real layouts.
    """

    name = "heuristic-opencv"

    def detect(self, image: Image.Image) -> list[Box]:
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        # Max channel: black print is dark in every channel, pale coloured security
        # backgrounds (green/yellow microtext on citizenship cards) are bright in one.
        gray = white_balance(rgb).max(axis=2)
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
        binary = adaptive & ((gray < otsu).astype(np.uint8) * 255)
        ink_any = cv2.dilate(ink_masks(rgb)["any"], np.ones((7, 7), np.uint8))
        binary[ink_any > 0] = 0  # coloured ink (stamps, emblems, signatures) is the ink detector's job

        horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1)))
        vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 30))))
        boxes: list[Box] = []
        grid = cv2.dilate(horiz | vert, np.ones((5, 5), np.uint8))
        for c in cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(c)
            roi_v, roi_h = vert[y:y + bh, x:x + bw], horiz[y:y + bh, x:x + bw]
            if bw * bh > 0.02 * w * h and roi_v.any() and roi_h.any():
                n_v = len(cv2.findContours(roi_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
                if n_v >= 2:
                    boxes.append(Box("table", 0.6, x, y, x + bw, y + bh, self.name))

        text = cv2.subtract(binary, horiz | vert)
        merged = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 45), max(3, h // 250))))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h // 120))))
        heights = []
        for c in cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < w * 0.02 or bh < h * 0.008 or bw * bh < 0.0006 * w * h:
                continue
            b = Box("text", 0.5, x, y, x + bw, y + bh, self.name)
            if any(coverage(b, t) > 0.6 for t in boxes if t.type == "table"):
                continue
            boxes.append(b)
            heights.append(bh)
        # The first prominent block in the top fifth is likely the heading.
        tops = sorted((b for b in boxes if b.type == "text" and b.y0 < 0.2 * h), key=lambda b: b.y0)
        if tops and heights and (tops[0].h > 1.15 * float(np.median(heights)) or len(tops) == 1):
            tops[0].type = "title"
        return boxes


# ---------------------------------------------------------------------------------
# Ink detection (stamps, signatures, emblems, handwriting)
# ---------------------------------------------------------------------------------

def white_balance(rgb: np.ndarray) -> np.ndarray:
    """Neutralise the colour cast of phone photos using the paper colour.

    Indoor/LED light makes paper bluish or yellowish; without this, black print
    reads as "blue ink" and skin/wood as "red ink".
    """
    f = rgb.astype(np.float32)
    lum = f.mean(axis=2)
    bright = f[lum >= np.percentile(lum, 70)]
    if bright.size == 0:
        return rgb
    paper = np.maximum(np.median(bright, axis=0), 1.0)
    return np.clip(f * (paper.mean() / paper), 0, 255).astype(np.uint8)


def ink_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Binary masks of red and blue ink (HSV after white balance). Black print and
    pale security backgrounds are excluded by requiring real saturation."""
    hsv = cv2.cvtColor(white_balance(rgb), cv2.COLOR_RGB2HSV)
    hch, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((hch <= 8) | (hch >= 168)) & (s >= 90) & (v >= 70)).astype(np.uint8) * 255
    blue = ((hch >= 100) & (hch <= 130) & (s >= 90) & (v >= 60)).astype(np.uint8) * 255
    k = np.ones((2, 2), np.uint8)
    red, blue = cv2.morphologyEx(red, cv2.MORPH_OPEN, k), cv2.morphologyEx(blue, cv2.MORPH_OPEN, k)
    return {"red": red, "blue": blue, "any": red | blue}


def border_ink_ratio(roi: np.ndarray) -> float:
    """Share of ink lying on an ellipse ring or rectangular frame of the box.

    Seals have a ring/frame (high); signatures and handwriting don't (low).
    """
    h, w = roi.shape
    ink = roi > 0
    total = ink.sum()
    if total == 0:
        return 0.0
    yy, xx = np.mgrid[:h, :w]
    r = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    ring = (r >= 0.70 ** 2) & (r <= 1.08 ** 2)
    m = max(2, int(0.12 * min(h, w)))
    frame = np.zeros_like(ink)
    frame[:m, :] = frame[-m:, :] = frame[:, :m] = frame[:, -m:] = True
    return float(max((ink & ring).sum(), (ink & frame).sum()) / total)


def detect_ink(rgb: np.ndarray) -> tuple[list[Box], dict[str, np.ndarray]]:
    h, w = rgb.shape[:2]
    masks = ink_masks(rgb)
    found: list[Box] = []
    for color in ("red", "blue"):
        mask = masks[color]
        if cv2.countNonZero(mask) < 150:
            continue
        k = max(5, int(0.012 * w))
        grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        for c in cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(c)
            roi = mask[y:y + bh, x:x + bw]
            ink = cv2.countNonZero(roi)
            if max(bw, bh) < 0.04 * w or ink < 250:
                continue
            fill, aspect = ink / float(bw * bh), bw / float(bh)
            cy = (y + bh / 2) / h
            n_strokes = cv2.connectedComponents(cv2.dilate(roi, np.ones((3, 3), np.uint8)))[0] - 1
            ring = border_ink_ratio(roi)
            attrs = {"ink_color": color, "ink_fill": round(fill, 3), "aspect": round(aspect, 2),
                     "strokes": int(n_strokes), "ring": round(ring, 2)}
            big = min(bw, bh) >= 0.07 * w
            if fill > 0.5 and big:
                kind, conf = "figure", 0.45          # solid colour block: photo, sticker, graphic
            elif cy < 0.25 and color == "red" and 0.6 < aspect < 1.7 and min(bw, bh) > 0.08 * w:
                kind, conf = "emblem", 0.6           # coloured crest along the top (centre or left)
            elif 0.55 <= aspect <= 1.8 and big and 0.05 <= fill <= 0.45 \
                    and (ring >= 0.7 or (ring >= 0.5 and n_strokes >= 3)):
                kind = "stamp"                       # seal: ink on a ring/frame + lettering inside
                conf = 0.5 + 0.25 * min(1.0, ring) + 0.15 * min(1.0, fill / 0.3)
            elif big and fill < 0.2 and ring < 0.5:
                kind, conf = "signature", 0.5        # large, sparse, no frame: pen strokes
            elif fill < 0.07 and max(bw, bh) > 0.1 * w and n_strokes <= 20:
                kind, conf = "signature", 0.5        # a few long, sparse pen strokes
            elif cy < 0.2 and bh < 0.08 * h:
                kind, conf = "text", 0.5             # coloured letterhead print (नेपाल सरकार …)
                attrs["colored_print"] = True
            elif aspect >= 1.8 and bh <= 0.15 * h and fill < 0.4 and cy > 0.45 and n_strokes <= 12:
                kind, conf = "signature", 0.5        # wide, few connected strokes, lower half
            else:
                kind, conf = "handwriting", 0.45     # coloured pen fill-in → OCR it
            found.append(Box(kind, round(min(conf, 0.95), 3), x, y, x + bw, y + bh, "ink-detector", attrs))
    return _merge_overlapping_ink(found), masks


_INK_PRIORITY = {"emblem": 0, "stamp": 1, "figure": 2, "signature": 3, "handwriting": 4, "text": 5}


def _merge_overlapping_ink(boxes: list[Box]) -> list[Box]:
    """Multi-colour marks (emblem: red ring + blue centre) come out as nested blobs."""
    boxes = sorted(boxes, key=lambda b: -b.area)
    merged: list[Box] = []
    for b in boxes:
        host = next((m for m in merged if coverage(b, m) > 0.6 or iou(b, m) > 0.5), None)
        if host is None:
            merged.append(b)
            continue
        host.x0, host.y0 = min(host.x0, b.x0), min(host.y0, b.y0)
        host.x1, host.y1 = max(host.x1, b.x1), max(host.y1, b.y1)
        if host.attributes.get("ink_color") != b.attributes.get("ink_color"):
            host.attributes["ink_color"] = "mixed"
        if _INK_PRIORITY[b.type] < _INK_PRIORITY[host.type]:
            host.type = b.type
        host.confidence = max(host.confidence, b.confidence)
    return merged


def _same_row_label_left(ink_box: Box, layout: list[Box], width: int) -> bool:
    """A printed label on the same line just to the left ("निवेदकको नाम: ____") means the
    ink is a handwritten form value, not a signature."""
    for b in layout:
        if b.type not in TEXT_TYPES or b.x1 > ink_box.x0 + 0.02 * width:
            continue
        overlap = min(b.y1, ink_box.y1) - max(b.y0, ink_box.y0)
        if overlap > 0.5 * min(b.h, ink_box.h) and ink_box.x0 - b.x1 < 0.3 * width:
            return True
    return False


def merge_layers(layout: list[Box], ink: list[Box], width: int, height: int) -> list[Box]:
    """Combine YOLO/heuristic boxes with ink findings and refine visual classes."""
    for ink_box in ink:
        ink_box.clip(width, height)
        if ink_box.type == "signature" and _same_row_label_left(ink_box, layout, width):
            ink_box.type = "handwriting"
            ink_box.attributes["reason"] = "next to a printed label"
    # Layout boxes that sit (mostly) inside a stamp/emblem/signature/handwriting are the
    # same object seen by the layout model — e.g. YOLO 'plain_text' on a seal's lettering.
    layout = [b for b in layout
              if not (b.type in TEXT_TYPES - {"table"}
                      and any(coverage(b, i) >= 0.6 for i in ink))]
    out = [b.clip(width, height) for b in layout if b.area > 4]
    for b in out:
        # A YOLO 'figure' near the top centre is usually the government emblem / logo.
        if b.type == "figure" and b.y1 < 0.3 * height and 0.6 < b.w / max(b.h, 1) < 1.7:
            b.type = "emblem" if 0.25 < (b.x0 + b.w / 2) / width < 0.75 else "logo"
    for ink_box in ink:
        host = max(out, key=lambda b: iou(b, ink_box), default=None)
        if host and (iou(host, ink_box) > 0.35 or coverage(host, ink_box) > 0.7 or coverage(ink_box, host) > 0.7) \
                and host.type in {"figure", "emblem", "logo", "header_footer"}:
            # YOLO saw *something* (often 'figure'/'abandon'); the ink tells us what.
            host.attributes.update(ink_box.attributes, model_type=host.type)
            host.type = ink_box.type if ink_box.type != "handwriting" else host.type
            host.confidence = round(max(host.confidence, ink_box.confidence), 3)
            continue
        if ink_box.type == "handwriting" and any(coverage(ink_box, b) > 0.8 for b in out if b.type in TEXT_TYPES):
            continue  # already inside a text region that will be OCR'd
        out.append(ink_box)
    # Note which text regions a stamp sits on (OCR uses a stamp-free copy for them).
    for s in (b for b in out if b.type == "stamp"):
        for t in (b for b in out if b.type in TEXT_TYPES and b is not s):
            if intersection(s, t) > 0:
                t.attributes["overlapped_by_stamp"] = True
    return out


# ---------------------------------------------------------------------------------
# Crops & transparency
# ---------------------------------------------------------------------------------

def make_transparent(rgb_crop: np.ndarray, ink_only: np.ndarray | None = None) -> np.ndarray:
    """RGBA where paper becomes transparent.

    * logos/emblems/signatures: alpha from colour distance to the paper colour
      (estimated from the crop border, in Lab so it is perceptual)
    * stamps: pass `ink_only` (the stamp's ink mask) to keep only the seal and
      drop black text printed underneath it
    """
    h, w = rgb_crop.shape[:2]
    if ink_only is not None:
        mask = cv2.dilate(ink_only, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(mask, (0, 0), 0.8)
    else:
        lab = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        m = max(2, int(0.05 * min(h, w)))
        border = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                                 lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
        paper = np.median(border, axis=0)
        dist = np.linalg.norm(lab - paper, axis=2)
        alpha = np.clip((dist - 12.0) / (40.0 - 12.0), 0, 1) * 255
        # Only paper *connected to the crop edge* is background; white shapes inside a
        # logo (the emblem's mountain, letter counters) stay opaque.
        paperlike = (alpha < 128).astype(np.uint8)
        n, labels = cv2.connectedComponents(paperlike, connectivity=4)
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        background = np.isin(labels, edge[edge > 0]) if n > 1 else np.zeros_like(paperlike, bool)
        alpha = np.where(background, alpha, 255).astype(np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    return np.dstack([rgb_crop, alpha.astype(np.uint8)])


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, "PNG", optimize=True)  # PNG from pixels: no EXIF/GPS carried over


def suppress_stamp_ink(rgb: np.ndarray, stamps: list[Box], masks: dict[str, np.ndarray]) -> np.ndarray:
    """Copy of the page with stamp ink painted paper-white, so OCR can read text under seals."""
    out = rgb.copy()
    for s in stamps:
        x0, y0, x1, y1 = s.as_int()
        color = s.attributes.get("ink_color")
        mask = masks[color] if color in ("red", "blue") else masks["any"]
        region = cv2.dilate(mask[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) > 0
        out[y0:y1, x0:x1][region] = 255
    return out


# ---------------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------------

class OcrEngine(Protocol):
    name: str

    def recognize(self, image: Image.Image, regions: list[tuple[str, Box]]) -> dict[str, dict]: ...


def html_to_text(html: str) -> str:
    if not html:
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    rows = soup.find_all("tr")
    if rows:
        return "\n".join(" | ".join(c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])) for r in rows)
    return soup.get_text("\n", strip=True)


class SuryaOcr:
    """Surya 2.x: one batched request set to a local llama.cpp server (spawned on demand).

    Requires `llama-server` (brew install llama.cpp) and the cached GGUF weights
    (scripts/download_models.py). Runs fully offline once cached.
    """

    def __init__(self):
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        import surya

        self.manager = SuryaInferenceManager()
        self.predictor = RecognitionPredictor(self.manager)
        self.predictor.disable_tqdm = True
        self.name = f"surya-ocr {getattr(surya, '__version__', '')} ({self.manager.method})".replace("  ", " ")

    def recognize(self, image: Image.Image, regions: list[tuple[str, Box]]) -> dict[str, dict]:
        from surya.layout.schema import LayoutBox, LayoutResult

        if not regions:
            return {}
        w, h = image.size
        boxes = []
        for i, (cid, b) in enumerate(regions):
            # Token budget hint: denser for tables, ~1 token per 900 px² of text area.
            count = int(min(3000, max(50, b.area / (600 if b.type == "table" else 900))) // 50 * 50)
            boxes.append(LayoutBox(polygon=[b.x0, b.y0, b.x1, b.y1], label=SURYA_LABEL.get(b.type, "Text"),
                                   raw_label=b.type, position=i, count=count, confidence=b.confidence))
        layout = LayoutResult(bboxes=boxes, image_bbox=[0, 0, w, h])
        page = self.predictor([image.convert("RGB")], [layout], full_page=False)[0]
        out: dict[str, dict] = {}
        for (cid, b), block in zip(regions, page.blocks):
            if block.skipped or block.error:
                continue
            out[cid] = {"text": html_to_text(block.html), "confidence": block.confidence,
                        "html": block.html if b.type == "table" else None}
        return out

    def close(self) -> None:
        try:
            self.manager.stop()
        except Exception:  # noqa: BLE001
            pass


def _erase_rules(gray: Image.Image) -> Image.Image:
    a = np.asarray(gray).copy()
    inv = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h, w = a.shape
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 20), 1)))
    lines |= cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 8))))
    a[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 255
    return Image.fromarray(a)


class TesseractOcr:
    """Fallback OCR (printed text only). `brew install tesseract tesseract-lang`."""

    def __init__(self, langs: str):
        import pytesseract

        installed = set(pytesseract.get_languages(config=""))
        chosen = [lang for lang in langs.split("+") if lang in installed]
        if not chosen:
            raise RuntimeError(f"none of {langs} installed for Tesseract")
        self.langs = "+".join(chosen)
        self.name = f"tesseract {pytesseract.get_tesseract_version()} ({self.langs})"

    def recognize(self, image: Image.Image, regions: list[tuple[str, Box]]) -> dict[str, dict]:
        import pytesseract

        out = {}
        for cid, b in regions:
            x0, y0, x1, y1 = b.as_int()
            pad = 8
            crop = image.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad)).convert("L")
            if b.type == "table":
                crop = _erase_rules(crop)  # Tesseract reads ruled tables as one line otherwise
            if crop.height < 60:
                f = 60 / max(crop.height, 1)
                crop = crop.resize((int(crop.width * f), 60), Image.LANCZOS)
            data = pytesseract.image_to_data(crop, lang=self.langs, config="--oem 1 --psm 6",
                                             output_type=pytesseract.Output.DICT)
            lines: dict[tuple, list[str]] = {}
            confs = []
            for i, word in enumerate(data["text"]):
                if word.strip() and float(data["conf"][i]) >= 0:
                    lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
                    confs.append(float(data["conf"][i]))
            out[cid] = {"text": "\n".join(" ".join(v) for _, v in sorted(lines.items())),
                        "confidence": round(sum(confs) / len(confs) / 100, 3) if confs else None, "html": None}
        return out


# ---------------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------------

def _gaps(intervals: list[tuple[float, float, int]], min_gap: float):
    """Merge 1-D intervals; return groups (lists of item idx) and the gaps between them."""
    intervals = sorted(intervals)
    groups, gaps = [[intervals[0][2]]], []
    end = intervals[0][1]
    for a, b, idx in intervals[1:]:
        if a - end > min_gap:
            gaps.append(a - end)
            groups.append([idx])
        else:
            groups[-1].append(idx)
        end = max(end, b)
    return groups, gaps


def _row_pairs(left: list[Box], right: list[Box]) -> float:
    """Share of left items that sit on the same row as a right item (form label → value)."""
    if not left:
        return 0.0
    paired = 0
    for lb in left:
        for rb in right:
            ov = min(lb.y1, rb.y1) - max(lb.y0, rb.y0)
            if ov > 0.5 * min(lb.h, rb.h):
                paired += 1
                break
    return paired / len(left)


def reading_order(boxes: list[Box], width: int, height: int) -> list[int]:
    """Recursive XY-cut: top-to-bottom, left-to-right, columns kept together.

    * Rows that share a column gutter are grouped first, so two text columns under a
      full-width heading are read column by column (classic XY-cut reads them row by row).
    * Short items paired line-by-line across the gutter ("label: value" forms) are
      read as rows instead of columns.
    """
    def gutter(idx: list[int]) -> tuple[float, float] | None:
        cols, vgaps = _gaps([(boxes[i].x0, boxes[i].x1, i) for i in idx], 0.012 * width)
        if len(cols) < 2:
            return None
        k = int(np.argmax(vgaps))
        return max(boxes[i].x1 for i in cols[k]), min(boxes[i].x0 for i in cols[k + 1])

    def form_like(idx: list[int]) -> bool:
        cols, _ = _gaps([(boxes[i].x0, boxes[i].x1, i) for i in idx], 0.012 * width)
        if len(cols) < 2:
            return False
        short = np.mean([boxes[i].h for i in idx]) < 0.05 * height
        return short and _row_pairs([boxes[i] for i in cols[0]], [boxes[i] for g in cols[1:] for i in g]) >= 0.5

    def cut(idx: list[int]) -> list[int]:
        if len(idx) <= 1:
            return idx
        rows, _ = _gaps([(boxes[i].y0, boxes[i].y1, i) for i in idx], 0.004 * height)
        cols, _ = _gaps([(boxes[i].x0, boxes[i].x1, i) for i in idx], 0.012 * width)
        if len(cols) > 1 and not form_like(idx):
            return [i for g in cols for i in cut(g)]
        if len(rows) > 1:
            bands: list[list[int]] = []
            for g in rows:
                gt, prev = gutter(g), gutter(bands[-1]) if bands else None
                if bands and gt and prev and min(gt[1], prev[1]) > max(gt[0], prev[0]) \
                        and not form_like(g) and not form_like(bands[-1]):
                    bands[-1] = bands[-1] + g          # same column structure → same band
                else:
                    bands.append(list(g))
            if len(bands) > 1:
                return [i for b in bands for i in cut(b)]
            return [i for g in rows for i in cut(g)]
        if len(cols) > 1:
            return [i for g in cols for i in cut(g)]
        return sorted(idx, key=lambda i: (round(boxes[i].y0 / max(1, 0.01 * height)), boxes[i].x0))

    return cut(list(range(len(boxes))))


# ---------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------

@dataclass
class Options:
    transparent: bool = True            # write *-alpha.png for logos/stamps/signatures
    suppress_stamp_ink: bool = True     # OCR text under stamps from a stamp-free copy
    ocr_stamps: bool = False            # stamps often hold curved text; usually noise
    ocr_tables: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "Options":
        d = d or {}
        return cls(**{k: bool(d[k]) for k in cls.__dataclass_fields__ if k in d})


StageCallback = Callable[[str], None]


class Extractor:
    """Loads models once; `process()` is not re-entrant (run it from one worker)."""

    def __init__(self, settings: Settings | None = None, layout: LayoutDetector | None = None,
                 ocr: OcrEngine | None = None):
        self.settings = settings or get_settings()
        self.warnings: list[str] = []
        self.layout = layout or self._load_layout()
        self.ocr = ocr if ocr is not None else self._load_ocr()

    # -- model loading with clean fallbacks ------------------------------------
    def _load_layout(self) -> LayoutDetector:
        engine = self.settings.layout_engine
        if engine in ("auto", "yolo"):
            try:
                return YoloLayout(self.settings)
            except Exception as exc:  # noqa: BLE001
                if engine == "yolo":
                    raise
                self.warnings.append(f"DocLayout-YOLO unavailable ({exc}); using OpenCV heuristic layout")
                log.warning(self.warnings[-1])
        return HeuristicLayout()

    def _load_ocr(self) -> OcrEngine | None:
        engine = self.settings.ocr_engine
        if engine == "none":
            return None
        if engine in ("auto", "surya"):
            try:
                if not shutil.which("llama-server") and not _surya_has_external_backend():
                    raise RuntimeError("llama-server not found (brew install llama.cpp)")
                return SuryaOcr()
            except Exception as exc:  # noqa: BLE001
                if engine == "surya":
                    raise
                self.warnings.append(f"Surya OCR unavailable ({exc}); falling back to Tesseract")
                log.warning(self.warnings[-1])
        try:
            return TesseractOcr(self.settings.tesseract_langs)
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"No OCR engine available ({exc}); text will be empty")
            log.warning(self.warnings[-1])
            return None

    def engines(self) -> dict:
        return {"layout": self.layout.name, "ocr": self.ocr.name if self.ocr else None,
                "device": getattr(self.layout, "device", "cpu")}

    # -- main entry --------------------------------------------------------------
    def process(self, doc_id: str, pages: list[Image.Image], source: dict, options: Options | None = None,
                on_stage: StageCallback | None = None) -> dict:
        opts = options or Options()
        stage = on_stage or (lambda s: None)
        timings: dict[str, int] = {}
        all_components: list[dict] = []
        page_meta: list[dict] = []
        warnings = list(self.warnings)
        t_start = time.perf_counter()

        for page_no, image in enumerate(pages, start=1):
            image = image.convert("RGB")
            rgb = np.asarray(image)
            height, width = rgb.shape[:2]
            page_path = self.settings.pages_dir / doc_id / f"page-{page_no}.png"
            save_png(rgb, page_path)

            stage(f"page {page_no}/{len(pages)}: detecting layout")
            t = time.perf_counter()
            layout_boxes = self.layout.detect(image)
            timings["layout_ms"] = timings.get("layout_ms", 0) + int((time.perf_counter() - t) * 1000)
            free_accelerator_memory()

            stage(f"page {page_no}/{len(pages)}: finding stamps & signatures")
            t = time.perf_counter()
            ink_boxes, masks = detect_ink(rgb)
            boxes = merge_layers(layout_boxes, ink_boxes, width, height)
            boxes = [b for b in boxes if b.w >= 4 and b.h >= 4]
            timings["ink_ms"] = timings.get("ink_ms", 0) + int((time.perf_counter() - t) * 1000)

            order = reading_order(boxes, width, height)
            ordered = [boxes[i] for i in order]
            ids = [f"p{page_no}-c{n:03d}" for n in range(1, len(ordered) + 1)]

            stage(f"page {page_no}/{len(pages)}: cropping visual layers")
            assets: dict[str, dict] = {}
            for cid, b in zip(ids, ordered):
                if b.type not in CROP_TYPES:
                    continue
                x0, y0, x1, y1 = b.as_int()
                crop = rgb[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                rel = Path("assets") / doc_id / f"{cid}.png"
                save_png(crop, self.settings.data_dir / rel)
                entry = {"png": rel.as_posix(), "transparent_png": None}
                if opts.transparent and b.type in TRANSPARENT_TYPES:
                    ink = None
                    if b.type == "stamp":
                        color = b.attributes.get("ink_color")
                        ink = (masks[color] if color in ("red", "blue") else masks["any"])[y0:y1, x0:x1]
                        if cv2.countNonZero(ink) < 50:
                            ink = None
                    rel_a = Path("assets") / doc_id / f"{cid}-alpha.png"
                    save_png(make_transparent(crop, ink), self.settings.data_dir / rel_a)
                    entry["transparent_png"] = rel_a.as_posix()
                assets[cid] = entry

            ocr_results: dict[str, dict] = {}
            if self.ocr is not None:
                wanted = TEXT_TYPES | ({"stamp"} if opts.ocr_stamps else set())
                if not opts.ocr_tables:
                    wanted = wanted - {"table"}
                regions = [(cid, b) for cid, b in zip(ids, ordered) if b.type in wanted]
                if regions:
                    stage(f"page {page_no}/{len(pages)}: reading text ({len(regions)} regions, {self.ocr.name})")
                    stamps = [b for b in ordered if b.type == "stamp"]
                    ocr_image = image
                    if opts.suppress_stamp_ink and stamps:
                        ocr_image = Image.fromarray(suppress_stamp_ink(rgb, stamps, masks))
                    t = time.perf_counter()
                    try:
                        ocr_results = self.ocr.recognize(ocr_image, regions)
                    except Exception as exc:  # noqa: BLE001 - keep the layout result even if OCR dies
                        log.exception("OCR failed")
                        warnings.append(f"OCR failed on page {page_no}: {type(exc).__name__}: {exc}")
                    timings["ocr_ms"] = timings.get("ocr_ms", 0) + int((time.perf_counter() - t) * 1000)

            for n, (cid, b) in enumerate(zip(ids, ordered), start=1):
                r = ocr_results.get(cid, {})
                text = clean_text(r.get("text") or "")
                all_components.append({
                    "id": cid, "page": page_no, "reading_order": len(all_components) + 1,
                    "type": b.type, "layer": layer_of(b.type),
                    "source": b.source, "confidence": round(float(b.confidence), 3),
                    "bbox": {"x0": round(b.x0), "y0": round(b.y0), "x1": round(b.x1), "y1": round(b.y1),
                             "width": round(b.w), "height": round(b.h)},
                    "bbox_norm": {"x0": round(b.x0 / width, 5), "y0": round(b.y0 / height, 5),
                                  "x1": round(b.x1 / width, 5), "y1": round(b.y1 / height, 5)},
                    "text": text or None,
                    "text_normalized": normalize_for_search(text) or None,
                    "text_confidence": round(r["confidence"], 3) if r.get("confidence") is not None else None,
                    "script": detect_script(text) if text else None,
                    "html": r.get("html"),
                    "asset": assets.get(cid),
                    "attributes": b.attributes or {},
                })
            page_meta.append({"page": page_no, "width": width, "height": height,
                              "image_path": page_path.relative_to(self.settings.data_dir).as_posix()})

        full_text = "\n\n".join(c["text"] for c in all_components if c["text"])
        by_type: dict[str, int] = {}
        for c in all_components:
            by_type[c["type"]] = by_type.get(c["type"], 0) + 1
        timings["total_ms"] = int((time.perf_counter() - t_start) * 1000)
        return {
            "schema_version": SCHEMA_VERSION,
            "document_id": doc_id,
            "source": source,
            "engines": self.engines(),
            "options": opts.__dict__,
            "page_count": len(pages),
            "pages": page_meta,
            "components": all_components,
            "fields": extract_fields(all_components),
            "text": full_text,
            "stats": {"components": len(all_components), "by_type": by_type, "timings": timings},
            "warnings": warnings,
        }

    def close(self) -> None:
        if hasattr(self.ocr, "close"):
            self.ocr.close()


def _surya_has_external_backend() -> bool:
    try:
        from surya.settings import settings as s

        return bool(s.SURYA_INFERENCE_URL) or (s.SURYA_INFERENCE_BACKEND or "").lower() in {"mlx", "vllm"}
    except Exception:  # noqa: BLE001
        return False


def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


__all__ = ["Box", "Extractor", "HeuristicLayout", "Options", "YoloLayout", "SuryaOcr", "TesseractOcr",
           "detect_ink", "make_transparent", "merge_layers", "reading_order", "select_device"]

