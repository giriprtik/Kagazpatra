"""Extraction engine: ink detection, merging, crops, transparency, reading order, OCR adapters."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.config import get_settings
from backend.extractor import (Box, Extractor, HeuristicLayout, Options, SuryaOcr, detect_ink, html_to_text,
                               make_transparent, merge_layers, reading_order, select_device)
from backend.postprocess import detect_script, extract_fields, parse_bs_date

FIX = Path(__file__).parent / "fixtures"
TRUTH = json.loads((FIX / "nepali_letter.truth.json").read_text())


def _letter():
    return Image.open(FIX / "nepali_letter.png").convert("RGB")


def _overlaps(box: Box, ref: list[int]) -> bool:
    r = Box("x", 1, *ref, "t")
    w = min(box.x1, r.x1) - max(box.x0, r.x0)
    h = min(box.y1, r.y1) - max(box.y0, r.y0)
    return w > 0 and h > 0 and w * h > 0.4 * min(box.area, r.area)


class FakeOcr:
    name = "fake-ocr"

    def __init__(self):
        self.calls = []

    def recognize(self, image, regions):
        self.calls.append((image, regions))
        return {cid: {"text": f"{b.type} text", "confidence": 0.9, "html": None} for cid, b in regions}


# ---- ink detection & classification ------------------------------------------------

def test_ink_detector_finds_emblem_stamp_signature_handwriting():
    rgb = np.asarray(_letter())
    found, _ = detect_ink(rgb)
    kinds = {b.type: b for b in found}
    assert _overlaps(kinds["emblem"], TRUTH["emblem"])
    assert _overlaps(kinds["stamp"], TRUTH["stamp"]) and kinds["stamp"].attributes["ink_color"] == "red"
    assert any(b.type in {"signature", "handwriting"} and _overlaps(b, TRUTH["signature"]) for b in found)
    assert sum(1 for b in found if b.type == "emblem") == 1  # red ring + blue centre merged


def test_handwritten_form_value_is_not_a_signature():
    img = _letter()
    boxes = merge_layers(HeuristicLayout().detect(img), detect_ink(np.asarray(img))[0], *img.size)
    hw = [b for b in boxes if _overlaps(b, TRUTH["handwriting"])]
    sig = [b for b in boxes if _overlaps(b, TRUTH["signature"])]
    assert hw and hw[0].type == "handwriting"
    assert sig and sig[0].type == "signature"


def test_layout_boxes_inside_a_stamp_are_dropped():
    layout = [Box("text", 0.8, 720, 1720, 980, 1980, "doclayout-yolo"),      # YOLO reading the seal lettering
              Box("text", 0.9, 100, 700, 1600, 900, "doclayout-yolo")]
    ink = [Box("stamp", 0.8, 700, 1700, 1000, 2000, "ink-detector", {"ink_color": "red"})]
    out = merge_layers(layout, ink, 1700, 2200)
    assert [b.type for b in out] == ["text", "stamp"]


def test_yolo_figure_upgraded_by_ink():
    layout = [Box("figure", 0.7, 700, 1700, 1000, 2000, "doclayout-yolo", {"model_label": "figure"})]
    ink = [Box("stamp", 0.8, 705, 1705, 995, 1995, "ink-detector", {"ink_color": "red"})]
    out = merge_layers(layout, ink, 1700, 2200)
    assert len(out) == 1 and out[0].type == "stamp" and out[0].attributes["model_type"] == "figure"


def test_yolo_top_figure_becomes_emblem():
    out = merge_layers([Box("figure", 0.7, 750, 50, 950, 250, "doclayout-yolo")], [], 1700, 2200)
    assert out[0].type == "emblem"


# ---- transparency -------------------------------------------------------------------------

def test_transparent_logo_keeps_inner_white():
    crop = np.full((120, 120, 3), 250, np.uint8)
    yy, xx = np.mgrid[:120, :120]
    crop[(yy - 60) ** 2 + (xx - 60) ** 2 < 50 ** 2] = (200, 30, 40)
    crop[(yy - 60) ** 2 + (xx - 60) ** 2 < 20 ** 2] = (250, 250, 250)   # white centre
    rgba = make_transparent(crop)
    assert rgba[2, 2, 3] == 0 and rgba[60, 60, 3] == 255 and rgba[60, 20, 3] == 255


def test_transparent_stamp_keeps_only_ink():
    crop = np.full((100, 100, 3), 250, np.uint8)
    crop[40:60, 10:90] = (20, 20, 20)                  # black text under the stamp
    ink = np.zeros((100, 100), np.uint8)
    ink[5:15, :] = 255
    crop[5:15, :] = (205, 30, 45)
    rgba = make_transparent(crop, ink)
    assert rgba[50, 50, 3] == 0 and rgba[10, 50, 3] > 200


# ---- reading order ----------------------------------------------------------------------

def test_reading_order_two_columns():
    W, H = 1000, 1000
    boxes = [Box("title", 1, 100, 50, 900, 100, "t"),
             Box("text", 1, 520, 150, 900, 400, "t"), Box("text", 1, 100, 150, 480, 400, "t"),
             Box("text", 1, 100, 420, 480, 700, "t"), Box("text", 1, 520, 420, 900, 700, "t")]
    order = reading_order(boxes, W, H)
    assert order == [0, 2, 3, 1, 4]  # title, left column top→bottom, then right column


def test_reading_order_form_rows_keep_label_with_value():
    W, H = 1000, 1000
    boxes = [Box("text", 1, 100, 100, 300, 140, "t"), Box("handwriting", 1, 500, 95, 800, 145, "t"),
             Box("text", 1, 100, 200, 300, 240, "t"), Box("handwriting", 1, 500, 195, 800, 245, "t"),
             Box("text", 1, 100, 300, 300, 340, "t"), Box("handwriting", 1, 500, 295, 800, 345, "t")]
    assert reading_order(boxes, W, H) == [0, 1, 2, 3, 4, 5]


# ---- end-to-end (heuristic layout + fake OCR) -------------------------------------------

def test_process_letter(data_dir):
    ocr = FakeOcr()
    ex = Extractor(get_settings(), layout=HeuristicLayout(), ocr=ocr)
    rec = ex.process("b" * 32, [_letter()], {"filename": "x.png"}, Options())
    types = [c["type"] for c in rec["components"]]
    assert types[0] == "emblem" and {"stamp", "signature", "handwriting", "table", "title"} <= set(types)
    assert [c["reading_order"] for c in rec["components"]] == list(range(1, len(types) + 1))
    # one batched OCR call per page; stamps not OCR'd by default; ink suppressed in the OCR image
    assert len(ocr.calls) == 1
    image, regions = ocr.calls[0]
    assert "stamp" not in {b.type for _, b in regions}
    stamp = next(c for c in rec["components"] if c["type"] == "stamp")
    b = stamp["bbox"]
    sx, sy = int((b["x0"] + b["x1"]) / 2), b["y0"] + 12
    assert np.asarray(image)[sy - 10:sy + 10, sx - 10:sx + 10].min() > 200  # red ring painted white
    # crops on disk, transparent versions only for visual marks
    assert (data_dir / stamp["asset"]["png"]).is_file() and (data_dir / stamp["asset"]["transparent_png"]).is_file()
    table = next(c for c in rec["components"] if c["type"] == "table")
    assert table["asset"]["transparent_png"] is None
    # normalised + pixel boxes agree
    page = rec["pages"][0]
    assert abs(stamp["bbox_norm"]["x0"] * page["width"] - b["x0"]) < 1
    assert rec["engines"]["ocr"] == "fake-ocr" and rec["schema_version"] == "1.0"


def test_ocr_failure_keeps_layout(data_dir):
    class Boom:
        name = "boom"

        def recognize(self, *_):
            raise RuntimeError("server died")

    rec = Extractor(get_settings(), layout=HeuristicLayout(), ocr=Boom()).process("c" * 32, [_letter()], {})
    assert rec["components"] and any("OCR failed" in w for w in rec["warnings"])


def test_missing_yolo_weights_falls_back(monkeypatch):
    monkeypatch.setattr(get_settings().__class__, "layout_engine", "auto", raising=False)
    ex = Extractor.__new__(Extractor)
    ex.settings, ex.warnings = get_settings(), []
    import dataclasses
    ex.settings = dataclasses.replace(get_settings(), layout_engine="auto", yolo_weights="/nope/x.pt")
    layout = ex._load_layout()
    assert isinstance(layout, HeuristicLayout) and "unavailable" in ex.warnings[0]


def test_device_selection_never_raises():
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in {"mps", "cpu"}


# ---- Surya adapter (real surya-ocr 0.24 types, fake predictor) ---------------------------

def test_surya_adapter_batches_regions_and_maps_blocks():
    pytest.importorskip("surya")
    from surya.layout.schema import LayoutResult
    from surya.recognition.schema import BlockOCRResult, PageOCRResult

    seen = {}

    def fake_predictor(images, layouts, full_page=None):
        seen["layout"], seen["full_page"] = layouts[0], full_page
        assert isinstance(layouts[0], LayoutResult)
        return [PageOCRResult(image_bbox=[0, 0, 100, 100], blocks=[
            BlockOCRResult(polygon=[0, 0, 10, 10], label="Text", reading_order=0, html="<p>नेपाल सरकार</p>",
                           confidence=0.93),
            BlockOCRResult(polygon=[0, 0, 10, 10], label="Table", reading_order=1,
                           html="<table><tr><td>कित्ता</td><td>१२३४</td></tr></table>", confidence=0.8)])]

    ocr = SuryaOcr.__new__(SuryaOcr)
    ocr.predictor, ocr.name = fake_predictor, "surya test"
    out = ocr.recognize(Image.new("RGB", (100, 100), "white"),
                        [("a", Box("title", 0.9, 0, 0, 50, 20, "t")), ("b", Box("table", 0.8, 0, 30, 90, 90, "t"))])
    assert seen["full_page"] is False and [b.label for b in seen["layout"].bboxes] == ["SectionHeader", "Table"]
    assert out["a"]["text"] == "नेपाल सरकार" and out["a"]["confidence"] == 0.93
    assert out["b"]["text"] == "कित्ता | १२३४" and out["b"]["html"].startswith("<table>")


def test_html_to_text():
    assert html_to_text("<p>a<br>b</p>") == "a\nb"


# ---- Nepali post-processing -------------------------------------------------------------------

def test_fields_and_dates():
    comps = [{"id": "p1-c001", "text": "प.सं. २०८१/८२\nच.नं. १२५", "text_confidence": 0.9},
             {"id": "p1-c002", "text": "मिति: २०८१/०५/१२   दर्ता नं. ४५६७", "text_confidence": 0.8}]
    f = {x["key"]: x for x in extract_fields(comps)}
    assert f["chalani_no"]["normalized"] == "125" and f["ref_no"]["normalized"] == "2081/82"
    assert f["darta_no"]["normalized"] == "4567"
    assert f["date"]["normalized"] == {"bs": "2081-05-12", "ad": "2024-08-28", "raw": "२०८१/०५/१२"}


def test_bs_date_rejects_impossible():
    assert parse_bs_date("2081/13/40") is None
    assert parse_bs_date("२०८१ भदौ १२")["ad"] == "2024-08-28"


def test_script_detection():
    assert detect_script("नेपाल सरकार") == "ne" and detect_script("Remarks OK") == "en"
    assert detect_script("अनुरोध छ। Please submit the documents") == "mixed"


# ---- real-photo conditions (regressions from a phone photo of a citizenship card) -----

def test_blue_colour_cast_does_not_turn_print_into_ink():
    """Phone photos under LED light: paper ~ (210,224,240). Black print must not read as blue ink."""
    rgb = np.asarray(_letter()).astype(np.float32)
    cast = np.clip(rgb * np.array([0.84, 0.90, 0.97]) + np.array([0, 4, 12]), 0, 255).astype(np.uint8)
    found, masks = detect_ink(cast)
    kinds = sorted(b.type for b in found)
    assert kinds.count("emblem") == 1 and kinds.count("stamp") == 1
    assert len(found) <= 5, kinds  # emblem, stamp, signature, handwriting (+ at most one fragment)
    body = masks["any"][690:910, 100:1600]  # four lines of black body text
    assert body.mean() < 1.0


def test_skin_toned_photo_is_not_a_stamp():
    img = _letter().copy()
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([150, 1450, 450, 1800], fill=(205, 160, 135))          # ID photo: skin/shirt tones
    d.ellipse([220, 1500, 380, 1700], fill=(190, 120, 95))
    found, _ = detect_ink(np.asarray(img))
    photo_area = Box("x", 1, 150, 1450, 450, 1800, "t")
    assert not any(b.type == "stamp" and _overlaps(b, [150, 1450, 450, 1800]) for b in found), \
        [(b.type, b.as_int()) for b in found if _overlaps(b, [150, 1450, 450, 1800])]
    del photo_area


def test_red_pen_signature_is_not_a_stamp():
    img = _letter().copy()
    from PIL import ImageDraw
    import math
    d = ImageDraw.Draw(img)
    pts = [(200 + t * 4, 1600 + 90 * math.sin(t / 9) + t * 0.8) for t in range(120)]
    d.line(pts, fill=(200, 30, 40), width=6)
    d.line([(260, 1560), (300, 1820)], fill=(200, 30, 40), width=6)
    found, _ = detect_ink(np.asarray(img))
    sig = [b for b in found if _overlaps(b, [200, 1550, 680, 1830])]
    assert sig and all(b.type == "signature" for b in sig), [(b.type, b.attributes) for b in sig]



def test_citizenship_dob_split_across_blocks():
    comps = [{"id": "p1-c010", "page": 1, "text": "जन्म मिति:\nबाबुको नाम,थर:", "text_confidence": 0.9,
              "bbox": {"x0": 660, "y0": 890, "x1": 1070, "y1": 1050}},
             {"id": "p1-c011", "page": 1, "text": "साल: २०५१ महिना: ०८ गते १३\nदेवानन्द गिरी", "text_confidence": 0.85,
              "bbox": {"x0": 1140, "y0": 895, "x1": 1890, "y1": 1050}}]
    f = {x["key"]: x for x in extract_fields(comps)}
    assert f["date_of_birth"]["normalized"]["bs"] == "2051-08-13"
    assert f["date_of_birth"]["normalized"]["ad"] == "1994-11-29"
    assert "date" not in f  # "जन्म मिति" is not the document date
