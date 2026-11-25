"""Nepali text post-processing: normalisation, script detection, key-value fields, BS dates."""
from __future__ import annotations

import re
import unicodedata

try:
    import nepali_datetime  # BS ↔ AD conversion
except ImportError:  # optional; dates are still parsed, just not converted
    nepali_datetime = None

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZW = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def clean_text(text: str) -> str:
    """Unicode-correct OCR text: NFC (so नुक्ता forms compare equal), no zero-width junk,
    tidy whitespace. Keeps the original numerals."""
    text = unicodedata.normalize("NFC", text or "").translate(_ZW).replace(" ", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def normalize_for_search(text: str) -> str:
    """ASCII digits, lowercase Latin, single spaces — for indexing and matching."""
    return re.sub(r"\s+", " ", clean_text(text).translate(DEVANAGARI_DIGITS)).strip().lower()


def detect_script(text: str) -> str | None:
    deva = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if not deva and not latin:
        return None
    if deva and latin and min(deva, latin) / max(deva, latin) > 0.15:
        return "mixed"
    return "ne" if deva > latin else "en"


# ---- Bikram Sambat dates ------------------------------------------------------------

BS_MONTHS = {
    "बैशाख": 1, "वैशाख": 1, "जेठ": 2, "जेष्ठ": 2, "असार": 3, "आषाढ": 3, "साउन": 4, "श्रावण": 4,
    "भदौ": 5, "भाद्र": 5, "असोज": 6, "आश्विन": 6, "कात्तिक": 7, "कार्तिक": 7, "मंसिर": 8, "मङ्सिर": 8,
    "पुस": 9, "पौष": 9, "माघ": 10, "फागुन": 11, "फाल्गुन": 11, "चैत": 12, "चैत्र": 12,
}


def parse_bs_date(raw: str) -> dict | None:
    """'२०८१/०५/१२', '2081-5-12', '2081.05.12', '२०८१ भदौ १२' → {'bs','ad','raw'}."""
    s = raw.translate(DEVANAGARI_DIGITS)
    y = m = d = None
    if mt := re.search(r"साल\D{0,4}(\d{4}).{0,15}?महिना\D{0,4}(\d{1,2}).{0,15}?गते\D{0,4}(\d{1,2})", s):
        y, m, d = map(int, mt.groups())                       # citizenship style: साल २०५१ महिना ०८ गते १३
    elif mt := re.search(r"(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})", s):
        y, m, d = map(int, mt.groups())
    elif mt := re.search(r"(\d{4})\s+(\S+)\s+(\d{1,2})", s):
        month = BS_MONTHS.get(mt.group(2))
        if month:
            y, m, d = int(mt.group(1)), month, int(mt.group(3))
    if not y or not (1970 <= y <= 2150 and 1 <= m <= 12 and 1 <= d <= 32):
        return None
    ad = None
    if nepali_datetime is not None:
        try:
            ad = nepali_datetime.date(y, m, d).to_datetime_date().isoformat()
        except (ValueError, OverflowError, IndexError, KeyError):
            return None  # e.g. 32 Baisakh in a 31-day year
    return {"bs": f"{y:04d}-{m:02d}-{d:02d}", "ad": ad, "raw": raw.strip()}


# ---- key-value fields ----------------------------------------------------------------

_SEP = r"\s*[:：\-–.।]?\s*"
# (key, English label, label patterns, value pattern)
FIELD_RULES: list[tuple[str, str, list[str], str]] = [
    ("chalani_no", "Chalani No.",
     [r"च\.?\s*नं\.?", r"चलानी\s*(?:नं|नम्बर)\.?", r"chalani\s*no\.?", r"dispatch\s*no\.?"],
     r"([0-9०-९][0-9०-९/\-\s]{0,15}[0-9०-९]|[0-9०-९])"),
    ("darta_no", "Darta No.",
     [r"द\.?\s*नं\.?", r"दर्ता\s*(?:नं|नम्बर)\.?", r"darta\s*no\.?", r"registration\s*no\.?", r"reg\.?\s*no\.?"],
     r"([0-9०-९][0-9०-९/\-\s]{0,15}[0-9०-९]|[0-9०-९])"),
    ("ref_no", "Letter ref. (Patra Sankhya)",
     [r"प\.?\s*सं\.?", r"पत्र\s*संख्या", r"ref(?:erence)?\.?\s*no\.?"],
     r"([0-9०-९A-Za-z][0-9०-९A-Za-z/\-.\s]{0,20}[0-9०-९A-Za-z])"),
    ("fiscal_year", "Fiscal year",
     [r"आ\.?\s*व\.?", r"आर्थिक\s*वर्ष", r"fiscal\s*year", r"f\.?\s*y\.?"],
     r"([0-9०-९]{4}\s*[/\-]\s*[0-9०-९]{2,4})"),
    ("citizenship_no", "Citizenship No.",
     [r"ना\.?\s*प्र\.?\s*नं\.?", r"नागरिकता\s*(?:प्रमाणपत्र)?\s*नं\.?", r"citizenship\s*(?:certificate)?\s*no\.?"],
     r"([0-9०-९][0-9०-९/\-\s]{2,24}[0-9०-९])"),
    ("date_of_birth", "Date of birth",
     [r"जन्म\s*मिति", r"date\s*of\s*birth"],
     r"(साल\S*\s*[0-9०-९]{4}.{0,15}?महिना\S*\s*[0-9०-९]{1,2}.{0,15}?गते\S*\s*[0-9०-९]{1,2}"
     r"|[0-9०-९]{4}\s*[/\-.]\s*[0-9०-९]{1,2}\s*[/\-.]\s*[0-9०-९]{1,2})"),
    ("date", "Date (Miti)",
     [r"(?<!जन्म )(?<!जन्म)मिति", r"(?<!of )date(?!\s*of\s*birth)"],
     r"([0-9०-९]{4}\s*[/\-.]\s*[0-9०-९]{1,2}\s*[/\-.]\s*[0-9०-९]{1,2}|[0-9०-९]{4}\s+\S+\s+[0-9०-९]{1,2})"),
]


def _compile(label: str) -> str:
    return rf"(?<![A-Za-z]){label}" if label[:1].isascii() else label


_COMPILED = [(key, en, [re.compile(_compile(lbl) + _SEP + value, re.IGNORECASE) for lbl in labels])
             for key, en, labels, value in FIELD_RULES]


def visual_rows(components: list[dict]) -> list[dict]:
    """Rebuild printed lines across layout blocks.

    Layout models often put a form's label column and value column in separate
    blocks ("जन्म मिति:" | "साल २०५१ महिना ०८ गते १३"). We split each block into its
    text lines, estimate each line's vertical centre, and join lines from different
    blocks that sit on the same row, left to right.
    """
    lines = []
    for c in components:
        if not c.get("text") or not c.get("bbox"):
            continue
        parts = [ln for ln in c["text"].split("\n") if ln.strip()]
        b = c["bbox"]
        step = (b["y1"] - b["y0"]) / max(len(parts), 1)
        for i, ln in enumerate(parts):
            lines.append({"page": c.get("page", 1), "y": b["y0"] + (i + 0.5) * step, "h": step,
                          "x": b["x0"], "text": ln, "id": c["id"], "conf": c.get("text_confidence")})
    lines.sort(key=lambda ln: (ln["page"], ln["y"]))
    rows: list[list[dict]] = []
    for ln in lines:
        row = rows[-1] if rows else None
        if row and row[0]["page"] == ln["page"] and abs(ln["y"] - row[0]["y"]) < 0.5 * min(ln["h"], row[0]["h"]):
            row.append(ln)
        else:
            rows.append([ln])
    out = []
    for row in rows:
        row.sort(key=lambda ln: ln["x"])
        if len({ln["id"] for ln in row}) < 2:
            continue  # single-block rows are already covered by the block itself
        confs = [ln["conf"] for ln in row if ln["conf"] is not None]
        out.append({"id": row[-1]["id"], "page": row[0]["page"], "text": " ".join(ln["text"] for ln in row),
                    "text_confidence": min(confs) if confs else None})
    return out


def extract_fields(components: list[dict]) -> list[dict]:
    """Find labelled values (Chalani/Darta/Ref no., BS dates…) in reading order.

    Searches each block, then rebuilt visual rows (label and value in different
    blocks). Each hit keeps the component it came from so the UI can highlight it.
    """
    components = list(components) + visual_rows(components)
    fields: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for comp in components:
        text = comp.get("text") or ""
        if not text:
            continue
        for key, en, patterns in _COMPILED:
            for rx in patterns:
                for m in rx.finditer(text):
                    is_date = key in ("date", "date_of_birth")
                    raw = m.group(1).strip() if is_date else re.sub(r"\s+", "", m.group(1))
                    normalized: object = raw.translate(DEVANAGARI_DIGITS)
                    if is_date:
                        normalized = parse_bs_date(raw)
                        if normalized is None:
                            continue
                    if (key, str(normalized)) in seen:
                        continue
                    seen.add((key, str(normalized)))
                    fields.append({
                        "key": key, "label": en, "value": raw, "normalized": normalized,
                        "component_id": comp["id"], "page": comp.get("page", 1),
                        "confidence": comp.get("text_confidence"),
                    })
                if any(f["key"] == key and f["component_id"] == comp["id"] for f in fields):
                    break  # first matching label variant wins for this component
    return fields
