"""Correction workflow: edit / add / delete components, re-OCR, review, history."""
import time

import pytest
from fastapi.testclient import TestClient

from backend.main import app

AUTH = {"Authorization": "Bearer test-token"}


class FakeOcr:
    name = "fake-ocr"

    def recognize(self, image, regions):
        return {cid: {"text": "च.नं. ७७७", "confidence": 0.88, "html": None} for cid, _ in regions}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1") as c:
        end = time.time() + 120
        while c.get("/api/health").json()["models"] != "ready":
            assert time.time() < end
            time.sleep(0.1)
        app.state.worker.extractor.ocr = FakeOcr()
        yield c
        app.state.worker.extractor.ocr = None


@pytest.fixture()
def doc(client, letter_png):
    r = client.post("/api/documents", headers=AUTH, files={"file": ("l.png", letter_png, "image/png")})
    doc_id = r.json()["id"]
    end = time.time() + 120
    while (d := client.get(f"/api/documents/{doc_id}", headers=AUTH).json())["status"] not in ("done", "failed"):
        assert time.time() < end
        time.sleep(0.1)
    assert d["status"] == "done", d
    return d


def _comp(rec, type_):
    return next(c for c in rec["components"] if c["type"] == type_)


def test_edit_text_updates_fields_and_history(client, doc):
    c = next(x for x in doc["record"]["components"] if x["type"] == "text")
    r = client.patch(f"/api/documents/{doc['id']}/components/{c['id']}", headers=AUTH,
                     json={"text": "दर्ता नं. ४५६७"})
    assert r.status_code == 200, r.text
    rec = r.json()["record"]
    edited = next(x for x in rec["components"] if x["id"] == c["id"])
    assert edited["text"] == "दर्ता नं. ४५६७" and edited["text_source"] == "manual" and edited["text_confidence"] == 1.0
    assert any(f["key"] == "darta_no" and f["normalized"] == "4567" for f in rec["fields"])
    assert rec["corrections"][-1]["action"] == "update" and rec["corrections"][-1]["before"]["text"] == c["text"]
    assert rec["review"]["edited"] is True
    # persisted, and searchable through the components table
    assert client.get("/api/components?text=4567", headers=AUTH).json()


def test_change_type_creates_or_removes_crop(client, doc, data_dir):
    t = next(x for x in doc["record"]["components"] if x["type"] == "text")
    rec = client.patch(f"/api/documents/{doc['id']}/components/{t['id']}", headers=AUTH,
                       json={"type": "signature"}).json()["record"]
    sig = next(x for x in rec["components"] if x["id"] == t["id"])
    assert sig["layer"] == "visual" and sig["asset"]["transparent_png"]
    assert (data_dir / sig["asset"]["png"]).is_file()
    rec = client.patch(f"/api/documents/{doc['id']}/components/{t['id']}", headers=AUTH,
                       json={"type": "text"}).json()["record"]
    assert next(x for x in rec["components"] if x["id"] == t["id"])["asset"] is None
    assert not (data_dir / sig["asset"]["png"]).exists()


def test_resize_box_recrops_and_reocr(client, doc, data_dir):
    from PIL import Image

    stamp = _comp(doc["record"], "stamp")
    b = stamp["bbox"]
    new = {"x0": b["x0"] - 20, "y0": b["y0"] - 20, "x1": b["x1"] + 20, "y1": b["y1"] + 20}
    rec = client.patch(f"/api/documents/{doc['id']}/components/{stamp['id']}", headers=AUTH,
                       json={"bbox": new}).json()["record"]
    s2 = next(x for x in rec["components"] if x["id"] == stamp["id"])
    assert s2["bbox"]["width"] == b["width"] + 40
    assert Image.open(data_dir / s2["asset"]["png"]).size == (s2["bbox"]["width"], s2["bbox"]["height"])
    assert "?v=" in s2["asset"]["png_url"]  # cache-busting after a re-crop

    t = next(x for x in rec["components"] if x["type"] == "text")
    rec = client.patch(f"/api/documents/{doc['id']}/components/{t['id']}", headers=AUTH,
                       json={"reocr": True}).json()["record"]
    assert next(x for x in rec["components"] if x["id"] == t["id"])["text"] == "च.नं. ७७७"


def test_add_missing_component_with_ocr_then_delete(client, doc, data_dir):
    page = doc["record"]["pages"][0]
    r = client.post(f"/api/documents/{doc['id']}/components", headers=AUTH,
                    json={"page": 1, "type": "text", "bbox": {"x0": 100, "y0": 2050, "x1": 900, "y1": 2120}})
    assert r.status_code == 201, r.text
    out = r.json()
    cid = out["new_component_id"]
    rec = out["record"]
    new = next(x for x in rec["components"] if x["id"] == cid)
    assert new["text"] == "च.नं. ७७७" and new["attributes"]["added_by_user"]
    assert new["reading_order"] == len(rec["components"])  # bottom of the page → last
    assert new["bbox_norm"]["x1"] == round(900 / page["width"], 5)
    assert any(f["key"] == "chalani_no" and f["normalized"] == "777" for f in rec["fields"])

    sig = client.post(f"/api/documents/{doc['id']}/components", headers=AUTH,
                      json={"page": 1, "type": "signature", "bbox": {"x0": 300, "y0": 1500, "x1": 600, "y1": 1600}}).json()
    sid = sig["new_component_id"]
    png = next(x for x in sig["record"]["components"] if x["id"] == sid)["asset"]["png"]
    assert (data_dir / png).is_file()

    rec = client.delete(f"/api/documents/{doc['id']}/components/{sid}", headers=AUTH).json()["record"]
    assert all(x["id"] != sid for x in rec["components"]) and not (data_dir / png).exists()
    assert rec["corrections"][-1]["action"] == "delete"


def test_review_flag(client, doc):
    rec = client.post(f"/api/documents/{doc['id']}/review", headers=AUTH, json={"verified": True}).json()["record"]
    assert rec["review"]["verified"] is True and rec["review"]["verified_at"]


@pytest.mark.parametrize("body,code", [
    ({"type": "dragon"}, 422),
    ({"bbox": {"x0": 10, "y0": 10, "x1": 12, "y1": 12}}, 422),
    ({}, 400),
])
def test_invalid_edits_rejected(client, doc, body, code):
    c = doc["record"]["components"][0]
    assert client.patch(f"/api/documents/{doc['id']}/components/{c['id']}", headers=AUTH, json=body).status_code == code


def test_edits_require_auth_and_valid_ids(client, doc):
    c = doc["record"]["components"][0]
    assert client.patch(f"/api/documents/{doc['id']}/components/{c['id']}", json={"text": "x"}).status_code == 401
    assert client.patch(f"/api/documents/{doc['id']}/components/p9-c999", headers=AUTH, json={"text": "x"}).status_code == 422
    assert client.delete(f"/api/documents/{'f' * 32}/components/p1-c001", headers=AUTH).status_code == 404
