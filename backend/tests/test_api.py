"""API flow + security controls."""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1") as c:
        end = time.time() + 120  # generous: model loading is slow on a busy machine
        while c.get("/api/health").json()["models"] != "ready":
            assert time.time() < end, "worker never became ready"
            time.sleep(0.1)
        yield c


def _wait(client, doc_id, timeout=120):
    end, d = time.time() + timeout, {}
    while time.time() < end:
        d = client.get(f"/api/documents/{doc_id}", headers=AUTH).json()
        if d["status"] in ("done", "failed"):
            return d
        time.sleep(0.1)
    raise AssertionError(f"timeout: status={d.get('status')} stage={d.get('stage')}")


def _png(w=400, h=300, color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def _upload(client, data, name="doc.png", headers=AUTH, **form):
    return client.post("/api/documents", headers=headers, files={"file": (name, data, "image/png")}, data=form)


# ---- happy path ------------------------------------------------------------------------

def test_upload_process_and_fetch(client, letter_png, data_dir):
    r = _upload(client, letter_png, "letter.png")
    assert r.status_code == 202, r.text
    doc = _wait(client, r.json()["id"])
    assert doc["status"] == "done", doc
    rec = doc["record"]
    types = {c["type"] for c in rec["components"]}
    assert {"emblem", "stamp", "signature", "table"} <= types
    stamp = next(c for c in rec["components"] if c["type"] == "stamp")
    img = client.get(stamp["asset"]["transparent_url"], headers=AUTH)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert img.headers["cache-control"].startswith("private")
    assert client.get(rec["pages"][0]["image_url"], headers=AUTH).status_code == 200

    listing = client.get("/api/documents", headers=AUTH).json()
    assert listing[0]["id"] == doc["id"] and listing[0]["by_type"]["stamp"] == 1
    stamps = client.get("/api/components?type=stamp", headers=AUTH).json()
    assert any(s["document_id"] == doc["id"] for s in stamps)   # other test modules share the DB
    dl = client.get(f"/api/documents/{doc['id']}/record.json", headers=AUTH)
    assert "attachment" in dl.headers["content-disposition"] and dl.json()["document_id"] == doc["id"]

    assert client.delete(f"/api/documents/{doc['id']}", headers=AUTH).status_code == 204
    assert not (data_dir / "assets" / doc["id"]).exists() and not (data_dir / "pages" / doc["id"]).exists()


def test_options_are_respected(client, letter_png):
    doc = _wait(client, _upload(client, letter_png, transparent="false").json()["id"])
    assert all(c["asset"] is None or c["asset"]["transparent_png"] is None for c in doc["record"]["components"])


def test_multipage_pdf(client):
    buf = io.BytesIO()
    pages = [Image.new("RGB", (800, 1100), "white"), Image.new("RGB", (800, 1100), (250, 250, 250))]
    pages[0].save(buf, "PDF", save_all=True, append_images=pages[1:])
    r = client.post("/api/documents", headers=AUTH, files={"file": ("a.pdf", buf.getvalue(), "application/pdf")})
    assert r.status_code == 202 and r.json()["pages"] == 2
    assert _wait(client, r.json()["id"])["record"]["page_count"] == 2


def test_phone_jpeg_with_extra_frames_is_accepted(client):
    """iPhone/Samsung JPEGs with depth/portrait data are 'MPO' to Pillow — still JPEG."""
    a, b = Image.new("RGB", (600, 400), "white"), Image.new("RGB", (300, 200), "gray")
    buf = io.BytesIO()
    a.save(buf, "MPO", save_all=True, append_images=[b])
    r = _upload(client, buf.getvalue(), "IMG_2041.jpeg")
    assert r.status_code == 202, r.text
    assert r.json()["pages"] == 1  # the depth frame is not treated as a second page


def test_heic_photo_is_accepted(client):
    pytest.importorskip("pillow_heif")
    buf = io.BytesIO()
    Image.new("RGB", (600, 400), "white").save(buf, "HEIF")
    r = _upload(client, buf.getvalue(), "IMG_2042.HEIC")
    assert r.status_code == 202, r.text
    assert _wait(client, r.json()["id"])["status"] == "done"


# ---- auth & network exposure --------------------------------------------------------------

def test_requires_token(client):
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_dns_rebinding_host_rejected(client):
    r = client.get("/api/documents", headers={**AUTH, "Host": "evil.example.com"})
    assert r.status_code == 403


def test_cookie_session_blocks_cross_site_writes(client, letter_png):
    client.cookies.set("nepdoc_session", "test-token", path="/api")
    try:
        assert client.get("/api/documents").status_code == 200
        r = client.post("/api/documents", headers={"Origin": "https://evil.example.com"},
                        files={"file": ("x.png", letter_png, "image/png")})
        assert r.status_code == 403
    finally:
        client.cookies.clear()


def test_cors_closed_by_default(client):
    r = client.options("/api/documents", headers={"Origin": "https://evil.example.com",
                                                  "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


# ---- uploads ----------------------------------------------------------------------------

def test_rejects_disguised_file(client):
    assert _upload(client, b"MZ\x90\x00 this is an exe", "invoice.png").status_code == 415


def test_rejects_decompression_bomb(client):
    bomb = _png(8000, 8000)  # 64 MP of white compresses to a few KB
    assert len(bomb) < 1_000_000
    r = _upload(client, bomb, "bomb.png")
    assert r.status_code == 413


def test_rejects_oversized_body_before_parsing(client):
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (6 * 1024 * 1024)
    assert _upload(client, big).status_code == 413


def test_rejects_truncated_image(client):
    assert _upload(client, _png()[:60]).status_code == 400


def test_filename_never_used_for_paths(client, letter_png):
    r = _upload(client, letter_png, "../../../etc/passwd.png")
    doc = _wait(client, r.json()["id"])
    assert doc["filename"] == "passwd.png"


# ---- path traversal on assets ---------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/api/assets/" + "a" * 32 + "/..%2F..%2F.api_token",
    "/api/assets/" + "a" * 32 + "/../../nepdoc.db",
    "/api/assets/..%2F..%2Fetc/passwd",
    "/api/assets/" + "a" * 32 + "/p1-c001.png%00.txt",
    "/api/assets/not-a-uuid/p1-c001.png",
    "/api/documents/..%2F..%2Fnepdoc",
])
def test_path_traversal_blocked(client, path):
    assert client.get(path, headers=AUTH).status_code == 404


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"
