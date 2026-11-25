"""Security helpers: upload validation, safe decoding, local auth, path safety."""
from __future__ import annotations

import hmac
import io
import os
import re
import secrets
import stat
import tempfile
import warnings
from pathlib import Path

from fastapi import HTTPException, Request, status
from PIL import Image, ImageOps

from .config import Settings

# --- uploads ------------------------------------------------------------------------

_MAGIC: list[tuple[bytes, int, str]] = [
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"WEBP", 8, "image/webp"),
    (b"II*\x00", 0, "image/tiff"),
    (b"MM\x00*", 0, "image/tiff"),
    (b"%PDF-", 0, "application/pdf"),
]
# iPhone/Android HEIC & HEIF: ISO-BMFF box "ftyp" + brand at offset 4.
_HEIF_BRANDS = (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypheim", b"ftypheis", b"ftypmif1", b"ftypmsf1")
# Pillow reports phone JPEGs that carry extra frames (depth map / portrait data) as
# "MPO" — they are still JPEGs and must be accepted.
_PIL_FORMATS = {"JPEG": "image/jpeg", "MPO": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
                "TIFF": "image/tiff", "HEIF": "image/heic", "AVIF": "image/heic"}

try:  # optional: HEIC support (pip install pillow-heif)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False


def sniff(head: bytes) -> str | None:
    for sig, off, ctype in _MAGIC:
        if head[off:off + len(sig)] == sig:
            return ctype
    if head[4:12] in _HEIF_BRANDS:
        return "image/heic"
    return None


def _bad(msg: str, code: int = status.HTTP_400_BAD_REQUEST) -> HTTPException:
    return HTTPException(code, msg)


def decode_upload(data: bytes, settings: Settings) -> tuple[list[Image.Image], str]:
    """Bytes → list of RGB pages. Rejects disguised files, decompression bombs and
    oversized PDFs *before* allocating full-size pixel buffers."""
    if not data:
        raise _bad("Empty file")
    ctype = sniff(data[:16])
    if ctype is None:
        raise _bad("Unsupported file. Upload JPEG, PNG, WEBP, TIFF, HEIC or PDF.", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    if ctype == "image/heic" and not HEIF_SUPPORTED:
        raise _bad("This is an iPhone HEIC photo. Install HEIC support (pip install pillow-heif) "
                   "or export it as JPEG.", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    if ctype == "application/pdf":
        return _decode_pdf(data, settings), ctype

    Image.MAX_IMAGE_PIXELS = settings.max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))          # lazy: reads header only
            if _PIL_FORMATS.get(img.format) != ctype:     # header says one thing, magic another
                raise _bad(f"File content ({img.format}) does not match its type ({ctype})",
                           status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            w, h = img.size
            if w * h > settings.max_pixels or w < 32 or h < 32:
                raise _bad(f"Image is {w}×{h}; allowed up to {settings.max_pixels // 1_000_000} MP",
                           status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
            # Multi-frame TIFFs are documents (pages); MPO/HEIC extra frames are depth maps
            # or thumbnails — only the first frame is the photo.
            frames = getattr(img, "n_frames", 1) if ctype == "image/tiff" else 1
            pages = []
            for i in range(min(frames, settings.max_pdf_pages)):   # multi-page TIFF
                img.seek(i)
                pages.append(ImageOps.exif_transpose(img).convert("RGB"))  # EXIF dropped on re-encode
            return pages, ctype
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise _bad("Image dimensions are too large (decompression bomb protection)",
                   status.HTTP_413_REQUEST_ENTITY_TOO_LARGE) from None
    except Exception as exc:  # noqa: BLE001 - malformed/truncated images
        raise _bad(f"Could not read image: {type(exc).__name__}") from None


def _decode_pdf(data: bytes, settings: Settings) -> list[Image.Image]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception:  # noqa: BLE001
        raise _bad("Could not read PDF") from None
    try:
        n = len(pdf)
        if n == 0:
            raise _bad("PDF has no pages")
        pages = []
        scale = settings.pdf_dpi / 72
        for i in range(min(n, settings.max_pdf_pages)):
            page = pdf[i]
            w, h = page.get_size()
            if (w * scale) * (h * scale) > settings.max_pixels:   # check before rendering
                scale = (settings.max_pixels / (w * h)) ** 0.5
            pages.append(page.render(scale=scale).to_pil().convert("RGB"))
        return pages
    finally:
        pdf.close()


# --- local API auth -----------------------------------------------------------------

SESSION_COOKIE = "nepdoc_session"


def load_or_create_token(settings: Settings) -> str:
    """Token from NEPDOC_API_TOKEN, else a random one persisted to data/.api_token (0600).

    The Vite dev proxy reads that file and injects the header, so the browser never
    needs to know it, but other local processes and web pages can't call the API.
    """
    env = os.getenv("NEPDOC_API_TOKEN")
    if env:
        return env
    path = settings.token_file
    if path.exists():
        return path.read_text().strip()
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    return token


def _host_allowed(request: Request, settings: Settings) -> bool:
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
    allowed = {"localhost", "127.0.0.1", "::1", *[h.lower() for h in settings.allowed_hosts]}
    return host in allowed


def require_auth(request: Request, token: str, settings: Settings) -> None:
    # DNS rebinding: a malicious site can resolve its own name to 127.0.0.1, but the
    # browser still sends *its* Host header — so we only answer to loopback names.
    if not _host_allowed(request, settings):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Host not allowed")
    supplied = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    elif request.cookies.get(SESSION_COOKIE):
        supplied = request.cookies[SESSION_COOKIE]
        # Cookie auth is browser-only: block cross-site use of state-changing verbs.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin") or request.headers.get("referer") or ""
            if origin and not re.match(rf"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?(/|$)", origin):
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Cross-site request blocked")
    if not supplied or not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API token")


# --- filesystem ---------------------------------------------------------------------

UUID_RE = re.compile(r"^[0-9a-f]{32}$")
ASSET_RE = re.compile(r"^p\d{1,3}-c\d{3}(-alpha)?\.png$|^page-\d{1,3}\.png$")


def safe_child(root: Path, *parts: str) -> Path:
    """Join and resolve; refuse anything that escapes `root` (../, symlinks, absolute)."""
    root = root.resolve()
    target = root.joinpath(*parts).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return target


def prepare_storage(settings: Settings) -> None:
    """Private data dirs (0700) and a private temp dir for Starlette upload spooling."""
    for d in (settings.data_dir, settings.assets_dir, settings.pages_dir, settings.tmp_dir):
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, stat.S_IRWXU)
        except OSError:
            pass
    # Uploads >1 MB are spooled to disk by Starlette; keep them in our private dir
    # instead of the shared system /tmp, and clear leftovers from crashed runs.
    tempfile.tempdir = str(settings.tmp_dir)
    for leftover in settings.tmp_dir.iterdir():
        try:
            leftover.unlink()
        except OSError:
            pass


def sanitize_filename(name: str | None) -> str:
    """Display-only name (never used to build paths)."""
    name = os.path.basename(name or "document")
    name = re.sub(r"[^\w\-. ऀ-ॿ]", "_", name)[:120].strip(" .")
    return name or "document"
