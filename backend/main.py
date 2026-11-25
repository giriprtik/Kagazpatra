"""FastAPI app: upload → background extraction → JSON record + assets.

Run (from the project root):
    uvicorn backend.main:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import hashlib
import logging
import queue
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .config import ROOT, get_settings
from . import corrections, llm_review
from .database import Database
from .extractor import Extractor, Options
from .security import (ASSET_RE, SESSION_COOKIE, UUID_RE, decode_upload, load_or_create_token, prepare_storage,
                       require_auth, safe_child, sanitize_filename)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("nepdoc")
settings = get_settings()


class Worker:
    """Single consumer thread: the models are loaded once and are not thread-safe,
    and one document at a time keeps memory predictable on a 16 GB Mac."""

    def __init__(self, db: Database):
        self.db = db
        self.jobs: queue.Queue = queue.Queue(maxsize=settings.max_queue)
        self.extractor: Extractor | None = None
        self.state, self.error = "loading", None
        self.thread = threading.Thread(target=self._run, name="extractor", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            self.extractor = Extractor(settings)
            self.state = "ready"
            log.info("Models ready: %s", self.extractor.engines())
        except Exception as exc:  # noqa: BLE001
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            log.exception("Model loading failed")
            return
        while True:
            job = self.jobs.get()
            if job is None:
                break
            if job[0] == "__call__":            # serialized edit / re-OCR request
                _, fn, fut = job
                try:
                    fut.set_result(fn(self.extractor))
                except Exception as exc:  # noqa: BLE001
                    fut.set_exception(exc)
                finally:
                    self.jobs.task_done()
                continue
            doc_id, pages, source, opts = job
            try:
                self.db.set_stage(doc_id, "starting")
                record = self.extractor.process(doc_id, pages, source, opts,
                                                on_stage=lambda s, d=doc_id: self.db.set_stage(d, s))
                self.db.save_result(doc_id, record)
                log.info("Document %s done: %d components", doc_id, len(record["components"]))
            except Exception as exc:  # noqa: BLE001
                log.exception("Extraction failed for %s", doc_id)
                self.db.fail(doc_id, f"{type(exc).__name__}: {exc}")
            finally:
                del pages
                self.jobs.task_done()

    def call(self, fn, timeout: float = 600):
        """Run fn(extractor) on the worker thread (after any running job) and wait."""
        if self.state != "ready":
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Models are not ready")
        if not self.thread.is_alive():
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Worker is not running")
        fut: Future = Future()
        try:
            self.jobs.put(("__call__", fn, fut), timeout=5)
        except queue.Full:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Busy — try again shortly") from None
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Timed out waiting for the worker") from None

    def stop(self) -> None:
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        if self.extractor:
            self.extractor.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    prepare_storage(settings)
    app.state.token = load_or_create_token(settings)
    db = Database(settings)
    db.init()
    if (n := db.purge_older_than(settings.retention_days)):
        log.info("Retention: deleted %d old documents", n)
    if (stale := db.requeue_interrupted()):
        log.info("Marked %d interrupted jobs as failed", len(stale))
    app.state.db = db
    app.state.llm_client = None   # tests inject a fake; None = real Ollama client per call
    app.state.worker = Worker(db)   # fresh worker per app start (a stopped thread can't be reused)
    app.state.worker.start()
    log.info("API token in %s — the frontend dev proxy injects it automatically", settings.token_file)
    yield
    app.state.worker.stop()
    app.state.worker.thread.join(timeout=10)


app = FastAPI(title="NepDoc Local", version="1.0.0", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)

if settings.cors_origins:  # closed unless explicitly configured
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def guard(request: Request, call_next):
    # Reject oversized bodies before Starlette spools the multipart upload to disk.
    if request.method == "POST":
        length = request.headers.get("content-length")
        limit = settings.max_upload_mb * 1024 * 1024 + 64 * 1024
        if length is None and request.url.path.startswith("/api/documents"):
            return JSONResponse({"detail": "Content-Length required"}, status.HTTP_411_LENGTH_REQUIRED)
        if length and int(length) > limit:
            return JSONResponse({"detail": f"File larger than {settings.max_upload_mb} MB"},
                                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    else:
        response.headers.setdefault("Content-Security-Policy",
                                    "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
                                    "font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
    return response


def auth(request: Request) -> None:
    require_auth(request, request.app.state.token, settings)


def _summary(doc) -> dict:
    rec = doc.record or {}
    stats = rec.get("stats", {})
    return {"id": doc.id, "filename": doc.filename, "status": doc.status, "stage": doc.stage, "error": doc.error,
            "page_count": doc.page_count, "created_at": doc.created_at.isoformat(),
            "components": stats.get("components"), "by_type": stats.get("by_type"),
            "thumbnail": f"/api/assets/{doc.id}/page-1.png" if doc.status == "done" else None}


def _with_urls(record: dict) -> dict:
    """Add browser URLs next to the stored relative asset paths (copy; DB record untouched).
    `?v=` changes after each correction so re-cropped images aren't served stale."""
    import copy

    record = copy.deepcopy(record)
    doc_id = record["document_id"]
    v = len(record.get("corrections", []))
    for p in record.get("pages", []):
        p["image_url"] = f"/api/assets/{doc_id}/{Path(p['image_path']).name}"
    for c in record.get("components", []):
        if c.get("asset"):
            c["asset"]["png_url"] = f"/api/assets/{doc_id}/{Path(c['asset']['png']).name}?v={v}"
            if c["asset"].get("transparent_png"):
                c["asset"]["transparent_url"] = (
                    f"/api/assets/{doc_id}/{Path(c['asset']['transparent_png']).name}?v={v}")
    return record


def _doc_or_404(request: Request, doc_id: str):
    if not UUID_RE.match(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    doc = request.app.state.db.get(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return doc


# ---- endpoints -------------------------------------------------------------------------

@app.get("/api/health")
def health(request: Request):
    w = request.app.state.worker
    return {"status": "ok", "models": w.state}


@app.get("/api/engines", dependencies=[Depends(auth)])
def engines(request: Request):
    w = request.app.state.worker
    ex = w.extractor
    return {"state": w.state, "error": w.error, "engines": ex.engines() if ex else None,
            "warnings": ex.warnings if ex else [], "queue": w.jobs.qsize(),
            "limits": {"max_upload_mb": settings.max_upload_mb, "max_pdf_pages": settings.max_pdf_pages}}


@app.post("/api/documents", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(auth)])
async def upload(request: Request, file: UploadFile = File(...),
                 transparent: bool = Form(True), suppress_stamp_ink: bool = Form(True),
                 ocr_stamps: bool = Form(False), ocr_tables: bool = Form(True)):
    worker: Worker = request.app.state.worker
    if worker.state == "error":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Models failed to load: {worker.error}")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    await file.close()  # removes the spooled temp file immediately
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File larger than {settings.max_upload_mb} MB")
    pages, ctype = decode_upload(data, settings)
    opts = Options(transparent=transparent, suppress_stamp_ink=suppress_stamp_ink,
                   ocr_stamps=ocr_stamps, ocr_tables=ocr_tables)
    source = {"filename": sanitize_filename(file.filename), "content_type": ctype, "size_bytes": len(data),
              "sha256": hashlib.sha256(data).hexdigest()}
    db: Database = request.app.state.db
    doc = db.create(filename=source["filename"], content_type=ctype, sha256=source["sha256"],
                    size_bytes=len(data), options=opts.__dict__)
    try:
        worker.jobs.put_nowait((doc.id, pages, source, opts))
    except queue.Full:
        db.fail(doc.id, "Queue full")
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Busy — try again shortly") from None
    return {"id": doc.id, "status": "queued", "pages": len(pages)}


@app.get("/api/documents", dependencies=[Depends(auth)])
def list_documents(request: Request, q: str | None = Query(None, max_length=100),
                   limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return [_summary(d) for d in request.app.state.db.list(q, limit, offset)]


@app.get("/api/documents/{doc_id}", dependencies=[Depends(auth)])
def get_document(request: Request, doc_id: str):
    doc = _doc_or_404(request, doc_id)
    out = _summary(doc)
    out["record"] = _with_urls(doc.record) if doc.record else None
    return out


@app.get("/api/documents/{doc_id}/record.json", dependencies=[Depends(auth)])
def download_record(request: Request, doc_id: str):
    doc = _doc_or_404(request, doc_id)
    if not doc.record:
        raise HTTPException(status.HTTP_409_CONFLICT, "Not processed yet")
    return JSONResponse(doc.record, headers={"Content-Disposition": f'attachment; filename="{doc_id}.json"'})


@app.delete("/api/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(auth)])
def delete_document(request: Request, doc_id: str):
    _doc_or_404(request, doc_id)
    request.app.state.db.delete(doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/components", dependencies=[Depends(auth)])
def query_components(request: Request, type: str | None = Query(None, max_length=24),
                     text: str | None = Query(None, max_length=100)):
    rows = request.app.state.db.find_components(type, text)
    return [{"document_id": r.document_id, "component_id": r.component_id, "page": r.page, "type": r.type,
             "confidence": r.confidence, "bbox": [r.x0, r.y0, r.x1, r.y1], "text": r.text} for r in rows]


@app.get("/api/assets/{doc_id}/{name}", dependencies=[Depends(auth)])
def asset(request: Request, doc_id: str, name: str):
    """Authenticated asset serving (crops contain PII, so no public static mount)."""
    if not UUID_RE.match(doc_id) or not ASSET_RE.match(name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    base = settings.pages_dir if name.startswith("page-") else settings.assets_dir
    path = safe_child(base, doc_id, name)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


# ---- corrections ------------------------------------------------------------------------

class BBoxIn(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ComponentPatch(BaseModel):
    type: str | None = Field(default=None, max_length=24)
    text: str | None = Field(default=None, max_length=20000)
    bbox: BBoxIn | None = None
    reocr: bool = False


class ComponentNew(BaseModel):
    page: int = Field(ge=1, le=500)
    bbox: BBoxIn
    type: str = Field(max_length=24)
    text: str | None = Field(default=None, max_length=20000)
    ocr: bool = True


class ReviewIn(BaseModel):
    verified: bool


async def _edit(request: Request, doc_id: str, op) -> dict:
    """Load → apply `op(record, extractor)` on the worker thread → save → return."""
    doc = _doc_or_404(request, doc_id)
    if doc.status != "done" or not doc.record:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is not processed yet")
    worker: Worker = request.app.state.worker
    db: Database = request.app.state.db

    def run(extractor):
        record = db.get(doc_id).record   # re-read inside the serialized section
        updated = op(record, extractor)
        db.save_result(doc_id, updated)
        return updated

    try:
        record = await run_in_threadpool(worker.call, run)
    except corrections.CorrectionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
    out = _summary(db.get(doc_id))
    out["record"] = _with_urls(record)
    return out


@app.patch("/api/documents/{doc_id}/components/{cid}", dependencies=[Depends(auth)])
async def patch_component(request: Request, doc_id: str, cid: str, body: ComponentPatch):
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    if "bbox" in changes:
        changes["bbox"] = body.bbox.model_dump()
    if not changes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nothing to change")
    return await _edit(request, doc_id, lambda rec, ex: corrections.update_component(
        settings, rec, cid, changes, ocr=ex.ocr))


@app.post("/api/documents/{doc_id}/components", status_code=status.HTTP_201_CREATED, dependencies=[Depends(auth)])
async def add_component(request: Request, doc_id: str, body: ComponentNew):
    created: dict = {}

    def op(rec, ex):
        updated, cid = corrections.add_component(settings, rec, body.page, body.bbox.model_dump(), body.type,
                                                 text=body.text, ocr=ex.ocr, run_ocr=body.ocr)
        created["id"] = cid
        return updated
    out = await _edit(request, doc_id, op)
    out["new_component_id"] = created.get("id")
    return out


@app.delete("/api/documents/{doc_id}/components/{cid}", dependencies=[Depends(auth)])
async def delete_component(request: Request, doc_id: str, cid: str):
    return await _edit(request, doc_id, lambda rec, ex: corrections.delete_component(settings, rec, cid))


@app.post("/api/documents/{doc_id}/review", dependencies=[Depends(auth)])
async def review(request: Request, doc_id: str, body: ReviewIn):
    return await _edit(request, doc_id, lambda rec, ex: corrections.set_review(rec, body.verified))


# ---- AI review (local LLM suggests, a person decides) -----------------------------------

class AcceptIn(BaseModel):
    value: str | None = Field(default=None, max_length=20000)


_ai_lock = threading.Lock()   # one LLM review at a time: a 16 GB Mac can't run two 8B models well


def _sid(sid: str) -> str:
    if not llm_review.SID_RE.match(sid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return sid


@app.get("/api/llm", dependencies=[Depends(auth)])
def llm_status():
    try:
        client = llm_review.OllamaClient(settings)
    except llm_review.ReviewError as exc:
        return {"enabled": settings.llm_enabled, "model": settings.llm_model, "reachable": False, "error": str(exc)}
    return client.status() if settings.llm_enabled else {"enabled": False, "model": settings.llm_model}


@app.post("/api/documents/{doc_id}/ai-review", dependencies=[Depends(auth)])
async def run_ai_review(request: Request, doc_id: str):
    if not settings.llm_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "AI review is disabled (NEPDOC_LLM_ENABLED=0)")
    doc = _doc_or_404(request, doc_id)
    if doc.status != "done" or not doc.record:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is not processed yet")
    if not _ai_lock.acquire(blocking=False):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "An AI review is already running")
    try:
        # The LLM call runs in a thread-pool thread, not the worker, so uploads keep processing.
        # Accepting checks each suggestion against the current text, so edits made meanwhile are safe.
        review = await run_in_threadpool(llm_review.run_review, settings, doc.record, request.app.state.llm_client)
    except llm_review.ReviewError as exc:
        raise HTTPException(exc.status, str(exc)) from None
    finally:
        _ai_lock.release()
    return await _edit(request, doc_id, lambda rec, ex: llm_review.store(rec, review))


class RecheckIn(BaseModel):
    component_id: str = Field(max_length=32, pattern=r"^p\d{1,3}-c\d{3,4}$")


@app.post("/api/documents/{doc_id}/ai-review/recheck", dependencies=[Depends(auth)])
async def ai_recheck(request: Request, doc_id: str, body: RecheckIn):
    """Re-check around one manually corrected component (called by the UI after an edit)."""
    if not settings.llm_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "AI review is disabled (NEPDOC_LLM_ENABLED=0)")
    doc = _doc_or_404(request, doc_id)
    if doc.status != "done" or not doc.record:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is not processed yet")
    if not _ai_lock.acquire(blocking=False):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "An AI review is already running")
    try:
        result = await run_in_threadpool(llm_review.run_recheck, settings, doc.record, body.component_id,
                                         request.app.state.llm_client)
    except llm_review.ReviewError as exc:
        raise HTTPException(exc.status, str(exc)) from None
    finally:
        _ai_lock.release()
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No correction found for that component")
    out = await _edit(request, doc_id, lambda rec, ex: llm_review.merge_recheck(rec, result))
    out["recheck"] = {"component_id": result["component_id"], "suggestions": len(result["suggestions"]),
                      "checked": len(result["checked"])}
    return out


@app.post("/api/documents/{doc_id}/ai-review/accept-safe", dependencies=[Depends(auth)])
async def ai_accept_safe(request: Request, doc_id: str):
    return await _edit(request, doc_id, lambda rec, ex: llm_review.accept_safe(settings, rec))


@app.post("/api/documents/{doc_id}/ai-review/{sid}/accept", dependencies=[Depends(auth)])
async def ai_accept(request: Request, doc_id: str, sid: str, body: AcceptIn | None = None):
    sid, value = _sid(sid), (body.value if body else None)
    return await _edit(request, doc_id, lambda rec, ex: llm_review.accept(settings, rec, sid, value))


@app.post("/api/documents/{doc_id}/ai-review/{sid}/reject", dependencies=[Depends(auth)])
async def ai_reject(request: Request, doc_id: str, sid: str):
    sid = _sid(sid)
    return await _edit(request, doc_id, lambda rec, ex: llm_review.reject(rec, sid))


@app.delete("/api/documents/{doc_id}/ai-review", dependencies=[Depends(auth)])
async def ai_discard(request: Request, doc_id: str):
    return await _edit(request, doc_id, lambda rec, ex: llm_review.discard(rec))


# ---- built frontend (optional): `npm run build` then open http://127.0.0.1:8765 ----------

DIST = ROOT / "frontend" / "dist"


@app.get("/{path:path}", include_in_schema=False)
def spa(request: Request, path: str):
    if path.startswith("api/") or not DIST.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    candidate = safe_child(DIST, path) if path else DIST / "index.html"
    target = candidate if candidate.is_file() else DIST / "index.html"
    resp = FileResponse(target)
    if target.name == "index.html":
        require_host = request.headers.get("host", "").rsplit(":", 1)[0].strip("[]")
        if require_host in {"localhost", "127.0.0.1", "::1", *settings.allowed_hosts}:
            # Same-origin browser session; SameSite=Strict keeps other sites from using it.
            resp.set_cookie(SESSION_COOKIE, request.app.state.token, httponly=True, samesite="strict",
                            secure=False, path="/api")
    return resp
