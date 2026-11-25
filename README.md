# NepDoc Local

An offline document-processing workbench for Nepali documents: mixed Devanagari and English text, government emblems, ink stamps, signatures, tables and forms. Everything runs on your Mac. No document, image or text leaves the machine.

```
upload ─► DocLayout-YOLO (layout, MPS) ─► ink detector (stamps / signatures / emblems / handwriting)
       ─► crops + transparent PNGs ─► stamp ink removed ─► Surya OCR (ne + en, batched, local llama.cpp)
       ─► reading order ─► fields (च.नं., द.नं., प.सं., मिति BS→AD …) ─► JSON ─► SQLite / Postgres JSONB
```

**Read [AUDIT.md](AUDIT.md) before using this with real citizenship or land records.** It covers accuracy limits, security controls and licensing.

## What changed from the original brief, and why

| Brief | Reality (verified against surya-ocr 0.24 and doclayout-yolo 0.0.4) | What this app does |
|---|---|---|
| Surya on PyTorch MPS | Surya 2.x is a vision-language model served by a local **llama.cpp** server (Metal on Apple Silicon). It no longer depends on PyTorch. | Runs Surya through `llama-server` (Metal). PyTorch MPS is used only for DocLayout-YOLO. |
| Surya languages `ne`/`en` | 2.x has no language parameter; one model reads 90+ languages, Nepali included. | No language config needed. Nepali post-processing runs afterwards. |
| YOLO detects logos, stamps, signatures | DocStructBench classes are `title, plain_text, abandon, figure, figure_caption, table, table_caption, table_footnote, isolate_formula, formula_caption`. There are no stamp, signature or logo classes. | An **OpenCV ink detector** (red/blue HSV) finds stamps, signatures, emblems and handwritten fill-ins, and relabels YOLO's `figure`/`abandon` boxes. |

## Setup (Apple Silicon, one time, needs internet once)

```bash
./nepdoc.sh          # sets up whatever is missing, then starts Ollama, backend and UI
./nepdoc.sh setup    # setup / update only
./nepdoc.sh run      # start only
./nepdoc.sh stop     # stop leftover NepDoc processes
./nepdoc.sh check    # what's installed and running
./nepdoc.sh ports 9000 5200   # backend 9000, UI 5200 (saved in .env); `ports default` = 8765 / 5173
./nepdoc.sh reset             # delete all documents, crops, AI reviews and the database (asks first)
./nepdoc.sh reset settings    # .env back to defaults (old one kept as .env.bak)
```

`nepdoc.sh` is safe to re-run. It installs missing Homebrew packages (Python 3.12, Node, llama.cpp, Tesseract with Nepali, Ollama), creates `.venv` and reinstalls packages only when `requirements.txt` changes, creates `.env`, downloads the models once and then turns on offline mode, runs `npm install` when `package.json` changes, and pulls the Ollama model. `run` stops any old NepDoc instance first, waits for the backend, opens the browser and shows both logs (also in `data/logs/`). Ctrl+C stops everything it started.

Or step by step:

```bash
brew install python@3.12 node llama.cpp tesseract tesseract-lang
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python scripts/download_models.py          # caches the models (below)
echo "HF_HUB_OFFLINE=1" >> .env            # strict offline from now on
python scripts/download_models.py --check  # verifies with networking disabled

uvicorn backend.main:app --host 127.0.0.1 --port 8765
cd frontend && npm install && npm run dev  # → http://127.0.0.1:5173
```

**First-run model caching:** `download_models.py` fetches two things.
- **DocLayout-YOLO:** `doclayout_yolo_docstructbench_imgsz1024.pt` from `juliozhao/DocLayout-YOLO-DocStructBench`, saved to `models/`.
- **Surya:** the GGUF model and vision projector from `datalab-to/surya-ocr-2.5-gguf`, saved to the Hugging Face cache.

After that, set `HF_HUB_OFFLINE=1`; nothing reaches the network. The first OCR of each session starts `llama-server` (about 10–30 s), and later documents reuse it.

**Single-process mode:** run `cd frontend && npm run build`, then open http://127.0.0.1:8765. FastAPI serves the UI and sets a same-site session cookie.

**Graceful fallbacks:** if the YOLO weights are missing, an OpenCV layout heuristic is used. If `llama-server` is missing, Tesseract (`nep+eng`) is used. The UI header and each record's `warnings` say what actually ran.

## API

All endpoints are under `/api` and need `Authorization: Bearer <data/.api_token>`. In dev, the Vite proxy adds it automatically.

| Method | Path | |
|---|---|---|
| GET | `/api/health` | liveness (no auth) |
| GET | `/api/engines` | engines, device, fallbacks, limits |
| POST | `/api/documents` | multipart `file` + `transparent`, `suppress_stamp_ink`, `ocr_tables`, `ocr_stamps` → `202 {id}` |
| GET | `/api/documents?q=` | list / full-text search |
| GET | `/api/documents/{id}` | status (`queued → processing → done/failed`, with `stage`) + record |
| GET | `/api/documents/{id}/record.json` | download the JSON |
| DELETE | `/api/documents/{id}` | deletes the row, crops and page images |
| GET | `/api/components?type=stamp&text=…` | query components across documents |
| GET | `/api/assets/{id}/{file}.png` | crops and page images (authenticated, `no-store`) |
| PATCH | `/api/documents/{id}/components/{cid}` | `{type?, text?, bbox?, reocr?}`: fix a component |
| POST | `/api/documents/{id}/components` | `{page, bbox, type, text?, ocr?}`: add a missing component |
| DELETE | `/api/documents/{id}/components/{cid}` | remove a false detection |
| POST | `/api/documents/{id}/review` | `{verified: bool}` |
| GET | `/api/llm` | local LLM status (reachable, model installed) |
| POST | `/api/documents/{id}/ai-review` | run the AI review → pending suggestions in `record.ai_review` |
| POST | `/api/documents/{id}/ai-review/{sid}/accept` | `{value?}`: apply one suggestion (optionally edited) |
| POST | `/api/documents/{id}/ai-review/{sid}/reject` | reject one suggestion |
| POST | `/api/documents/{id}/ai-review/recheck` | `{component_id}`: re-check around one corrected component |
| POST | `/api/documents/{id}/ai-review/accept-safe` | accept all suggestions without warning flags |
| DELETE | `/api/documents/{id}/ai-review` | discard the review |

## Correcting results

Open a document and click **✎ Correct mistakes**.

- **Wrong text:** click the box, edit the text on the right, then click **Save text**. **↻ Re-read** runs OCR again on just that box, which helps after you resize it.
- **Wrong box size or position:** drag the box to move it, or drag a corner handle to resize it. The crop PNG is regenerated.
- **Wrong type** (for example a signature detected as a stamp): change the type in the dropdown. Crops are created or removed to match.
- **Missing component:** choose a type next to **＋ Draw missing box**, then drag a rectangle on the page. Text and table boxes are OCR'd automatically.
- **False detection:** click **Delete**.
- **Mark reviewed** records that a person has checked the document.

Every edit is saved immediately and runs on the same worker as processing, so it never races OCR. After each edit, fields (Chalani, Miti and so on), full text, stats and the `components` table are recomputed. Each edit is appended to `record.corrections` (action, before, after, timestamp) and shown in the **History** tab. Edited components carry `text_source: "manual"` and `attributes.corrected: true`, so you can export them as ground truth later. IDs stay stable; only `reading_order` is renumbered.

## AI review (local LLM)

The **AI review** tab sends the OCR text (never images) to a local Ollama model, `qwen3:8b` by default, and lists the changes it suggests:

- **Fix text:** shown as a word diff. Use **Edit first** to adjust it before accepting.
- **Change type:** for example a printed label detected as a signature.
- **Delete box:** for noise.

Nothing changes until you click **Accept**. Suggestions that change digits, rewrite a lot, change a type or delete a box carry a ⚠ flag and are left out of **Accept all safe**. Text you corrected by hand is locked, so the model never sees it as editable. If you edit a component after the review ran, its suggestion becomes *stale* and isn't applied.

**Automatic re-check after your edits** (on by default, toggle in the AI review tab). When you save text, change a type, re-read or add a box, the UI waits about 1.5 s for a pause in editing and then asks the model to look again. It sends the box you corrected (locked, as ground truth with its before/after text), its 3 neighbours on each side, and up to 10 other boxes that contain the words you changed. The model then suggests the same fix where the same OCR mistake appears. The re-checks run one at a time in the background, and the page header shows "AI re-checking…" and then "AI found N more suggestions". The new suggestions replace older pending ones for the same boxes (marked `superseded`) and still need your Accept. Box moves alone don't trigger a re-check.

Setup: `ollama serve` and `ollama pull qwen3:8b`. Settings are in `.env` (`NEPDOC_LLM_*`). The URL must be local unless you set `NEPDOC_LLM_ALLOW_REMOTE=1`. See AUDIT.md §3.8 for the security model.

## JSON record (schema 1.0)

```jsonc
{
  "schema_version": "1.0",
  "document_id": "a5c01edf…",
  "source": { "filename": "letter.png", "content_type": "image/png", "size_bytes": 175740, "sha256": "…" },
  "engines": { "layout": "doclayout-yolo@mps", "ocr": "surya-ocr 0.24.0 (llamacpp)", "device": "mps" },
  "page_count": 1,
  "pages": [{ "page": 1, "width": 1700, "height": 2200, "image_path": "pages/<id>/page-1.png" }],
  "components": [{
    "id": "p1-c013", "page": 1, "reading_order": 13,
    "type": "stamp", "layer": "visual", "source": "ink-detector", "confidence": 0.71,
    "bbox": { "x0": 710, "y0": 1710, "x1": 995, "y1": 1995, "width": 285, "height": 285 },
    "bbox_norm": { "x0": 0.418, "y0": 0.777, "x1": 0.585, "y1": 0.907 },
    "text": null, "text_normalized": null, "text_confidence": null, "script": null, "html": null,
    "asset": { "png": "assets/<id>/p1-c013.png", "transparent_png": "assets/<id>/p1-c013-alpha.png" },
    "attributes": { "ink_color": "red", "ink_fill": 0.21, "aspect": 1.0 }
  }],
  "fields": [{ "key": "date", "label": "Date (Miti)", "value": "२०८१/०५/१२",
               "normalized": { "bs": "2081-05-12", "ad": "2024-08-28" }, "component_id": "p1-c004" }],
  "text": "…all text in reading order…",
  "stats": { "components": 15, "by_type": { "stamp": 1, "…": 0 }, "timings": { "layout_ms": 79, "ocr_ms": 2316 } },
  "warnings": []
}
```

**Layer types:**
- **Visual:** `emblem`, `logo`, `figure`, `stamp`, `signature`.
- **Text:** `title`, `text`, `header_footer`, `caption`, `footnote`, `handwriting`.
- **Other:** `table` (OCR'd; Surya also returns `html`) and `formula`.

`text` keeps the original Unicode, NFC-normalised. `text_normalized` has ASCII digits and is used for search.

## Database

SQLite is the default (`data/nepdoc.db`, WAL, `secure_delete`). For PostgreSQL, set `NEPDOC_DATABASE_URL=postgresql+psycopg://…` and install `psycopg`. The record is then stored as **JSONB** with a GIN index.

```sql
-- every stamp across all documents
SELECT document_id, component_id, x0, y0, x1, y1 FROM components WHERE type = 'stamp';
-- Postgres JSONB: documents with Chalani No. 125
SELECT id, filename FROM documents
WHERE record @> '{"fields": [{"key": "chalani_no", "normalized": "125"}]}';
```

## Tests

```bash
pytest backend/tests        # 37 tests: extraction, Surya adapter, reading order, security controls
```

The tests use the OpenCV layout and a fake or Tesseract OCR, so they run without the large models. The Surya adapter test uses Surya's real `LayoutResult` and `PageOCRResult` types.

## Layout

```
backend/
  config.py        settings (.env), MPS fallback env, llama-server API key
  extractor.py     models, MPS/CPU, layout, ink detector, crops, transparency, OCR batching, reading order
  postprocess.py   Nepali normalisation, script detection, Chalani/Darta/Patra/Miti fields, BS→AD
  security.py      upload validation, decompression-bomb guard, token + Host checks, safe paths
  database.py      SQLAlchemy models (JSON/JSONB), CRUD, retention purge
  main.py          FastAPI app, worker thread, endpoints, asset serving, SPA
frontend/src/
  App.jsx, api.js, types.js
  components/      UploadZone, DocumentCanvas (bbox overlay), ComponentList, LayerGallery,
                   FieldsPanel, JsonInspector, HistorySidebar, EngineBar
scripts/           setup_mac.sh, run.sh, download_models.py
```
