# Technical audit: improvements, limitations, vulnerabilities

Status markers: ✅ implemented in this codebase · 🔧 recommended next step · ⚠️ residual risk you must manage.

Versions reviewed: surya-ocr 0.24.0, doclayout-yolo 0.0.4 (DocStructBench weights), FastAPI, Pillow, OpenCV, pypdfium2.

---

## 0. Licensing (read first if this will be a product)

| Component | Code | Weights | Implication |
|---|---|---|---|
| doclayout-yolo | **AGPL-3.0** | Apache-2.0 | Offering this app over a network to others triggers AGPL source-sharing for the whole app. For a closed product: export the weights to ONNX and run them with onnxruntime (no AGPL code), or switch the layout model. |
| surya-ocr | Apache-2.0 | Modified AI Pubs Open Rail-M | Free for research, personal use and startups under $5M funding/revenue; otherwise a commercial licence is needed. |
| Tesseract (fallback) | Apache-2.0 | Apache-2.0 | No restrictions. |

---

## 1. Architectural and accuracy improvements

### 1.1 Nepali OCR post-processing
- ✅ NFC normalisation, so नुक्ता and composed forms compare equal. Zero-width joiners (ZWJ/ZWNJ) are stripped.
- ✅ Separate `text` (original numerals) and `text_normalized` (ASCII digits) fields. Each component is tagged with its script (`ne` / `en` / `mixed`).
- ✅ Bikram Sambat dates are validated and converted to AD with `nepali-datetime`, so impossible dates such as 32 Baisakh in a 31-day year are rejected.
- 🔧 **Confusable-aware lexicon correction.** Tesseract-family OCR swaps थ/ध, व/ब, ड/ह and ७/©. Run edit-distance correction weighted for those pairs against closed vocabularies: 77 districts, 753 local levels, office names (मालपोत कार्यालय, जिल्ला प्रशासन कार्यालय) and common surnames. Mark corrected values with a lower confidence.
- 🔧 **Numeric-context repair.** Inside fields known to be numeric, map look-alikes (`o/O→0`, `l/|→1`, `©→7`, `%→5`). Never apply this globally.
- 🔧 **Cross-field validation.** Issue date ≥ date of birth. The fiscal year `2081/82` must agree with the Miti year. Chalani/Darta numbers are numeric.
- 🔧 **Dual-engine agreement.** Run Surya and Tesseract on low-confidence regions. Agreement means high confidence; disagreement goes to human review. Surya is a VLM, so it fails *plausibly*; a second engine catches confident hallucinations.
- 🔧 **Human-in-the-loop corrections.** Store corrections as `source: manual`. They become the evaluation set and, later, fine-tuning data.

### 1.2 Stamp and text overlap
- ✅ The ink detector builds per-colour masks. Before OCR, stamp ink is painted paper-white on a copy of the page, so text under seals can be read. The original crops are untouched.
- ✅ Transparent stamp PNGs use the **ink mask** rather than background distance, so black text printed under a seal doesn't end up in the stamp asset.
- 🔧 **Use `cv2.inpaint` instead of white fill** on textured or guilloché paper (citizenship cards), so the stamp area doesn't leave white "holes" that confuse recognition.
- 🔧 **Colour unmixing where ink crosses text.** Pixels that are dark *and* reddish are a stamp-over-text mixture. Keep them as text (low saturation after removing the red component) instead of erasing them.
- 🔧 **OCR both versions** (original and stamp-suppressed) for regions flagged `overlapped_by_stamp`, and keep the higher-confidence reading.

### 1.3 Key-value fields (Chalani No., Darta No., Patra Sankhya, Miti)
- ✅ Regex rules with Nepali and English label variants (`च.नं.`, `चलानी नं.`, `द.नं.`, `प.सं.`, `आ.व.`, `ना.प्र.नं.`, `मिति`) run over reading-ordered text. Every hit links back to its component, so the UI can highlight it.
- 🔧 **Spatial pairing instead of text-only matching.** Find the label's box, then take the nearest value box to the right on the same baseline, else directly below. This is more robust when OCR splits a line or the value is in a separate cell or handwritten.
- 🔧 **Per-document-type schemas** (government letter, citizenship, lalpurja, birth certificate), with required fields and per-field validators. Reuse the schema-driven parser from the Digital Locker project.
- 🔧 **Larger collections:** fine-tune a layout-aware model (LayoutLMv3-style) on 300+ labelled Nepali forms.

### 1.4 M2 / 16 GB memory and speed
- ✅ One worker thread, models loaded once, one document at a time (bounded queue → 429). Memory use stays predictable.
- ✅ All text regions on a page go to Surya in **one batched call** (block mode via `LayoutResult`). YOLO runs on MPS with automatic CPU retry. The MPS cache is cleared between stages.
- ✅ Defaults: `SURYA_INFERENCE_PARALLEL=2`, PDFs rendered at 200 DPI, at most 5 pages per upload.
- 🔧 Set `SURYA_INFERENCE_KEEP_ALIVE=true` if you restart the backend often, so the model isn't reloaded. Then pin `SURYA_INFERENCE_API_KEY` yourself, because the per-run key changes with each restart.
- 🔧 **Batch across pages:** collect every region of a multi-page PDF into one Surya call to use the server's parallel slots.
- 🔧 Don't run Ollama or other GPU-heavy apps at the same time. Watch Activity Monitor → Memory Pressure; yellow means swapping, which makes llama.cpp 5–10× slower.
- 🔧 Consider Surya's experimental MLX backend (`SURYA_INFERENCE_BACKEND=mlx`) and benchmark it against llama.cpp on your own documents.
- 🔧 **Deskew** before layout. A projection-profile search, as in the Digital Locker preprocessing, is cheap and protects both models (see 2.1).

---

## 2. Known limitations

### 2.1 DocLayout-YOLO
- **No stamp, signature or logo classes.** DocStructBench covers academic, textbook, market-analysis and financial documents. Seals come out as `figure`, `abandon` or nothing. This app compensates with the ink detector, which only works for coloured ink (2.3).
- **Out of distribution for Nepali forms.** Devanagari headers, ruled government forms and citizenship cards weren't in its training data. Expect merged or split blocks, form fields detected as one big `plain_text`, and dense tables missed.
- **Skewed scans.** It outputs axis-aligned boxes. Beyond about 5° of skew, boxes swallow neighbouring lines and the reading order degrades. This app does not deskew yet (🔧 1.4).
- **The OpenCV fallback layout** is only suitable for clean, ruled letters and forms.

### 2.2 Surya OCR (2.x)
- **Nepali is supported but not perfect.** Surya's own multilingual benchmark reports 84.9% for Nepali, below its overall average. Measure on your documents.
- **Handwritten Devanagari is not a design target.** Results on handwritten names and numbers are unreliable. Because it's a VLM, it can return fluent but wrong text instead of visibly broken output.
- **Confidence is mean token probability, not calibrated accuracy.** Treat it as a ranking signal and route low values to review.
- **Block mode is less accurate than full-page mode,** by Surya's own documentation. We use block mode so each text maps to a YOLO box. 🔧 Optionally run full-page OCR and align its lines to the boxes.
- **First call is slow:** about 10–30 s while `llama-server` starts and loads the model.

### 2.3 Overlapping red and blue ink stamps
- HSV thresholds assume saturated red and blue ink. **Faded, purple, green or black stamps aren't detected.** Black stamps can't be told apart from print by colour at all.
- **Blue stamps and blue ballpoint handwriting are both blue.** Classification uses shape and position heuristics (aspect ratio, fill, a printed label to the left), which will sometimes be wrong.
- **Red letterheads or red printed text** may be flagged as ink, and stamp suppression would erase it before OCR. Suppression only applies inside detected stamp boxes, which limits but doesn't remove this risk.
- **Two overlapping stamps** of different colours merge into one component.
- **Signatures in black ink** are not detected. YOLO may label them `abandon` or `figure`.

### 2.4 Multi-page PDFs
- Only the first `NEPDOC_MAX_PDF_PAGES` (default 5) are processed. Pages are rendered at 200 DPI.
- **No cross-page logic:** tables that continue onto the next page, and reading order that spans pages, are not stitched.
- **Digital PDFs' embedded text layer is ignored** (everything is rasterised and OCR'd). 🔧 Extract the text layer with pypdfium2 when present: it's exact and instant.

### 2.5 Reading order
- XY-cut handles single and two-column layouts and label-value form rows. It fails on sidebars, wrapped figures, rotated text and tables of contents.

---

## 3. Security vulnerabilities and mitigations

Threat model: a single-user Mac holding citizenship cards, land records and legal papers.
- **Main risks:** other software on the machine, malicious websites reaching `localhost`, hostile files, and copies of data in backups or sync.
- **Out of scope:** malware running as your own user can read everything regardless.

### 3.1 Path traversal on uploads and assets
- ✅ The client filename is **display-only**, sanitised to its basename. Storage paths use server-generated UUIDs (`[0-9a-f]{32}`).
- ✅ Asset names must match `^p\d+-c\d{3}(-alpha)?\.png$|^page-\d+\.png$`. Paths are resolved and must stay inside the data root, which blocks `../`, encoded `%2F`, null bytes and symlinks.
- ✅ The SPA file route uses the same `safe_child` check. Tests cover six traversal payloads.

### 3.2 Malicious files and decompression bombs
- ✅ **Size limit before parsing:** `Content-Length` is checked in middleware (413), so oversized bodies aren't spooled to disk. The body read is also capped.
- ✅ **Type by magic bytes**, cross-checked against Pillow's detected format. An `.exe` renamed `.png` gets 415.
- ✅ **Pixel bomb guard:** dimensions are read from the header before decoding (`Image.open` is lazy). `MAX_IMAGE_PIXELS` is set (60 MP default) and `DecompressionBombWarning` is treated as an error. A test uses an 8000×8000 PNG of a few KB.
- ✅ **PDFs:** page count capped. Render size is computed from the page box before rendering and the scale reduced to stay under the pixel cap. Multi-frame TIFFs are capped too.
- ✅ **EXIF/GPS stripped:** pages and crops are re-encoded from pixels. The original upload is **never written to disk**; only its SHA-256 is kept.
- ⚠️ **Parser vulnerabilities** in Pillow, pdfium or OpenCV are the residual risk. Keep them updated (`pip list --outdated`). 🔧 Decode in a short-lived subprocess with `resource.setrlimit(RLIMIT_AS, …)` and a timeout, so a parser crash or memory blow-up can't take down the app.
- ⚠️ **Model weights are pickles.** `.pt` files execute code when loaded. Download only from the official repo. 🔧 Pin and verify the SHA-256 in `download_models.py`, or use an ONNX export.

### 3.3 Unauthenticated local API exposure
- ✅ **Bound to 127.0.0.1** by default; the Vite dev server is also forced to 127.0.0.1.
- ✅ **Bearer token:** a random 256-bit token in `data/.api_token` (mode 0600), compared in constant time. The **dev proxy injects it server-side**, so browser JavaScript never sees it.
- ✅ **DNS-rebinding guard:** only `Host: localhost / 127.0.0.1 / ::1` is accepted (403 otherwise). A malicious site that resolves to 127.0.0.1 still sends its own Host header.
- ✅ **Built-UI mode:** an `HttpOnly; SameSite=Strict` session cookie scoped to `/api`. Cookie-authenticated POST/DELETE requests must come from a loopback Origin.
- ✅ **llama-server hardening:** Surya starts `llama-server` without `--api-key`. This app passes a per-run random key via `LLAMA_CPP_EXTRA_ARGS` and `SURYA_INFERENCE_API_KEY`.
  - ⚠️ I verified this against Surya's source but couldn't run llama-server here. If OCR falls back to Tesseract with a 401, set `NEPDOC_LLAMA_API_KEY=0`.
- ✅ A queue bound (429) and a single worker prevent local resource-exhaustion floods.
- ⚠️ `/api/docs` (OpenAPI) is readable without a token. It reveals endpoint shapes, no data. Remove `docs_url` if you prefer.
- ⚠️ Never set `NEPDOC_HOST=0.0.0.0` on a laptop. If you need LAN access, put it behind a reverse proxy with TLS and real user accounts.

### 3.4 CORS misconfiguration
- ✅ **CORS is closed by default.** The dev proxy makes the UI same-origin, so no CORS is needed at all. If you set `NEPDOC_CORS_ORIGINS`, origins must be explicit and `allow_credentials=False`. Never use `*` together with credentials.
- ✅ A test asserts that a preflight from `evil.example.com` gets no `Access-Control-Allow-Origin`.

### 3.5 Temporary files
- ✅ Starlette spools uploads over 1 MB to disk. `tempfile.tempdir` points to `data/tmp` (0700) instead of the shared `/tmp`. The spooled file is closed and deleted immediately after reading, and leftovers from crashes are cleared at startup.
- ✅ Decoded pages live in memory only, until the page PNG is written to the private data dir.
- ⚠️ macOS swap can still page memory to disk. It's encrypted when FileVault is on.

### 3.6 Sensitive asset storage
- ✅ The data dir is 0700. Assets are served only through the authenticated route with `Cache-Control: private, no-store`, `nosniff` and CORP `same-origin`.
- ✅ Deleting a document removes its database rows (cascade), crops and page images. SQLite `secure_delete=ON` overwrites freed pages. Optional retention: `NEPDOC_RETENTION_DAYS`.
- ✅ Logs never contain extracted text; only IDs and counts are logged.
- ⚠️ **Data is stored unencrypted by the app.** Turn on **FileVault**.
- ⚠️ **Keep the project out of iCloud Drive.** On many Macs `~/Documents` and `~/Desktop` sync to iCloud, which would upload every scanned citizenship card. Use `~/Work/nepdoc` or set `NEPDOC_DATA_DIR` to a non-synced folder.
- ⚠️ **Exclude `data/` from Time Machine,** or encrypt the backup disk. `setup_mac.sh` adds `.metadata_never_index` so Spotlight doesn't index the document text.
- 🔧 **App-level encryption:** a per-document key wrapped by a key in the macOS Keychain, with envelope encryption as in the Digital Locker project. Deleting the key then crypto-shreds the document.

### 3.7 Model and supply-chain integrity
- ✅ Offline mode (`HF_HUB_OFFLINE=1`) after the first download, so there are no silent model updates.
- 🔧 Pin exact versions with hashes (`pip-compile --generate-hashes`) and verify model SHA-256 values. Review dependency updates before upgrading. `doclayout-yolo` pulls in a large tree (ultralytics fork, matplotlib, etc.).
- ⚠️ **VLM prompt injection:** a document containing text such as "ignore previous instructions" could steer Surya's output. The output is only stored as text, never executed or used for decisions, so the impact is limited to wrong extracted text. That's another reason to keep human review for legal records.

### 3.8 AI review (local LLM)
- ✅ **Stays local by default.** `NEPDOC_LLM_URL` must be a loopback address; any other host is refused unless `NEPDOC_LLM_ALLOW_REMOTE=1`. The HTTP client ignores `HTTP(S)_PROXY`/`ALL_PROXY` (`trust_env=False`) and doesn't follow redirects, so a system proxy can't route document text off the machine.
- ✅ **Text only.** No page images or crops are sent. Each component's text is capped at 1,500 characters, and truncated components can't receive a text replacement.
- ✅ **Untrusted output.** The reply must match a JSON schema (enforced by Ollama's `format`, then re-checked). Suggestions for unknown or unsent IDs, unknown actions or types, human-corrected (locked) components, no-op or oversized rewrites, and duplicates are dropped and counted as `rejected`. The model has no tools and can't write; suggestions are stored as `pending`.
- ✅ **Human in the loop.** Changes apply only on Accept. Digit changes, large rewrites, type changes and deletions are flagged and excluded from "Accept all safe". Accepting re-checks that the component still has the text the model saw; if not, the suggestion becomes `stale` and nothing changes.
- ✅ **Audit trail.** Accepted changes carry `text_source: "llm"` (or `"manual"` if you edited the suggestion), `attributes.suggested_by`, and `via: "ai_review:<model>"` in `corrections`. The run records a SHA-256 of exactly what was sent. Document text is never logged.
- ✅ **Automatic re-checks** use the same client, validation and lock. Each one sends a bounded set of components (the corrected one is always locked) and records a SHA-256 of what was sent in `ai_review.rechecks`. Only one LLM request runs at a time; extra requests get 429 and the UI retries.
- ⚠️ **Prompt injection** in a document can still make the model suggest wrong text. The impact is limited to a suggestion that a person has to accept.
- ⚠️ **Ollama itself** listens on 127.0.0.1:11434 with no authentication, so any local process can use it. Don't set `OLLAMA_HOST=0.0.0.0`.
- 🔧 **Cloud LLMs (not implemented):** if added later, send only redacted text (citizenship numbers, names, DOB replaced by placeholders and restored locally), use paid API terms that exclude training, opt in per document, and never send images.
