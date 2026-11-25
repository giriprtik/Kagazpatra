"""Settings (environment variables, `.env` in the project root is loaded automatically)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            if value[:1] in ('"', "'") and value[:1] in value[1:]:
                value = value[1:value.index(value[0], 1)]       # quoted: keep everything inside
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # drop inline comment
            os.environ.setdefault(key.strip(), value)


_load_dotenv()

# Must be set before torch is imported anywhere: unsupported MPS ops fall back to CPU
# instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Surya spawns `llama-server` on a random localhost port *without* an API key, so any
# local process could use it. Give it a per-run random key (read by Surya's client via
# SURYA_INFERENCE_API_KEY; passed to the server via LLAMA_CPP_EXTRA_ARGS). Must happen
# before `surya` is imported. Disable with NEPDOC_LLAMA_API_KEY=0 if your llama.cpp
# build rejects the unauthenticated /health probe.
if os.getenv("NEPDOC_LLAMA_API_KEY", "1").lower() not in {"0", "false", "no"} \
        and not os.getenv("SURYA_INFERENCE_API_KEY") and not os.getenv("SURYA_INFERENCE_URL"):
    import secrets as _secrets

    _key = _secrets.token_urlsafe(24)
    os.environ["SURYA_INFERENCE_API_KEY"] = _key
    os.environ["LLAMA_CPP_EXTRA_ARGS"] = f"{os.getenv('LLAMA_CPP_EXTRA_ARGS', '')} --api-key {_key}".strip()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _path("NEPDOC_DATA_DIR", ROOT / "data"))
    models_dir: Path = field(default_factory=lambda: _path("NEPDOC_MODELS_DIR", ROOT / "models"))
    database_url: str = os.getenv("NEPDOC_DATABASE_URL", "")  # default: sqlite in data_dir

    # Network exposure. Keep 127.0.0.1 unless you really mean to serve the LAN.
    host: str = os.getenv("NEPDOC_HOST", "127.0.0.1")
    port: int = int(os.getenv("NEPDOC_PORT", "8765"))
    # Extra Host header values to accept (DNS-rebinding guard), comma separated.
    allowed_hosts: tuple[str, ...] = tuple(
        h.strip() for h in os.getenv("NEPDOC_ALLOWED_HOSTS", "").split(",") if h.strip())
    # The Vite dev proxy makes the browser same-origin, so CORS stays closed by default.
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.getenv("NEPDOC_CORS_ORIGINS", "").split(",") if o.strip())

    # Upload limits
    max_upload_mb: int = int(os.getenv("NEPDOC_MAX_UPLOAD_MB", "25"))
    max_pixels: int = int(os.getenv("NEPDOC_MAX_PIXELS", str(60_000_000)))  # ~ A3 @ 600 dpi
    max_pdf_pages: int = int(os.getenv("NEPDOC_MAX_PDF_PAGES", "5"))
    pdf_dpi: int = int(os.getenv("NEPDOC_PDF_DPI", "200"))
    max_queue: int = int(os.getenv("NEPDOC_MAX_QUEUE", "5"))
    retention_days: int = int(os.getenv("NEPDOC_RETENTION_DAYS", "0"))  # 0 = keep forever

    # Layout (DocLayout-YOLO)
    layout_engine: str = os.getenv("NEPDOC_LAYOUT_ENGINE", "auto")  # auto | yolo | heuristic
    yolo_weights: str = os.getenv("NEPDOC_YOLO_WEIGHTS", "doclayout_yolo_docstructbench_imgsz1024.pt")
    yolo_imgsz: int = int(os.getenv("NEPDOC_YOLO_IMGSZ", "1024"))
    yolo_conf: float = float(os.getenv("NEPDOC_YOLO_CONF", "0.25"))
    device: str = os.getenv("NEPDOC_DEVICE", "auto")  # auto | mps | cpu

    # OCR
    ocr_engine: str = os.getenv("NEPDOC_OCR_ENGINE", "auto")  # auto | surya | tesseract | none
    tesseract_langs: str = os.getenv("NEPDOC_TESSERACT_LANGS", "nep+eng")

    # AI review (local LLM via Ollama). Only loopback URLs are accepted unless
    # NEPDOC_LLM_ALLOW_REMOTE=1, so document text never leaves the machine by accident.
    llm_enabled: bool = _bool("NEPDOC_LLM_ENABLED", True)
    llm_url: str = os.getenv("NEPDOC_LLM_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("NEPDOC_LLM_MODEL", "qwen3:8b")
    llm_timeout: int = int(os.getenv("NEPDOC_LLM_TIMEOUT", "300"))
    llm_num_ctx: int = int(os.getenv("NEPDOC_LLM_NUM_CTX", "8192"))
    llm_batch_chars: int = int(os.getenv("NEPDOC_LLM_BATCH_CHARS", "3500"))
    llm_allow_remote: bool = _bool("NEPDOC_LLM_ALLOW_REMOTE", False)

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def token_file(self) -> Path:
        return self.data_dir / ".api_token"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'nepdoc.db'}"

    @property
    def yolo_weights_path(self) -> Path:
        p = Path(self.yolo_weights)
        return p if p.is_absolute() else self.models_dir / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
