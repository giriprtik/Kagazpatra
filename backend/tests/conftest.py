import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="nepdoc-test-")
os.environ.update({
    "NEPDOC_DATA_DIR": _tmp, "NEPDOC_LAYOUT_ENGINE": "heuristic", "NEPDOC_OCR_ENGINE": "none",
    "NEPDOC_API_TOKEN": "test-token", "NEPDOC_MAX_UPLOAD_MB": "5", "NEPDOC_MAX_PIXELS": "30000000",
    "NEPDOC_LLM_URL": "http://127.0.0.1:9",  # never reach a real Ollama from tests
})
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def letter_png() -> bytes:
    return (FIXTURES / "nepali_letter.png").read_bytes()


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(_tmp)
