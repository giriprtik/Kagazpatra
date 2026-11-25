"""First-run model caching. Run once while online; afterwards the app works offline.

    python scripts/download_models.py            # DocLayout-YOLO + Surya GGUF
    python scripts/download_models.py --check    # verify everything is cached (no network)

Then set HF_HUB_OFFLINE=1 in .env so nothing tries to reach the internet again.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402

YOLO_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
YOLO_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only verify the local cache")
    ap.add_argument("--skip-surya", action="store_true")
    args = ap.parse_args()
    if args.check:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from huggingface_hub import hf_hub_download

    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    # 1. DocLayout-YOLO weights → models/ (explicit path, no hidden cache)
    target = s.yolo_weights_path
    if target.exists():
        print(f"✓ DocLayout-YOLO weights: {target}")
    elif args.check:
        print(f"✗ DocLayout-YOLO weights missing: {target}"); ok = False
    else:
        print(f"↓ DocLayout-YOLO from {YOLO_REPO} …")
        path = hf_hub_download(YOLO_REPO, YOLO_FILE)
        shutil.copy(path, target)
        print(f"✓ saved {target} ({target.stat().st_size / 1e6:.0f} MB)")

    # 2. Surya 2.x GGUF model + vision projector (llama.cpp backend) → HF cache
    if not args.skip_surya:
        from surya.settings import settings as ss

        for fname in (ss.SURYA_GGUF_MODEL_FILE, ss.SURYA_GGUF_MMPROJ_FILE):
            try:
                p = hf_hub_download(ss.SURYA_GGUF_REPO, fname, local_files_only=args.check)
                print(f"✓ Surya {fname}: {p}")
            except Exception as exc:  # noqa: BLE001
                print(f"✗ Surya {fname}: {exc}"); ok = False
        if not shutil.which(ss.LLAMA_CPP_BINARY or "llama-server"):
            print("✗ llama-server not on PATH — install with: brew install llama.cpp"); ok = False
        else:
            print(f"✓ llama-server: {shutil.which('llama-server')}")

    # 3. Device report
    from backend.extractor import select_device
    print(f"✓ YOLO device: {select_device(s.device)}")
    print("\nAll set — add HF_HUB_OFFLINE=1 to .env for strict offline mode." if ok
          else "\nSome items are missing (see ✗ above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
