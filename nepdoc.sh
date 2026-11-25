#!/usr/bin/env bash
# NepDoc Local: one script to set up and run everything on an Apple Silicon Mac.
#
#   ./nepdoc.sh          set up whatever is missing, then start everything
#   ./nepdoc.sh setup    set up / update only
#   ./nepdoc.sh run      start only (no setup checks)
#   ./nepdoc.sh stop     stop NepDoc processes left running
#   ./nepdoc.sh check    show what is installed and running
#   ./nepdoc.sh ports 9000 5200     use backend port 9000 and UI port 5200 (saved in .env)
#   ./nepdoc.sh ports default       back to 8765 / 5173
#   ./nepdoc.sh reset               delete all documents, crops and the database (asks first)
#   ./nepdoc.sh reset settings      put .env back to the defaults (old one saved as .env.bak)
#
# Safe to run again and again: each step is skipped when it's already done.
# Starts: Ollama (if not running) · backend 127.0.0.1:8765 · UI http://127.0.0.1:5173
# Press Ctrl+C to stop everything this script started.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

OLLAMA_URL="http://127.0.0.1:11434"
PY=python3.12

bold() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

env_value() {   # read KEY from .env (ignores comments and quotes)
  [ -f .env ] || return 0
  { grep -E "^$1=" .env || true; } | tail -1 | cut -d= -f2- | sed -E 's/[[:space:]]+#.*$//; s/^["'\'']//; s/["'\'']$//'
}

# Ports and data folder come from the environment, then .env, then the defaults.
API_PORT="${NEPDOC_PORT:-$(env_value NEPDOC_PORT)}"; API_PORT="${API_PORT:-8765}"
UI_PORT="${NEPDOC_UI_PORT:-$(env_value NEPDOC_UI_PORT)}"; UI_PORT="${UI_PORT:-5173}"
DATA_DIR="${NEPDOC_DATA_DIR:-$(env_value NEPDOC_DATA_DIR)}"; DATA_DIR="${DATA_DIR:-$ROOT/data}"
case "$DATA_DIR" in "~"*) DATA_DIR="$HOME${DATA_DIR#\~}" ;; /*) ;; *) DATA_DIR="$ROOT/$DATA_DIR" ;; esac
LOGS="$DATA_DIR/logs"
export NEPDOC_PORT="$API_PORT" NEPDOC_UI_PORT="$UI_PORT" NEPDOC_DATA_DIR="$DATA_DIR"

llm_model() { local m; m="$(env_value NEPDOC_LLM_MODEL)"; echo "${m:-qwen3:8b}"; }
llm_enabled() { case "$(env_value NEPDOC_LLM_ENABLED)" in 0|false|no|off) return 1 ;; *) return 0 ;; esac; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
ollama_up() { curl -fsS -m 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

start_ollama() {
  ollama_up && { ok "Ollama is running"; return 0; }
  command -v ollama >/dev/null || { warn "Ollama not installed; AI review will be unavailable"; return 0; }
  mkdir -p "$LOGS"
  if [ -d /Applications/Ollama.app ]; then
    open -g -a Ollama
  else
    nohup ollama serve >"$LOGS/ollama.log" 2>&1 &
    OLLAMA_PID=$!
  fi
  for _ in $(seq 1 30); do ollama_up && { ok "Ollama started"; return 0; }; sleep 1; done
  warn "Ollama didn't start (see $LOGS/ollama.log); AI review will be unavailable"
}

# ---- setup -----------------------------------------------------------------------------

setup() {
  [ "$(uname -s)" = "Darwin" ] || die "This script is for macOS."
  [ "$(uname -m)" = "arm64" ] || warn "Not Apple Silicon: it will run, but slowly (CPU only)."

  bold "Homebrew packages"
  command -v brew >/dev/null || die "Install Homebrew first: https://brew.sh  (then run this script again)"
  local missing=""
  for f in python@3.12 node llama.cpp tesseract tesseract-lang; do
    brew list --formula "$f" >/dev/null 2>&1 || missing="$missing $f"
  done
  command -v ollama >/dev/null || [ -d /Applications/Ollama.app ] || missing="$missing ollama"
  if [ -n "$missing" ]; then
    echo "  installing:$missing"
    # shellcheck disable=SC2086
    brew install $missing
  fi
  ok "python 3.12, node, llama.cpp, tesseract (+nep), ollama"

  bold "Python environment"
  command -v "$PY" >/dev/null || PY="$(brew --prefix python@3.12)/bin/python3.12"
  if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv
    ok "created .venv"
  fi
  local req_sha; req_sha="$(sha requirements.txt)"
  if [ "$(cat .venv/.requirements.sha 2>/dev/null || true)" != "$req_sha" ]; then
    .venv/bin/python -m pip install --upgrade pip -q
    .venv/bin/python -m pip install -r requirements.txt
    echo "$req_sha" > .venv/.requirements.sha
  fi
  ok "Python packages up to date"

  bold "Configuration"
  if [ ! -f .env ]; then cp .env.example .env; ok "created .env from .env.example"; else ok ".env exists (kept)"; fi
  mkdir -p "$DATA_DIR" "$LOGS" && chmod 700 "$DATA_DIR"
  touch "$DATA_DIR/.metadata_never_index"
  ok "data folder is private (700) and excluded from Spotlight"

  bold "OCR and layout models"
  if .venv/bin/python scripts/download_models.py --check >/dev/null 2>&1; then
    ok "DocLayout-YOLO and Surya are cached"
  else
    echo "  downloading (needs internet once; ~1-2 GB)…"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 .venv/bin/python scripts/download_models.py \
      || warn "Model download incomplete: NepDoc will use the OpenCV/Tesseract fallbacks until it succeeds"
  fi
  if .venv/bin/python scripts/download_models.py --check >/dev/null 2>&1 && ! grep -q '^HF_HUB_OFFLINE=1' .env; then
    printf '\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\n' >> .env
    ok "strict offline mode enabled in .env"
  fi

  bold "Frontend"
  local lock_sha; lock_sha="$(sha frontend/package.json)"
  if [ ! -d frontend/node_modules ] || [ "$(cat frontend/node_modules/.nepdoc.sha 2>/dev/null || true)" != "$lock_sha" ]; then
    (cd frontend && npm install --no-audit --no-fund)
    echo "$lock_sha" > frontend/node_modules/.nepdoc.sha
  fi
  ok "node modules up to date"

  bold "Local LLM for AI review"
  if llm_enabled; then
    start_ollama
    if ollama_up; then
      local model; model="$(llm_model)"
      if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx -e "$model" -e "$model:latest"; then
        ok "$model is installed"
      else
        echo "  pulling $model (one time, ~5 GB)…"
        ollama pull "$model" || warn "Couldn't pull $model; AI review will be unavailable"
      fi
    fi
  else
    ok "AI review disabled in .env (NEPDOC_LLM_ENABLED=0)"
  fi
}

# ---- run -------------------------------------------------------------------------------

stop_nepdoc() {
  pkill -f "uvicorn backend.main:app" 2>/dev/null || true
  pkill -f "$ROOT/frontend/node_modules/.bin/vite" 2>/dev/null || true
  pkill -f "node .*vite.*--strictPort" 2>/dev/null || true
  sleep 1
}

cleanup() {
  trap - INT TERM EXIT
  echo
  bold "Stopping"
  [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null || true
  [ -n "${UI_PID:-}" ] && kill "$UI_PID" 2>/dev/null || true
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
  [ -n "${OLLAMA_PID:-}" ] && kill "$OLLAMA_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  ok "stopped"
}

run() {
  [ -x .venv/bin/python ] || die "Not set up yet. Run: ./nepdoc.sh setup"
  [ -d frontend/node_modules ] || die "Frontend not installed. Run: ./nepdoc.sh setup"
  mkdir -p "$LOGS"

  bold "Starting NepDoc"
  # Old copies would keep serving old code (the cause of the earlier 415 error), so stop them first.
  if port_busy "$API_PORT" || port_busy "$UI_PORT"; then
    warn "stopping a previous NepDoc instance"
    stop_nepdoc
  fi
  port_busy "$API_PORT" && die "Port $API_PORT is used by another program: lsof -nP -iTCP:$API_PORT"
  port_busy "$UI_PORT" && die "Port $UI_PORT is used by another program: lsof -nP -iTCP:$UI_PORT"

  trap cleanup INT TERM EXIT
  llm_enabled && start_ollama

  : > "$LOGS/backend.log"; : > "$LOGS/frontend.log"
  .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "$API_PORT" >>"$LOGS/backend.log" 2>&1 &
  API_PID=$!
  printf '  waiting for the backend'
  for _ in $(seq 1 120); do
    kill -0 "$API_PID" 2>/dev/null || { echo; tail -30 "$LOGS/backend.log"; die "Backend exited (log: $LOGS/backend.log)"; }
    curl -fsS -m 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 && [ -f "$DATA_DIR/.api_token" ] && break
    printf '.'; sleep 1
  done
  echo
  ok "backend on http://127.0.0.1:$API_PORT (models keep loading in the background)"

  (cd frontend && exec npx vite --host 127.0.0.1 --port "$UI_PORT" --strictPort) >>"$LOGS/frontend.log" 2>&1 &
  UI_PID=$!
  for _ in $(seq 1 30); do curl -fsS -m 2 "http://127.0.0.1:$UI_PORT/" >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS -m 2 "http://127.0.0.1:$UI_PORT/" >/dev/null 2>&1 \
    || { tail -20 "$LOGS/frontend.log"; die "UI didn't start (log: $LOGS/frontend.log)"; }
  ok "UI on http://127.0.0.1:$UI_PORT"
  open "http://127.0.0.1:$UI_PORT" 2>/dev/null || true

  bold "Running. Logs below (also in $LOGS). Press Ctrl+C to stop."
  tail -n 0 -F "$LOGS/backend.log" "$LOGS/frontend.log" &
  TAIL_PID=$!
  while kill -0 "$API_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do sleep 2; done
  warn "a process exited; see $LOGS"
}

check() {
  bold "Status"
  [ -x .venv/bin/python ] && ok ".venv" || warn ".venv missing"
  [ -d frontend/node_modules ] && ok "frontend/node_modules" || warn "frontend not installed"
  [ -f .env ] && ok ".env" || warn ".env missing"
  if [ -x .venv/bin/python ]; then .venv/bin/python scripts/download_models.py --check || true; fi
  command -v ollama >/dev/null && ok "ollama installed" || warn "ollama not installed"
  ollama_up && ok "Ollama running; models: $(ollama list 2>/dev/null | awk 'NR>1 {printf "%s ", $1}')" || warn "Ollama not running"
  port_busy "$API_PORT" && ok "backend listening on $API_PORT" || warn "backend not running"
  port_busy "$UI_PORT" && ok "UI listening on $UI_PORT" || warn "UI not running"
  ok "ports from settings: backend $API_PORT, UI $UI_PORT · data folder $DATA_DIR"
}

set_ports() {
  local api ui
  if [ "${1:-}" = "default" ]; then api=8765; ui=5173
  else api="${1:-}"; ui="${2:-$UI_PORT}"; fi
  for p in "$api" "$ui"; do
    case "$p" in ''|*[!0-9]*) die "Usage: ./nepdoc.sh ports <backend-port> [ui-port]   or   ./nepdoc.sh ports default" ;; esac
    [ "$p" -ge 1024 ] && [ "$p" -le 65535 ] || die "Ports must be between 1024 and 65535"
  done
  [ "$api" != "$ui" ] || die "The backend and UI need different ports"
  [ -f .env ] || cp .env.example .env
  local tmp; tmp="$(mktemp)"
  grep -vE '^(NEPDOC_PORT|NEPDOC_UI_PORT)=' .env > "$tmp" || true
  printf 'NEPDOC_PORT=%s\nNEPDOC_UI_PORT=%s\n' "$api" "$ui" >> "$tmp"
  mv "$tmp" .env
  ok "backend port $api, UI port $ui (saved in .env)"
  port_busy "$API_PORT" || port_busy "$UI_PORT" && warn "NepDoc is running on the old ports; restart it: ./nepdoc.sh run"
  echo "  Open http://127.0.0.1:$ui after starting."
}

reset_data() {
  local db; db="$(env_value NEPDOC_DATABASE_URL)"
  case "$DATA_DIR" in /|"$HOME"|"$HOME/"|"") die "Refusing to reset: data folder is $DATA_DIR" ;; esac
  bold "Reset documents"
  echo "  This permanently deletes every document, crop, page image, AI review and the database in:"
  echo "    $DATA_DIR"
  [ -n "$db" ] && warn "NEPDOC_DATABASE_URL is set: that external database is NOT touched; clear it yourself."
  echo "  Models, settings (.env), .venv and node_modules are kept."
  printf '  Type RESET to continue: '
  local answer; read -r answer
  [ "$answer" = "RESET" ] || { echo "  Cancelled."; exit 0; }
  stop_nepdoc
  rm -rf "$DATA_DIR/assets" "$DATA_DIR/pages" "$DATA_DIR/tmp" "$DATA_DIR/logs"
  rm -f "$DATA_DIR/nepdoc.db" "$DATA_DIR/nepdoc.db-wal" "$DATA_DIR/nepdoc.db-shm" "$DATA_DIR/.api_token"
  ok "deleted. A fresh database and API token are created on the next start: ./nepdoc.sh run"
}

reset_settings() {
  bold "Reset settings"
  [ -f .env ] && cp .env .env.bak && ok "old settings saved to .env.bak"
  cp .env.example .env
  ok ".env restored to defaults (ports 8765 / 5173)"
  if [ -x .venv/bin/python ] && .venv/bin/python scripts/download_models.py --check >/dev/null 2>&1; then
    printf '\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\n' >> .env
    ok "offline mode kept on (models are cached)"
  fi
}

case "${1:-all}" in
  all)   setup; run ;;
  setup) setup; bold "Setup complete. Start with: ./nepdoc.sh run" ;;
  run)   run ;;
  stop)  stop_nepdoc; ok "stopped" ;;
  check) check ;;
  ports) shift; set_ports "$@" ;;
  reset) if [ "${2:-}" = "settings" ]; then reset_settings; else reset_data; fi ;;
  *)     sed -n '2,16p' "$0"; exit 1 ;;
esac
