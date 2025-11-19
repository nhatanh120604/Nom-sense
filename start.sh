#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=10000}"
: "${PERSIST_DIR:=/opt/render/project/data/chroma}"
: "${CHROMA_ARCHIVE_ID:?CHROMA_ARCHIVE_ID env var is required}"

# Download DB only if the dir is missing or empty
if [ ! -d "$PERSIST_DIR" ] || [ -z "$(ls -A "$PERSIST_DIR" 2>/dev/null || true)" ]; then
  echo "Downloading Chroma archive..."
  mkdir -p "$PERSIST_DIR"
  python -m gdown "$CHROMA_ARCHIVE_ID" -O /tmp/chroma_db.zip
  unzip -qo /tmp/chroma_db.zip -d .
  rm /tmp/chroma_db.zip
fi

# Debugging: List files and show env
echo "Current directory: $(pwd)"
echo "Listing files in current directory:"
ls -la
echo "Listing files in app directory:"
ls -la app || echo "app directory not found"
echo "PYTHONPATH: $PYTHONPATH"

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
