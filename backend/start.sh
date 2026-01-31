#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=10000}"


# Debugging: List files and show env
echo "Current directory: $(pwd)"
echo "Listing files in current directory:"
ls -la
echo "Listing files in app directory:"
ls -la app || echo "app directory not found"
echo "PYTHONPATH: $PYTHONPATH"

exec waitress-serve --port="$PORT" app.main:app
