#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8000}
UVICORN_CMD="uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port ${PORT} --log-level info"

if [[ "${RELOAD:-}" != "" ]]; then
  ${UVICORN_CMD} --reload
else
  ${UVICORN_CMD}
fi
