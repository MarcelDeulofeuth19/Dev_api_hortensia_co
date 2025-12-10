#!/usr/bin/env bash
set -euo pipefail

export FLASK_APP=src.api.flask_app:app
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=${PORT:-5000}

python -m flask run

