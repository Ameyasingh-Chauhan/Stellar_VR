#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p "${MODELS_DIR:-/tmp/stellar_vr_backend_cpu/models}"
echo "Setup complete. Run 'python setup_models.py' to pre-download heavy models."
