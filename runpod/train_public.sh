#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
EPOCHS="${EPOCHS:-40}"
RUN_NAME="${RUN_NAME:-runpod_public_v1}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if [[ -z "${ROBOFLOW_API_KEY:-}" && -n "${RUNPOD_SECRET_ROBOFLOW_API_KEY:-}" ]]; then
  export ROBOFLOW_API_KEY="$RUNPOD_SECRET_ROBOFLOW_API_KEY"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url "$CUDA_INDEX_URL"
python -m pip install -r prototype/requirements.txt

python - <<'PY'
import json
import torch

details = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(details, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this Pod")
PY

if [[ ! -f dataset/roboflow_bp_display/data.yaml ]]; then
  if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
    echo "ROBOFLOW_API_KEY or RUNPOD_SECRET_ROBOFLOW_API_KEY is required to download the public dataset" >&2
    exit 2
  fi
  python -m pip install roboflow
  python download_roboflow_dataset.py
fi

python evaluation/dataset_tools.py
python prototype/prepare_data.py
python prototype/bootstrap.py
python prototype/train_detector.py --epochs "$EPOCHS" --name "$RUN_NAME" --device 0
python prototype/train_digits.py --epochs "$EPOCHS" --device 0

BEST="prototype/runs/$RUN_NAME/weights/best.pt"
install -m 0644 "$BEST" prototype/models/bp-detector.pt
python prototype/export_model.py "$BEST"
python prototype/verify_export.py
python prototype/evaluate.py \
  --model prototype/web/models/bp-detector.onnx \
  --digits-model prototype/web/models/bp-digits.onnx \
  --truth evaluation/artifacts/roboflow_valid_truth.json \
  --output prototype/reports/runpod-public-validation.json \
  --min-score 0.25 \
  --accept-score 0.75 \
  --dataset-role validation
python prototype/calibrate.py \
  --model prototype/web/models/bp-detector.onnx \
  --output prototype/reports/runpod-public-calibration.json \
  --version runpod-public-v1

python prototype/test_reading.py
python evaluation/test_metrics.py

tar -czf prototype/reports/runpod-public-output.tar.gz \
  "$BEST" \
  prototype/models/bp-detector.pt \
  prototype/models/digits.pt \
  prototype/models/digits_validation.json \
  prototype/web/models/bp-detector.onnx \
  prototype/web/models/bp-digits.onnx \
  prototype/web/models/config.json \
  prototype/reports/export_parity.json \
  prototype/reports/runpod-public-validation.json \
  prototype/reports/runpod-public-calibration.json

echo "Training complete: $ROOT/prototype/reports/runpod-public-output.tar.gz"
