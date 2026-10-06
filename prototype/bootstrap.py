# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Download the official starting weights once, before offline local training."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "prototype/.cache/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.checks import check_font


if __name__ == "__main__":
    models = ROOT / "prototype/models"
    models.mkdir(parents=True, exist_ok=True)
    print(attempt_download_asset(str(models / "yolo11n.pt")), flush=True)
    print(check_font("Arial.ttf"), flush=True)
