# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Export a detector and reset baseline config; run verify_export.py separately."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT / "prototype/.cache/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True,exist_ok=True)
os.environ.setdefault("YOLO_AUTOINSTALL","false")
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--size",type=int,default=512)
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = YOLO(str(args.checkpoint)).export(format="onnx",imgsz=args.size,opset=17,simplify=False,dynamic=False,nms=False,device="cpu",batch=1)
    out = ROOT / "prototype/web/models"
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,out / "bp-detector.onnx")
    (out / "config.json").write_text(json.dumps({"version":"v1","size":args.size,"minScore":.25,"acceptScore":.75,"digitMinScore":.5,"classNames":["0","1","row","2","3","4","5","6","7","8","9"],"scope":"Upright common home monitors with three SYS/DIA/pulse rows","scoreNotice":"Detection and recognition scores are not calibrated correctness probabilities"},indent=2))
    print(out / "bp-detector.onnx")


if __name__ == "__main__":
    main()
