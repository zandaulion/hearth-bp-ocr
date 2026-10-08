# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Local transfer learning for numeric rows and seven-segment digits."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "prototype/.cache/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import torch
from ultralytics import YOLO, settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as 0")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--name", default="detector_v1")
    parser.add_argument("--weights", default=str(ROOT / "prototype/models/yolo11n.pt"))
    parser.add_argument("--data",default=str(ROOT / "prototype/data/bp.yaml"))
    parser.add_argument("--lr",type=float,default=.002)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = "0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} was requested, but torch.cuda.is_available() is false")
    print(json.dumps({
        "training_device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "amp": args.amp,
        "batch": args.batch,
    }), flush=True)
    settings.update({"sync": False, "tensorboard": False, "wandb": False, "mlflow": False, "comet": False, "clearml": False, "dvc": False, "raytune": False})
    model = YOLO(args.weights)
    started = time.perf_counter()

    def progress(trainer):
        metrics = {str(k): float(v) for k,v in trainer.metrics.items()}
        print("EPOCH_RESULT " + json.dumps({"epoch": trainer.epoch + 1, "elapsed_seconds": round(time.perf_counter()-started, 1), "metrics": metrics}), flush=True)

    model.add_callback("on_fit_epoch_end", progress)
    model.train(data=args.data, epochs=args.epochs,
                imgsz=args.size, batch=args.batch, device=device, workers=0,
                project=str(ROOT / "prototype/runs"), name=args.name,
                freeze=args.freeze, optimizer="AdamW", lr0=args.lr, lrf=.1,
                warmup_epochs=1, weight_decay=.0005, patience=12,
                degrees=12, translate=.12, scale=.3, shear=3,
                perspective=.0002, fliplr=0, flipud=0, mosaic=.3,
                close_mosaic=5, mixup=0, hsv_h=.02, hsv_s=.2, hsv_v=.3,
                cache="ram", plots=False, save=True, seed=20261006,
                deterministic=True, amp=args.amp, verbose=False)


if __name__ == "__main__":
    main()
