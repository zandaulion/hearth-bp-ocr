# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Evaluate complete readings with autonomous predictions, never annotation crops."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "evaluation"))
os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT / "prototype/.cache/ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True,exist_ok=True)
import cv2
from prototype.reading import CLASS_NAMES, assemble
from benchmark import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--truth",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--min-score",type=float,default=.25)
    parser.add_argument("--accept-score",type=float,default=.75)
    parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--digits-model",type=Path)
    parser.add_argument("--dataset-role",default="development",choices=("training","validation","first-held-out-test","development","regression"))
    args = parser.parse_args()
    if args.model.suffix == ".onnx":
        from prototype.detector import Detector
        model = Detector(args.model,digit_model=args.digits_model)
        def read(image):
            return model.read(image,args.min_score,args.accept_score)
    else:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        model = YOLO(str(args.model))
        from prototype.detector import DigitRecognizer
        refiner=DigitRecognizer(args.digits_model) if args.digits_model else None
        def read(image):
            result = model.predict(image,conf=args.min_score,iou=.45,imgsz=args.size,device="cpu",verbose=False)[0]
            detections = [{"class":CLASS_NAMES[int(cid)],"box":list(map(float,box)),"score":float(score)}
                          for box,score,cid in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist())]
            if refiner:detections=refiner.refine(image,detections)
            return assemble(detections,args.min_score,args.accept_score)
    records = []
    for item in json.loads(args.truth.read_text()):
        expected = {k:item[k] for k in ("sys","dia","pulse")}
        record = {"id":item["id"],"local_file":item["local_file"],"expected":expected,"prediction":None,"accepted":False,"error":None}
        start = time.perf_counter()
        try:
            image = cv2.imread(str(ROOT / item["local_file"]))
            if image is None:
                raise ValueError("Could not read image")
            result = read(image)
            record["result"] = result
            record["prediction"] = result["reading"]
            record["accepted"] = result["status"] == "candidate"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_ms"] = round((time.perf_counter()-start)*1000,2)
        record["field_correct"] = {k:None if v is None else record["prediction"] is not None and record["prediction"][k] == v for k,v in expected.items()}
        record["exact_triplet"] = all(record["field_correct"].values()) if all(v is not None for v in expected.values()) else None
        records.append(record)
        print(f'{record["id"]}: predicted={record["prediction"]} exact={record["exact_triplet"]}',flush=True)
    report = {"reader":"learned row/digit detector plus crop recognizer", "dataset_role":args.dataset_role,"model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"model":str(args.model),"digit_model":str(args.digits_model) if args.digits_model else None,"digit_model_sha256":hashlib.sha256(args.digits_model.read_bytes()).hexdigest() if args.digits_model else None,"truth":str(args.truth),"min_score":args.min_score,"accept_score":args.accept_score,"acceptance_definition":"Unambiguous three-row geometry and consistency checks; weakest detection/recognition score >= threshold. Scores are not calibrated probabilities.","summary":summarize(records),"records":records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report["summary"],indent=2))


if __name__ == "__main__":
    main()
