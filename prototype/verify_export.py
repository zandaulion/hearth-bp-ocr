# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Compare numeric predictions before/after export on fixed validation inputs."""
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT / "prototype/.cache/ultralytics"))
from ultralytics import YOLO
from prototype.detector import Detector,DigitRecognizer,letterbox
from prototype.digit_model import DigitNet,digit_tensor


def main():
    torch.set_num_threads(4)
    detector=Detector(ROOT / "prototype/web/models/bp-detector.onnx")
    torch_model=YOLO(str(ROOT / "prototype/models/bp-detector.pt")).model.eval().fuse()
    digit=DigitNet();digit.load_state_dict(torch.load(ROOT / "prototype/models/digits.pt",weights_only=True));digit.eval()
    recognizer=DigitRecognizer(ROOT / "prototype/web/models/bp-digits.onnx")
    truth=json.loads((ROOT / "evaluation/artifacts/roboflow_valid_truth.json").read_text())[:3]
    reports=[]
    for item in truth:
        image=cv2.imread(str(ROOT / item["local_file"]));x,_=letterbox(image,detector.size)
        with torch.inference_mode():expected=torch_model(torch.from_numpy(x))[0].numpy()
        actual=detector.session.run(None,{detector.input.name:x})[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=.003)
        crop=digit_tensor(image[image.shape[0]//4:image.shape[0]//2,image.shape[1]//4:image.shape[1]//2])[None]
        with torch.inference_mode():digit_expected=digit(torch.from_numpy(crop)).numpy()
        digit_actual=recognizer.session.run(None,{recognizer.input:crop})[0]
        np.testing.assert_allclose(digit_actual,digit_expected,rtol=1e-4,atol=1e-4)
        reports.append({"image":item["id"],"detector_max_abs_error":float(np.max(np.abs(actual-expected))),"classifier_max_abs_error":float(np.max(np.abs(digit_actual-digit_expected)))})
    out=ROOT / "prototype/reports/export_parity.json";out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"passed":True,"validation_only":True,"checks":reports},indent=2))
    print(out.read_text())


if __name__=="__main__":main()
