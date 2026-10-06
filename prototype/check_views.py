# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Development diagnostic for framing robustness; no threshold/model changes."""
import argparse
import json
from pathlib import Path
import sys
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prototype.detector import Detector


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--truth",type=Path,required=True)
    args=parser.parse_args()
    directory=ROOT / "prototype/web/models";config=json.loads((directory / "config.json").read_text())
    detector=Detector(directory / "bp-detector.onnx",digit_model=directory / "bp-digits.onnx")
    results=[]
    for item in json.loads(args.truth.read_text()):
        image=cv2.imread(str(ROOT / item["local_file"]));views=[]
        for scale in (1,.75,.55):
            result=detector.read(image,config["minScore"],config["acceptScore"],scale)
            views.append({"scale":scale,"result":result})
        results.append({"id":item["id"],"expected":{k:item[k] for k in ("sys","dia","pulse")},"views":views})
        print(item["id"],[(v["scale"],v["result"]["reading"]) for v in views],flush=True)
    (ROOT / "prototype/reports/framing_views.json").write_text(json.dumps(results,indent=2))
