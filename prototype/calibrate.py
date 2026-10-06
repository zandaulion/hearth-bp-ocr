# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Choose thresholds on validation data only and freeze a release fingerprint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT / "evaluation"))
from prototype.detector import Detector,decode_output,letterbox
from prototype.reading import assemble
from benchmark import summarize


def records_for(cached,min_score,accept_score):
    records=[]
    for item in cached:
        result=assemble(item["detections"],min_score,accept_score)
        expected=item["expected"];prediction=result["reading"]
        fields={k:prediction is not None and prediction[k]==v for k,v in expected.items()}
        records.append({"id":item["id"],"local_file":item["local_file"],"expected":expected,"prediction":prediction,"result":result,"accepted":result["status"]=="candidate","error":None,"elapsed_ms":item["elapsed_ms"],"field_correct":fields,"exact_triplet":all(fields.values())})
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--freeze",action="store_true",help="Update browser config and release fingerprint using validation-only selection")
    parser.add_argument("--version",default="v1",help="Human-readable model revision")
    args=parser.parse_args();digits=ROOT / "prototype/web/models/bp-digits.onnx"
    detector=Detector(args.model,digit_model=digits)
    truth=json.loads((ROOT / "evaluation/artifacts/roboflow_valid_truth.json").read_text())
    cached=[]
    for item in truth:
        image=cv2.imread(str(ROOT / item["local_file"]));started=time.perf_counter()
        tensor,t=letterbox(image,detector.size)
        output=detector.session.run(None,{detector.input.name:tensor})[0]
        raw=decode_output(output,t,.10)
        detections=detector.recognizer.refine(image,raw)
        cached.append({"id":item["id"],"local_file":item["local_file"],"expected":{k:item[k] for k in ("sys","dia","pulse")},"detections":detections,"elapsed_ms":round((time.perf_counter()-started)*1000,2)})
    trials=[]
    for min_score in (.10,.15,.20,.25,.30,.35):
        for accept_score in (.25,.35,.45,.55,.65,.75,.85,.95):
            summary=summarize(records_for(cached,min_score,accept_score))
            trials.append({"min_score":min_score,"accept_score":accept_score,"summary":summary})
    # Priority: complete-reading correctness, then candidate precision >90%,
    # then coverage. Tiny validation estimates do not prove field reliability.
    highest=max(t["summary"]["correct_triplets"] for t in trials)
    best_accuracy=[t for t in trials if t["summary"]["correct_triplets"]==highest]
    eligible=[t for t in best_accuracy if (t["summary"]["accepted_precision"] or 0)>.9]
    if eligible:
        selected=max(eligible,key=lambda t:(t["summary"]["accepted_triplets"],t["accept_score"],t["min_score"]))
    else:
        selected=max(best_accuracy,key=lambda t:((t["summary"]["accepted_precision"] or 0),t["summary"]["accepted_triplets"],t["accept_score"]))
    report={"selection_dataset":"Roboflow validation: 25 structurally usable annotated triplets","test_used_for_selection":False,"model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"digit_model_sha256":hashlib.sha256(digits.read_bytes()).hexdigest(),"selected":selected,"trials":trials,"records":records_for(cached,selected["min_score"],selected["accept_score"]),"cached_detections":cached}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    if args.freeze:
        model_dir=ROOT / "prototype/web/models";config=json.loads((model_dir / "config.json").read_text())
        config.update({"minScore":selected["min_score"],"acceptScore":selected["accept_score"],"detectorSha256":report["model_sha256"],"digitsSha256":report["digit_model_sha256"]})
        ui_hash=hashlib.sha256(b"".join(p.read_bytes() for p in sorted((ROOT / "prototype/web").glob("*.mjs")))).hexdigest()[:8]
        config["version"]=args.version+"-"+report["model_sha256"][:10]+"-"+report["digit_model_sha256"][:8]+"-"+str(selected["min_score"])+"-"+str(selected["accept_score"])+"-"+ui_hash
        (model_dir / "config.json").write_text(json.dumps(config,indent=2))
        sw=ROOT / "prototype/web/sw.js";source=sw.read_text();lines=source.splitlines();lines[0]=f"const CACHE='hearth-bp-{config['version']}';";sw.write_text("\n".join(lines)+"\n")
        (model_dir / "release.json").write_text(json.dumps({"config":config,"validation_summary":selected["summary"],"held_out_test_evaluated":False,"test_status":"First frozen test was consumed by v1. Subsequent evaluations of those 13 images are development/regression results." if args.version!="v1" else "Reserved until first frozen evaluation.","source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "prototype").glob("*.py")}},indent=2))
    print(json.dumps(selected,indent=2))


if __name__=="__main__":main()
