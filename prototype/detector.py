# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Single-pass Python ONNX reference; see docs/API_REFERENCE.md.

Input is BGR uint8. Browser resizing/grayscale rounding and fallback orchestration
can produce different results. Read thresholds from the released config rather
than relying on the research defaults below.
"""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from prototype.reading import CLASS_NAMES, assemble, nms
from prototype.digit_model import digit_tensor


def letterbox(image, size=512,view_scale=1):
    h,w = image.shape[:2]
    scale = min(size/w, size/h)*view_scale
    nw,nh = round(w*scale),round(h*scale)
    left,top = (size-nw)//2,(size-nh)//2
    canvas = np.full((size,size,3),114,dtype=np.uint8)
    canvas[top:top+nh,left:left+nw] = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    tensor = canvas[:,:,::-1].transpose(2,0,1).astype(np.float32)[None]/255
    return tensor,{"scale":scale,"left":left,"top":top,"width":w,"height":h}


def decode_output(output, transform, min_score=.25):
    predictions = np.asarray(output)[0]
    if predictions.shape[0] != 4+len(CLASS_NAMES):
        raise ValueError(f"Unexpected detector output shape {output.shape}")
    detections = []
    for values in predictions.T:
        cid = int(np.argmax(values[4:]))
        score = float(values[4+cid])
        if score < min_score:
            continue
        cx,cy,w,h = map(float,values[:4])
        box = [(cx-w/2-transform["left"])/transform["scale"],
               (cy-h/2-transform["top"])/transform["scale"],
               (cx+w/2-transform["left"])/transform["scale"],
               (cy+h/2-transform["top"])/transform["scale"]]
        box = [max(0,min(v,transform["width"] if i%2==0 else transform["height"])) for i,v in enumerate(box)]
        if box[2] > box[0] and box[3] > box[1]:
            detections.append({"class":CLASS_NAMES[cid],"score":score,"box":box})
    return nms(detections)


class DigitRecognizer:
    """Refine detected digit crops; class 10 is background, not a numeric row."""
    def __init__(self,model,threads=2,min_score=.5):
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
        self.input=self.session.get_inputs()[0].name;self.min_score=min_score

    def refine(self,image,detections):
        digits=[d for d in detections if d["class"]!="row"]
        if not digits:return detections
        crops=[];usable=[];h,w=image.shape[:2]
        for d in digits:
            x1,y1,x2,y2=d["box"]
            crop=image[max(0,int(y1)):min(h,int(np.ceil(y2))),max(0,int(x1)):min(w,int(np.ceil(x2)))]
            if crop.size:
                crops.append(digit_tensor(crop));usable.append(d)
        kept=[d for d in detections if d["class"]=="row"]
        if not crops:return kept
        logits=self.session.run(None,{self.input:np.stack(crops).astype(np.float32)})[0]
        exp=np.exp(logits-logits.max(axis=1,keepdims=True));prob=exp/exp.sum(axis=1,keepdims=True)
        for d,p in zip(usable,prob):
            cid=int(p.argmax());confidence=float(p[cid])
            if cid==10 or confidence<self.min_score:continue
            kept.append({**d,"class":str(cid),"detector_class":d["class"],"recognition_score":confidence,"score":min(d["score"],confidence)})
        return kept


class Detector:
    """Reusable CPU sessions for one image pass, without crop fallback stages."""
    def __init__(self, model, threads=4,digit_model=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.size = int(self.input.shape[-1])
        self.recognizer=DigitRecognizer(digit_model) if digit_model else None

    def read(self,image,min_score=.25,accept_score=.75,view_scale=1):
        tensor,transform = letterbox(image,self.size,view_scale)
        output = self.session.run(None,{self.input.name:tensor})[0]
        detections = decode_output(output,transform,min_score)
        if self.recognizer:detections=self.recognizer.refine(image,detections)
        return assemble(detections,min_score,accept_score)
