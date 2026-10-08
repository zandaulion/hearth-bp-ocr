# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Train/export a compact crop recognizer, using training images only."""
import json
import argparse
from pathlib import Path
import sys
import time
import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / "evaluation"))
from dataset_tools import annotations,DATA
from prototype.digit_model import DigitNet,digit_tensor
from prototype.reading import CLASS_NAMES


def load(split,clean_only=False):
    selected=None
    if clean_only:
        selected={Path(r["local_file"]).name for r in json.loads((ROOT / f"evaluation/artifacts/roboflow_{split}_truth.json").read_text())}
    crops,labels,metadata=[],[],[]
    for path in sorted((DATA / split / "images").glob("*.jpg")):
        if selected is not None and path.name not in selected:continue
        image=cv2.imread(str(path));h,w=image.shape[:2]
        for a in annotations(path):
            if a["class"]=="10":continue
            x1,y1,x2,y2=a["box"]
            crop=image[max(0,int(y1*h)):min(h,int(np.ceil(y2*h))),max(0,int(x1*w)):min(w,int(np.ceil(x2*w)))]
            crops.append(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY));labels.append(int(a["class"]));metadata.append({"file":path.name,"box":a["box"]})
    return crops,np.asarray(labels),metadata


class Crops(Dataset):
    def __init__(self,crops,labels,augment):self.crops=crops;self.labels=labels;self.augment=augment
    def __len__(self):return len(self.crops)
    def __getitem__(self,index):
        gray=cv2.resize(self.crops[index],(32,48),interpolation=cv2.INTER_LINEAR)
        if self.augment:
            matrix=cv2.getRotationMatrix2D((16,24),np.random.uniform(-12,12),np.random.uniform(.9,1.08))
            matrix[0,1]+=np.random.uniform(-.13,.13);matrix[:,2]+=np.random.uniform(-2,2,size=2)
            gray=cv2.warpAffine(gray,matrix,(32,48),borderValue=float(np.median(gray)))
            if np.random.random()<.18:gray=cv2.GaussianBlur(gray,(3,3),.6)
            if np.random.random()<.08:gray=255-gray
            gray=np.clip(gray.astype(np.float32)+np.random.normal(0,np.random.uniform(0,3),gray.shape),0,255)
        return torch.from_numpy(digit_tensor(gray)),int(self.labels[index])


def negatives(count=180):
    crops=[]
    for i in range(count):
        gray=np.full((48,32),np.random.uniform(50,210),dtype=np.float32)
        if i%3==0:cv2.circle(gray,(np.random.randint(8,25),np.random.randint(10,38)),np.random.randint(1,5),np.random.uniform(10,60),-1)
        if i%3==1:cv2.line(gray,(np.random.randint(4,28),12),(np.random.randint(4,28),25),np.random.uniform(10,60),np.random.randint(1,4))
        crops.append(np.clip(gray+np.random.normal(0,2,gray.shape),0,255).astype(np.uint8))
    return crops


def phone_crops():
    paths=sorted((ROOT / "prototype/data/phone_train/images").glob("*.jpg"));crops=[];labels=[]
    for path in paths:
        image=cv2.imread(str(path));h,w=image.shape[:2]
        for line in (path.parent.parent / "labels" / (path.stem+".txt")).read_text().splitlines():
            cid,cx,cy,bw,bh=map(float,line.split());name=CLASS_NAMES[int(cid)]
            if name=="row":continue
            crop=image[max(0,int((cy-bh/2)*h)):min(h,int(np.ceil((cy+bh/2)*h))),max(0,int((cx-bw/2)*w)):min(w,int(np.ceil((cx+bw/2)*w)))]
            crops.append(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY));labels.append(int(name))
    return crops,np.asarray(labels)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--phone",action="store_true");parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--device",default="auto",help="auto, cpu, or a CUDA device such as 0")
    parser.add_argument("--initial-weights",type=Path);parser.add_argument("--output-dir",type=Path,default=ROOT / "prototype/models")
    parser.add_argument("--web-dir",type=Path,default=ROOT / "prototype/web/models")
    args=parser.parse_args()
    requested="cuda:0" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    if requested.isdigit():requested=f"cuda:{requested}"
    device=torch.device(requested)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError(f"CUDA device {device} was requested, but torch.cuda.is_available() is false")
    print(json.dumps({"training_device":str(device),"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(device) if device.type=="cuda" else None}),flush=True)
    torch.set_num_threads(2);torch.manual_seed(20261006);np.random.seed(20261006);random.seed(20261006)
    train,labels,_=load("train",True);val,vlabels,meta=load("valid")
    n_original=len(train);phone,plabels=phone_crops() if args.phone else ([],np.asarray([],dtype=int))
    train+=phone;labels=np.concatenate((labels,plabels));n_real=len(train);bg=negatives();train+=bg;labels=np.concatenate((labels,np.full(len(bg),10)))
    counts=np.bincount(labels,minlength=11);weights=1/counts[labels];weights[n_original:n_real]*=12
    loader=DataLoader(Crops(train,labels,True),batch_size=96,sampler=WeightedRandomSampler(weights,2400,replacement=True),num_workers=0)
    vx=torch.stack([torch.from_numpy(digit_tensor(c)) for c in val]).to(device);vy=torch.from_numpy(vlabels).to(device)
    model=DigitNet().to(device)
    if args.initial_weights:model.load_state_dict(torch.load(args.initial_weights,weights_only=True,map_location=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007 if args.initial_weights else .0015,weight_decay=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.0001)
    px=torch.stack([torch.from_numpy(digit_tensor(c)) for c in phone]).to(device) if phone else None
    py=torch.from_numpy(plabels).to(device) if phone else None
    best=-1;started=time.perf_counter();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for x,y in loader:
            x=x.to(device);y=y.to(device)
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        schedule.step();model.eval()
        with torch.inference_mode():
            pred=model(vx).argmax(1);correct=int((pred==vy).sum());phone_correct=int((model(px).argmax(1)==py).sum()) if px is not None else 0
        selection=correct*1000+phone_correct
        if selection>best:best=selection;torch.save(model.state_dict(),out / "digits.pt")
        print(json.dumps({"epoch":epoch+1,"validation_digits_correct":correct,"validation_digits_total":len(vlabels),"phone_training_digits_correct":phone_correct,"phone_training_digits_total":len(phone),"selection_score":selection,"loss":round(float(np.mean(losses)),4),"elapsed_seconds":round(time.perf_counter()-started,1)}),flush=True)
    model.load_state_dict(torch.load(out / "digits.pt",weights_only=True,map_location=device));model.eval()
    with torch.inference_mode():prob=model(vx).softmax(1).cpu().numpy();pred=prob.argmax(1)
    groups={}
    for target,p,m in zip(vlabels,pred,meta):groups.setdefault(m["file"],True);groups[m["file"]]&=bool(target==p)
    report={"reader":"DigitNet crop recognition; annotated locations supplied", "training_images":239+(8 if phone else 0),"real_training_digits":n_real,"phone_training_digits":len(phone),"synthetic_background_crops":len(bg),"validation_digits":len(vlabels),"correct_digits":int((pred==vlabels).sum()),"all_digits_correct_images":sum(groups.values()),"validation_images":len(groups),"test_split_used":False,"errors":[{**m,"expected":int(t),"prediction":int(p),"score":float(prob[i,p])} for i,(t,p,m) in enumerate(zip(vlabels,pred,meta)) if t!=p]}
    (out / "digits_validation.json").write_text(json.dumps(report,indent=2))
    web=args.web_dir;web.mkdir(parents=True,exist_ok=True)
    model=model.cpu()
    torch.onnx.export(model,torch.zeros(1,1,48,32),str(web / "bp-digits.onnx"),input_names=["crops"],output_names=["logits"],dynamic_axes={"crops":{0:"batch"},"logits":{0:"batch"}},opset_version=17,dynamo=False)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
