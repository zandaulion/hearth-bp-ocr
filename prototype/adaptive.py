# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Locate numeric rows and normalize uneven illumination after a refusal."""
import math
import cv2
import numpy as np


def row_region(detections,width,height):
    rows=[d for d in detections if d['class']=='row' and d['score']>=.25]
    if not 2<=len(rows)<=3:return None
    rows.sort(key=lambda d:(d['box'][1]+d['box'][3])/2)
    boxes=np.array([d['box'] for d in rows]);heights=boxes[:,3]-boxes[:,1]
    widths=boxes[:,2]-boxes[:,0];centers=(boxes[:,:2]+boxes[:,2:])/2
    if min(heights[:2])/max(heights[:2])<.45:return None
    if max(centers[:,0])-min(centers[:,0])>max(widths):return None
    if any(centers[i+1,1]-centers[i,1]<.5*min(heights[i:i+2]) for i in range(len(rows)-1)):return None
    x1,y1=boxes[:,:2].min(0);x2,y2=boxes[:,2:].max(0)
    rw=x2-x1;rh=float(np.median(heights[:2]))
    left=max(0,math.floor(x1-.2*rw));top=max(0,math.floor(y1-.25*rh))
    right=min(width,math.ceil(x2+.2*rw));bottom=min(height,math.ceil(max(y2+.3*rh,y1+3.3*rh)))
    if right-left<40 or bottom-top<60:return None
    return [left,top,right-left,bottom-top]


def normalize_light(image,fraction):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    radius=max(3,math.floor(min(gray.shape)*fraction+.5))
    bg=gray.astype(np.float64)
    for _ in range(3):bg=cv2.boxFilter(bg,-1,(2*radius+1,2*radius+1),borderType=cv2.BORDER_REPLICATE)
    flat=np.clip(np.floor(gray*150./np.maximum(bg,1)+.5),0,255).astype(np.uint8)
    return cv2.cvtColor(flat,cv2.COLOR_GRAY2BGR)


def adaptive_read(detector,image,full,min_score=.2,accept_score=.25):
    if full['reading'] is not None:return full,[]
    h,w=image.shape[:2];rect=row_region(full['detections'],w,h)
    if rect is None:return full,[]
    x,y,cw,ch=rect;crop=image[y:y+ch,x:x+cw];views=[]
    for fraction in [.04,.05]:
        result=detector.read(normalize_light(crop,fraction),min_score,accept_score)
        views.append({'fraction':fraction,'rect':rect,'result':result})
    def plausible(result):
        r=result['reading']
        return r is not None and result['status'] in ('candidate','review') and all(isinstance(v,int) for v in r.values()) and 50<=r['sys']<=280 and 25<=r['dia']<=180 and 20<=r['pulse']<=250 and r['sys']>r['dia']
    if all(plausible(v['result']) for v in views) and views[0]['result']['reading']==views[1]['result']['reading']:
        def move(d):return {**d,'box':[v+(x if i%2==0 else y) for i,v in enumerate(d['box'])]}
        selected=views[0]['result']
        review=any(v['result']['status']=='review' for v in views)
        return {**selected,'status':'review' if review else 'candidate',
                'reasons':['Enhanced views agree, but at least one row or digit has low confidence; check every value.'] if review else [],
                'score':min(v['result']['score'] for v in views),
                'rows':[{**move(r),'cx':r['cx']+x,'cy':r['cy']+y,'digits':[move(d) for d in r['digits']]} for r in selected['rows']],
                'detections':[move(d) for d in selected['detections']],
                'adaptive':{'rect':rect,'fractions':[.04,.05]}},views
    return full,views
