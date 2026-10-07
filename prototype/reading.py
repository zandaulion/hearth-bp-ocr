# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Turn detector predictions into three readings without guessing missing digits."""
from itertools import combinations
import math

CLASS_NAMES = ("0", "1", "row", "2", "3", "4", "5", "6", "7", "8", "9")
FIELDS = ("sys", "dia", "pulse")


def is_plausible_reading(reading):
    return (reading is not None and all(isinstance(reading.get(k), int) for k in FIELDS)
            and 50 <= reading["sys"] <= 280 and 25 <= reading["dia"] <= 180
            and 20 <= reading["pulse"] <= 250 and reading["sys"] > reading["dia"])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return overlap / max(area_a + area_b - overlap, 1e-9)


def nms(detections, threshold=.45):
    """Suppress duplicate digit hypotheses across digit classes, separately from rows."""
    kept = []
    for det in sorted(detections, key=lambda d: d["score"], reverse=True):
        if all(not duplicates(det,old,threshold) for old in kept):
            kept.append(det)
    return kept


def duplicates(a,b,threshold=.45):
    if (a["class"]=="row")!=(b["class"]=="row"):return False
    if iou(a["box"],b["box"])>threshold:return True
    if a["class"]=="row":return False
    x1,y1,x2,y2=a["box"];u1,v1,u2,v2=b["box"]
    intersection=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    smaller=min(max(0,x2-x1)*max(0,y2-y1),max(0,u2-u1)*max(0,v2-v1))
    return smaller>0 and intersection/smaller>.75


def assemble(detections, min_score=.25, accept_score=.75):
    """Support three upright SYS/DIA/pulse rows. Scores are uncalibrated."""
    detections = nms([d for d in detections if math.isfinite(d["score"]) and d["score"] >= min_score])
    rows = sorted([d for d in detections if d["class"] == "row"], key=lambda d: d["score"], reverse=True)[:8]
    digits = [d for d in detections if d["class"] != "row"]
    candidates = []
    for row in rows:
        x1,y1,x2,y2 = row["box"]
        h = y2-y1
        group = [d for d in digits
                 if d["box"][2] >= x1-.2*h and d["box"][0] <= x2+.2*h
                 and y1-.08*h <= (d["box"][1]+d["box"][3])/2 <= y2+.08*h]
        group.sort(key=lambda d: (d["box"][0]+d["box"][2])/2)
        if len(group) not in (2,3) or any(d["class"] not in tuple(str(n) for n in range(10)) for d in group):
            continue
        text = "".join(d["class"] for d in group)
        if text[0] == "0":
            continue
        candidates.append({"value":int(text), "box":row["box"], "digits":group,
                           "score":min(row["score"], *(d["score"] for d in group)),
                           "cy":(y1+y2)/2, "cx":(x1+x2)/2, "height":h})
    valid_stacks = []
    for stack in combinations(candidates,3):
        stack = sorted(stack,key=lambda r:r["cy"])
        s,d,p = stack
        # Distinct vertical rows; no reused digit boxes or overlapping row regions.
        if any(iou(a["box"],b["box"]) > .15 for a,b in combinations(stack,2)):
            continue
        digit_ids = [id(q) for r in stack for q in r["digits"]]
        if len(set(digit_ids)) != len(digit_ids):
            continue
        if not (s["cy"] < d["cy"] < p["cy"]):
            continue
        span = p["cy"]-s["cy"]
        if min(d["cy"]-s["cy"],p["cy"]-d["cy"]) < .12*span:
            continue
        # Reject distinct side-by-side columns and very dissimilar SYS/DIA sizes.
        if abs(s["cx"]-d["cx"]) > max(s["box"][2]-s["box"][0],d["box"][2]-d["box"][0]):
            continue
        if abs(p["cx"]-d["cx"]) > max(s["box"][2]-s["box"][0],d["box"][2]-d["box"][0]):
            continue
        if min(s["height"],d["height"])/max(s["height"],d["height"]) < .45:
            continue
        valid_stacks.append(stack)
    if len(valid_stacks) != 1:
        return {"status":"retake", "reading":None, "score":None,
                "reasons":["Could not identify one unambiguous three-row reading"], "rows":[], "detections":detections}
    stack = valid_stacks[0]
    reading = dict(zip(FIELDS,[r["value"] for r in stack]))
    score = min(r["score"] for r in stack)
    reasons = []
    if not is_plausible_reading(reading):
        reasons.append("Values failed consistency checks; verify every displayed number")
    if score < accept_score:
        reasons.append("At least one detected row or digit has low confidence")
    return {"status":"review" if reasons else "candidate", "reading":reading, "score":score,
            "reasons":reasons, "rows":stack, "detections":detections}
