# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Read this export by class *name*, not class index."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/roboflow_bp_display"
# data.yaml: the string '10' is class index 2. Inspection shows it encloses
# each numeric row, not the whole LCD as dataset/README.md claims.
CLASS_NAMES = ("0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9")
def validate_class_names():
    export_names = tuple(re.findall(r"^- '([0-9]+)'\s*$", (DATA / "data.yaml").read_text(), re.MULTILINE))
    if export_names != CLASS_NAMES:
        raise ValueError("The export class order changed; review the dataset parser")


def annotations(image_path):
    validate_class_names()
    label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
    result = []
    for line in label_path.read_text().splitlines():
        cid, cx, cy, w, h = map(float, line.split())
        result.append({"class": CLASS_NAMES[int(cid)], "box": [cx-w/2, cy-h/2, cx+w/2, cy+h/2], "center": [cx, cy]})
    return result


def rows_from_annotations(labels):
    rows = sorted([a for a in labels if a["class"] == "10"], key=lambda a: a["center"][1])
    digits = [a for a in labels if a["class"] != "10"]
    grouped = [[] for _ in rows]
    unassigned = []
    for digit in digits:
        cx, cy = digit["center"]
        matches = [i for i, row in enumerate(rows) if row["box"][0] <= cx <= row["box"][2] and row["box"][1] <= cy <= row["box"][3]]
        if len(matches) == 1:
            grouped[matches[0]].append(digit)
        else:
            unassigned.append(digit)
    for group in grouped:
        group.sort(key=lambda d: d["center"][0])
    values = [int("".join(d["class"] for d in group)) if group else None for group in grouped]
    valid = len(rows) == 3 and not unassigned and all(2 <= len(group) <= 3 for group in grouped)
    return rows, grouped, values, valid


def make_manifests():
    out = ROOT / "evaluation/artifacts"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        result, rejected = [], []
        for path in sorted((DATA / split / "images").glob("*.jpg")):
            rows, grouped, values, valid = rows_from_annotations(annotations(path))
            if not valid:
                rejected.append({"image": path.name, "rows": len(rows), "digit_counts": [len(g) for g in grouped]})
                continue
            result.append({"id": f'{split}_{path.stem}', "local_file": path.relative_to(ROOT).as_posix(), **dict(zip(("sys", "dia", "pulse"), values)), "answer_source": "YOLO digit annotations; numeric rows assigned top to bottom; not independently verified"})
        (out / f"roboflow_{split}_truth.json").write_text(json.dumps(result, indent=2))
        (out / f"roboflow_{split}_ambiguous.json").write_text(json.dumps(rejected, indent=2))
        print(split, "usable:", len(result), "ambiguous:", len(rejected), rejected[:5])


if __name__ == "__main__":
    make_manifests()
