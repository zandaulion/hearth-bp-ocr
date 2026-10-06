# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Audit label semantics and likely leakage before interpreting a benchmark."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from dataset_tools import DATA, ROOT, annotations, make_manifests


def dhash(image):
    pixels = np.asarray(image.convert("L").resize((17, 16)))
    return int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), "big")


def main():
    make_manifests()
    counts = Counter()
    classes = Counter()
    image_records = []
    exact = defaultdict(list)
    for split in ("train", "valid", "test"):
        for p in sorted((DATA / split / "images").glob("*.jpg")):
            counts[split] += 1
            classes.update(a["class"] for a in annotations(p))
            with Image.open(p) as im:
                image_records.append({"split": split, "file": p.name, "hash": dhash(im)})
            exact[hashlib.sha256(p.read_bytes()).hexdigest()].append({"split": split, "file": p.name})
    duplicates = [group for group in exact.values() if len(group) > 1]
    near = []
    for i, a in enumerate(image_records):
        for b in image_records[i+1:]:
            distance = (a["hash"] ^ b["hash"]).bit_count()
            if distance <= 10:
                near.append({"a": {k:v for k,v in a.items() if k != "hash"}, "b": {k:v for k,v in b.items() if k != "hash"}, "dhash_distance": distance})
    kaggle = ROOT / "dataset/kaggle"
    objects = Counter()
    xml_paths = list((kaggle / "Annotations").rglob("*.xml"))
    for p in xml_paths:
        objects.update(o.findtext("name") for o in ET.parse(p).getroot().findall("object"))
    report = {
        "roboflow": {"image_counts": dict(counts), "class_name_counts": dict(classes), "class_10_semantics": "Numeric row bounding boxes, based on visual inspection; NOT LCD screen boxes", "exact_duplicate_groups": duplicates, "near_duplicate_method": "256-bit difference hash, Hamming distance <=10; candidate pairs require visual confirmation", "near_duplicate_pairs": near, "cross_split_near_duplicate_pairs": sum(p["a"]["split"] != p["b"]["split"] for p in near)},
        "kaggle": {"images": len(list((kaggle / "bp monitor_images").glob("*"))), "xml_files": len(xml_paths), "object_classes": dict(objects), "reading_ground_truth_available": False},
        "curated": {"images": len(json.loads((ROOT / "dataset/ground_truth.json").read_text())), "complete_triplets": sum(r["pulse"] is not None for r in json.loads((ROOT / "dataset/ground_truth.json").read_text()))},
    }
    target = ROOT / "evaluation/artifacts/dataset_audit.json"
    target.write_text(json.dumps(report, indent=2))
    print("Class counts:", dict(classes))
    print("Exact duplicate groups:", len(duplicates))
    print("Near duplicate pairs:", len(near), "cross-split:", report["roboflow"]["cross_split_near_duplicate_pairs"])
    print("Kaggle:", report["kaggle"])


if __name__ == "__main__":
    main()
