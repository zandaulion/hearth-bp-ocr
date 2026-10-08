#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Freeze the independent external v2 readable and refusal manifests.

This script never runs model inference. It derives candidate readings from the
source YOLO annotations, applies a manually reviewed exclusion list, and writes
checksummed manifests so evaluation cannot silently change the answer key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROBOFLOW = REPO / "dataset/roboflow_naphop_external_v2"
WIKIMEDIA = REPO / "dataset/wikimedia_bp_test_v2_candidates"
WIKIMEDIA_READABLE_TRUTH = WIKIMEDIA / "manual_readable_truth.json"
OUTPUT = REPO / "dataset/external_test_v2"

CLASS_NAMES = ["0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9"]
ROW_CLASS = 2

# One-based indices from the frozen, lexicographically sorted Roboflow v9 test
# export. These were excluded before inference after visual and duplicate review.
READABLE_EXCLUSIONS = {
    6,  # annotation includes an unrelated display digit
    22,  # annotation includes a date digit
    24,  # same source image as 7
    63,  # annotation includes an unrelated display digit
    76,  # annotation includes unrelated display digits
    79,  # near duplicate of 78
    87,  # hospital vital-sign monitor, outside home-monitor scope
    91, 94, 96, 97, 101, 104, 105,  # repeated Omron Slim1 source/session
    107,  # near duplicate of 106
    110,  # near duplicate of 62
    116,  # annotation includes an unrelated display digit
    117,  # near duplicate of 115
    133,  # only two target rows
    143,  # only two target rows
    149,  # near duplicate of 147
}

# Intentionally contains analog, powered-off, cuff-only, and disassembled
# devices. Images with identifiable people and unrelated search noise are not
# used. Index 29 is excluded because it overlaps the earlier v1 external set.
REFUSAL_INDICES = [
    1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
    22, 23, 26, 36, 39, 46, 47, 48,
]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_reading(label_path: Path) -> tuple[int, int, int] | None:
    boxes = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        cls = int(fields[0])
        x, y, width, height = map(float, fields[1:])
        boxes.append((cls, x, y, width, height))

    rows = sorted((box for box in boxes if box[0] == ROW_CLASS), key=lambda b: b[2])
    digits = [box for box in boxes if box[0] != ROW_CLASS]
    if len(rows) != 3:
        return None

    values = []
    for _, row_x, row_y, row_width, row_height in rows:
        left, right = row_x - row_width / 2, row_x + row_width / 2
        top, bottom = row_y - row_height / 2, row_y + row_height / 2
        members = [
            box for box in digits
            if left <= box[1] <= right and top <= box[2] <= bottom
        ]
        members.sort(key=lambda box: box[1])
        if not members:
            return None
        values.append(int("".join(CLASS_NAMES[box[0]] for box in members)))
    return tuple(values)


def roboflow_rows() -> list[dict]:
    image_dir = ROBOFLOW / "test/images"
    label_dir = ROBOFLOW / "test/labels"
    images = sorted(path for path in image_dir.iterdir() if path.is_file())
    selected = []
    for index, image_path in enumerate(images, 1):
        if index > 150 or index in READABLE_EXCLUSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        reading = parse_reading(label_path)
        if reading is None:
            continue
        systolic, diastolic, pulse = reading
        if not (50 <= systolic <= 260 and 30 <= diastolic <= 180 and 30 <= pulse <= 220):
            continue
        selected.append(
            {
                "id": f"external_v2_readable_{len(selected) + 1:03d}",
                "local_file": str(image_path.relative_to(REPO)),
                "sha256": sha256(image_path),
                "sys": systolic,
                "dia": diastolic,
                "pulse": pulse,
                "source": "Roboflow Universe: naphop/blood-pressure-monitor-digit-reader v9 test split",
                "source_url": "https://universe.roboflow.com/naphop/blood-pressure-monitor-digit-reader/dataset/9",
                "license": "CC BY 4.0",
                "source_index": index,
                "answer_source": "Source digit/row annotations, visually cross-checked before model inference",
            }
        )
        if len(selected) == 90:
            break
    if len(selected) != 90:
        raise RuntimeError(f"Expected 90 Roboflow readable cases, found {len(selected)}")
    return selected


def wikimedia_readable_rows(start: int) -> list[dict]:
    manifest = json.loads((WIKIMEDIA / "manifest.json").read_text())
    manual_truth = json.loads(WIKIMEDIA_READABLE_TRUTH.read_text())
    source_rows = manifest["images"]
    rows = []
    for sequence, truth in enumerate(manual_truth, start):
        index = truth["source_index"]
        source = source_rows[index - 1]
        image_path = WIKIMEDIA / source["local_file"]
        rows.append(
            {
                "id": f"external_v2_readable_{sequence:03d}",
                "local_file": str(image_path.relative_to(REPO)),
                "sha256": sha256(image_path),
                "sys": truth["sys"],
                "dia": truth["dia"],
                "pulse": truth["pulse"],
                "source": "Wikimedia Commons",
                "source_page": source["description_url"],
                "license": source["license"],
                "license_url": source["license_url"],
                "source_index": index,
                "answer_source": "Manual transcription from visible SYS, DIA, and pulse rows before model inference",
            }
        )
    return rows


def wikimedia_refusal_rows() -> list[dict]:
    manifest = json.loads((WIKIMEDIA / "manifest.json").read_text())
    source_rows = manifest["images"]
    rows = []
    for sequence, index in enumerate(REFUSAL_INDICES, 1):
        source = source_rows[index - 1]
        image_path = WIKIMEDIA / source["local_file"]
        rows.append(
            {
                "id": f"external_v2_refusal_{sequence:03d}",
                "local_file": str(image_path.relative_to(REPO)),
                "sha256": sha256(image_path),
                "expected": "refuse",
                "reason": "No readable three-row digital blood-pressure result",
                "source": "Wikimedia Commons",
                "source_page": source["description_url"],
                "license": source["license"],
                "license_url": source["license_url"],
                "source_index": index,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the frozen manifests")
    args = parser.parse_args()

    readable = roboflow_rows()
    readable.extend(wikimedia_readable_rows(len(readable) + 1))
    refusal = wikimedia_refusal_rows()
    print(f"readable={len(readable)} refusal={len(refusal)}")
    print(
        f"roboflow_source_indices={readable[0]['source_index']}.."
        f"{readable[89]['source_index']}"
    )
    print(f"wikimedia_readable_truth={WIKIMEDIA_READABLE_TRUTH.relative_to(REPO)}")

    if not args.write:
        print("dry_run=true (pass --write to freeze manifests)")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    (OUTPUT / "readable_truth.json").write_text(
        json.dumps(readable, indent=2) + "\n"
    )
    (OUTPUT / "refusal_truth.json").write_text(
        json.dumps(refusal, indent=2) + "\n"
    )
    metadata = {
        "schema_version": 1,
        "frozen_at": generated_at,
        "role": "test-only consumed regression benchmark",
        "permitted_uses": ["evaluation", "regression testing", "error analysis"],
        "prohibited_uses": ["training", "validation", "threshold calibration", "augmentation"],
        "readable_cases": len(readable),
        "refusal_cases": len(refusal),
        "readable_sources": {"roboflow_v9_test": 90, "wikimedia_commons": 10},
        "selection": "Manual scope/quality review, source-session deduplication, then deterministic first 100 eligible cases",
        "leakage_audit": {
            "reference_images": 309,
            "candidate_images": 215,
            "exact_matches": 0,
            "near_matches_excluded": 13,
            "dhash256_threshold": 18,
            "note": "All 13 Roboflow near matches were source indices 174-186 and are outside the selected range. No Wikimedia candidate was within threshold.",
        },
        "limitations": [
            "Readable answers originate from the source dataset annotations and were visually cross-checked, not independently double-transcribed.",
            "The readable set is dominated by one public source dataset and is a challenge set, not clinical performance evidence.",
            "The refusal set is intentionally heterogeneous and is scored separately from exact-triplet accuracy.",
        ],
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote={OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
