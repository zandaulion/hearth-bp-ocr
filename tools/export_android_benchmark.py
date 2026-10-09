# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Export the frozen public external-v2 suite for the Android ML Kit harness.

The output contains public benchmark images and per-image reference readings.
Keep it outside the repository and delete it from test devices after use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototype.adaptive import row_region
from prototype.detector import Detector
from prototype.evaluate_ablation import display_candidates, warp_quad


def write_view(destination, item_id, name, image, preprocess_ms):
    relative = Path("views") / name / f"{item_id}.jpg"
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise ValueError(f"Could not write {target}")
    return {
        "file": relative.as_posix(),
        "preprocessMs": round(preprocess_ms, 2),
    }


def derived_views(source, item_id, destination, detector, min_score, accept_score):
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not read {source}")
    started = time.perf_counter()
    result = detector.read(image, min_score, accept_score)
    detector_ms = (time.perf_counter() - started) * 1000
    views = {}

    started = time.perf_counter()
    rect = row_region(result["detections"], image.shape[1], image.shape[0])
    if rect:
        x, y, width, height = rect
        crop = image[y:y + height, x:x + width]
        views["rowCrop"] = write_view(
            destination,
            item_id,
            "row-crop",
            crop,
            detector_ms + (time.perf_counter() - started) * 1000,
        )

    started = time.perf_counter()
    candidates = display_candidates(image, result["detections"], limit=1)
    if candidates:
        warped = warp_quad(image, candidates[0][1])
        if warped is not None:
            views["rectified"] = write_view(
                destination,
                item_id,
                "rectified",
                warped,
                detector_ms + (time.perf_counter() - started) * 1000,
            )
    return views


def export_group(
    items, role, destination, detector=None, min_score=0.25, accept_score=0.75
):
    exported = []
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        source = ROOT / item["local_file"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Frozen image checksum mismatch for {item['id']}")
        suffix = source.suffix.lower() or ".jpg"
        relative = Path("images") / f"{item['id']}{suffix}"
        shutil.copy2(source, destination / relative)
        record = {
            "id": item["id"],
            "file": relative.as_posix(),
            "sha256": digest,
        }
        if role == "readable":
            record.update({field: item[field] for field in ("sys", "dia", "pulse")})
        if detector is not None:
            record["views"] = derived_views(
                source, item["id"], destination, detector, min_score, accept_score
            )
        exported.append(record)
    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--truth", type=Path,
        default=ROOT / "dataset/external_test_v2/readable_truth.json",
    )
    parser.add_argument(
        "--refusals", type=Path,
        default=ROOT / "dataset/external_test_v2/refusal_truth.json",
    )
    parser.add_argument(
        "--derived-views",
        action="store_true",
        help="Export detector-selected row crops and display rectifications.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT == output or ROOT in output.parents:
        raise ValueError("Export outside the repository so images cannot enter Git")
    output.mkdir(parents=True, exist_ok=True)
    detector = None
    min_score = 0.25
    accept_score = 0.75
    if args.derived_views:
        model_dir = ROOT / "prototype/web/models"
        config = json.loads((model_dir / "config.json").read_text())
        min_score = config["minScore"]
        accept_score = config["acceptScore"]
        detector = Detector(model_dir / "bp-detector.onnx")
    readable = export_group(
        json.loads(args.truth.read_text()), "readable", output,
        detector, min_score, accept_score,
    )
    refusal = export_group(
        json.loads(args.refusals.read_text()), "refusal", output,
        detector, min_score, accept_score,
    )
    manifest = {
        "benchmark": "external-v2 consumed regression suite",
        "independentHeldOutClaim": False,
        "selectionAllowed": False,
        "derivedViews": args.derived_views,
        "readable": readable,
        "refusal": refusal,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(readable)} readable and {len(refusal)} refusal cases to {output}")


if __name__ == "__main__":
    main()
