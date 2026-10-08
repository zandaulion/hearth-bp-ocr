# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Evaluate the frozen external v2 set with the shipped staged ONNX reader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import cv2

from benchmark import summarize, wilson
from prototype.adaptive import adaptive_read
from prototype.detector import Detector
from prototype.reading import is_plausible_reading


PORTRAIT_CROPS = [
    ("center-wide", (0.15, 0.25, 0.7, 0.6)),
    ("center-tight", (0.25, 0.32, 0.6, 0.5)),
]


def portrait_fallback(detector, image, full, min_score, accept_score):
    height, width = image.shape[:2]
    if height < 1.2 * width:
        return full, []
    replays = []
    for name, fractions in PORTRAIT_CROPS:
        x = round(fractions[0] * width)
        y = round(fractions[1] * height)
        crop_width = round(fractions[2] * width)
        crop_height = round(fractions[3] * height)
        result = detector.read(
            image[y:y + crop_height, x:x + crop_width], min_score, accept_score
        )
        replays.append({"name": name, "rect": [x, y, crop_width, crop_height], "result": result})
    eligible = all(
        replay["result"]["status"] == "candidate"
        and is_plausible_reading(replay["result"]["reading"])
        for replay in replays
    )
    if eligible and replays[0]["result"]["reading"] == replays[1]["result"]["reading"]:
        selected = dict(replays[0]["result"])
        selected["score"] = min(replay["result"]["score"] for replay in replays)
        selected["cropFallback"] = {
            "method": "two-agreeing-center-crops",
            "views": [{"name": replay["name"], "rect": replay["rect"]} for replay in replays],
        }
        return selected, replays
    return full, replays


def staged_read(detector, image, min_score, accept_score):
    full = detector.read(image, min_score, accept_score)
    selected = full
    stage = "full"
    portrait_views = []
    adaptive_views = []
    if not is_plausible_reading(selected["reading"]):
        selected, portrait_views = portrait_fallback(
            detector, image, full, min_score, accept_score
        )
        if selected is not full:
            stage = "portrait"
    if not is_plausible_reading(selected["reading"]):
        adaptive, adaptive_views = adaptive_read(
            detector, image, full, min_score, accept_score
        )
        if adaptive is not full:
            selected = adaptive
            stage = "adaptive"
    return full, selected, stage, portrait_views, adaptive_views


def readable_record(item, full, selected, stage, elapsed_ms):
    expected = {key: item[key] for key in ("sys", "dia", "pulse")}
    prediction = selected["reading"]
    fields = {
        key: prediction is not None and prediction[key] == value
        for key, value in expected.items()
    }
    return {
        "id": item["id"],
        "local_file": item["local_file"],
        "source": item["source"],
        "expected": expected,
        "prediction": prediction,
        "accepted": selected["status"] == "candidate",
        "exact_triplet": all(fields.values()),
        "field_correct": fields,
        "stage": stage,
        "baseline_status": full["status"],
        "status": selected["status"],
        "score": selected["score"],
        "error": None,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def refusal_summary(records):
    total = len(records)
    safe = sum(record["prediction"] is None for record in records)
    accepted = sum(record["accepted"] for record in records)
    returned = total - safe
    return {
        "images": total,
        "safe_refusals": safe,
        "safe_refusal_rate": safe / total if total else None,
        "safe_refusal_rate_wilson_95": wilson(safe, total),
        "returned_readings": returned,
        "unsafe_candidate_outputs": accepted,
        "unsafe_candidate_rate": accepted / total if total else None,
    }


def checked_image(item):
    path = ROOT / item["local_file"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise ValueError(
            f"Frozen image checksum mismatch for {item['id']}: "
            f"expected {item['sha256']}, got {actual}"
        )
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read {item['local_file']}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth", type=Path,
        default=ROOT / "dataset/external_test_v2/readable_truth.json",
    )
    parser.add_argument(
        "--refusals", type=Path,
        default=ROOT / "dataset/external_test_v2/refusal_truth.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "prototype/reports/external_test_v2.json",
    )
    args = parser.parse_args()

    model_dir = ROOT / "prototype/web/models"
    detector_path = model_dir / "bp-detector.onnx"
    digits_path = model_dir / "bp-digits.onnx"
    config = json.loads((model_dir / "config.json").read_text())
    detector = Detector(detector_path, digit_model=digits_path)
    min_score = config["minScore"]
    accept_score = config["acceptScore"]

    baseline_readable = []
    readable = []
    baseline_refusal = []
    refusal = []
    for item in json.loads(args.truth.read_text()):
        image = checked_image(item)
        started = time.perf_counter()
        full, selected, stage, _, _ = staged_read(
            detector, image, min_score, accept_score
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        baseline_readable.append(readable_record(
            item, full, full, "full", elapsed_ms
        ))
        readable.append(readable_record(item, full, selected, stage, elapsed_ms))
        print(
            f"{item['id']}: {selected['reading']} "
            f"exact={readable[-1]['exact_triplet']} status={selected['status']} stage={stage}",
            flush=True,
        )

    for item in json.loads(args.refusals.read_text()):
        image = checked_image(item)
        started = time.perf_counter()
        full, selected, stage, _, _ = staged_read(
            detector, image, min_score, accept_score
        )
        common = {
            "id": item["id"],
            "local_file": item["local_file"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        baseline_refusal.append({
            **common,
            "prediction": full["reading"],
            "accepted": full["status"] == "candidate",
            "stage": "full",
            "status": full["status"],
            "score": full["score"],
        })
        refusal.append({
            **common,
            "prediction": selected["reading"],
            "accepted": selected["status"] == "candidate",
            "stage": stage,
            "baseline_status": full["status"],
            "status": selected["status"],
            "score": selected["score"],
        })
        print(
            f"{item['id']}: {selected['reading']} status={selected['status']} stage={stage}",
            flush=True,
        )

    accepted_readable = [record for record in readable if record["accepted"]]
    correct_accepted = sum(record["exact_triplet"] for record in accepted_readable)
    all_candidate_outputs = len(accepted_readable) + sum(r["accepted"] for r in refusal)
    report = {
        "reader": "shipped staged Python ONNX reference (full, portrait agreement, adaptive agreement)",
        "independent_held_out_claim": False,
        "reason": "Public source labels and test construction were reviewed in this development session; results are external challenge evidence, not a prospectively untouched clinical test.",
        "config": config,
        "detector_sha256": hashlib.sha256(detector_path.read_bytes()).hexdigest(),
        "digits_sha256": hashlib.sha256(digits_path.read_bytes()).hexdigest(),
        "truth_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
        "refusals_sha256": hashlib.sha256(args.refusals.read_bytes()).hexdigest(),
        "baseline_readable_summary": summarize(baseline_readable),
        "readable_summary": summarize(readable),
        "baseline_refusal_summary": refusal_summary(baseline_refusal),
        "refusal_summary": refusal_summary(refusal),
        "readable_source_summaries": {
            source: summarize([r for r in readable if r["source"] == source])
            for source in sorted({r["source"] for r in readable})
        },
        "combined_candidate_precision": (
            correct_accepted / all_candidate_outputs if all_candidate_outputs else None
        ),
        "combined_candidate_precision_wilson_95": wilson(
            correct_accepted, all_candidate_outputs
        ),
        "stage_counts": {
            stage: sum(record["stage"] == stage for record in readable + refusal)
            for stage in ("full", "portrait", "adaptive")
        },
        "readable_records": readable,
        "refusal_records": refusal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "baseline_readable_summary": report["baseline_readable_summary"],
        "readable_summary": report["readable_summary"],
        "refusal_summary": report["refusal_summary"],
        "combined_candidate_precision": report["combined_candidate_precision"],
        "stage_counts": report["stage_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
