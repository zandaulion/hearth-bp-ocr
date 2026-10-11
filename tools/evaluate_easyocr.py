#!/usr/bin/env python3
"""Evaluate EasyOCR with bounded three-row post-processing.

This is a conventional scene-text OCR baseline. Recognition is restricted to
digits, and detected tokens are assembled by image geometry. The parser accepts
only one unambiguous SYS/DIA/pulse triplet and never changes a recognized digit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_teleocr import FIELDS, percentile, plausible


def geometry_reading(results: list, minimum_confidence: float) -> tuple[dict | None, str, list]:
    tokens = []
    for box, text, confidence in results:
        digits = "".join(character for character in text if character.isdigit())
        if not digits or confidence < minimum_confidence:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        tokens.append(
            {
                "text": digits,
                "confidence": float(confidence),
                "left": min(xs),
                "center_y": (min(ys) + max(ys)) / 2,
                "height": max(ys) - min(ys),
            }
        )
    tokens.sort(key=lambda token: (token["center_y"], token["left"]))

    rows: list[list[dict]] = []
    for token in tokens:
        match = None
        for row in rows:
            center = statistics.mean(item["center_y"] for item in row)
            height = statistics.median(item["height"] for item in row)
            if abs(token["center_y"] - center) <= 0.6 * max(height, token["height"]):
                match = row
                break
        if match is None:
            match = []
            rows.append(match)
        match.append(token)

    values = []
    serialized_rows = []
    for row in rows:
        row.sort(key=lambda token: token["left"])
        text = "".join(token["text"] for token in row)
        serialized_rows.append({"text": text, "tokens": row})
        if 2 <= len(text) <= 3:
            values.append(int(text))

    candidates: set[tuple[int, int, int]] = set()
    for triple in itertools.combinations(values, 3):
        if plausible(dict(zip(FIELDS, triple))):
            candidates.add(triple)
    if len(candidates) == 1:
        return dict(zip(FIELDS, next(iter(candidates)))), "unique-bounded-row-stack", serialized_rows
    return None, "ambiguous" if candidates else "no-plausible-triplet", serialized_rows


def load_frozen(paths: list[Path], expected_refusal: bool) -> list[dict]:
    cases = []
    for path in paths:
        for item in json.loads(path.read_text()):
            image = Path(item["local_file"])
            if hashlib.sha256(image.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"checksum mismatch for {item['id']}: {image}")
            cases.append(
                {
                    **item,
                    "image": image,
                    "available": True,
                    "expected_refusal": expected_refusal,
                }
            )
    return cases


def load_bundles(paths: list[Path], view: str) -> list[dict]:
    cases = []
    for path in paths:
        manifest = json.loads(path.read_text())
        for role in ("readable", "refusal"):
            for item in manifest[role]:
                view_item = item.get("views", {}).get(view)
                image = path.parent / view_item["file"] if view_item else None
                cases.append(
                    {
                        **item,
                        "image": image,
                        "available": image is not None and image.exists(),
                        "expected_refusal": role == "refusal",
                    }
                )
    return cases


def summarize(records: list[dict], model_load_ms: float) -> dict:
    readable = [record for record in records if not record["expected_refusal"]]
    negatives = [record for record in records if record["expected_refusal"]]
    returned = [record for record in readable if record["reading"] is not None]
    exact = [record for record in returned if record["exact_triplet"]]
    times = [record["elapsed_ms"] for record in records if record["available"]]
    return {
        "readable_images": len(readable),
        "negative_images": len(negatives),
        "available_views": sum(record["available"] for record in records),
        "available_readable_views": sum(record["available"] for record in readable),
        "available_negative_views": sum(record["available"] for record in negatives),
        "exact_triplets": len(exact),
        "exact_triplet_accuracy": len(exact) / len(readable) if readable else None,
        "returned_readings": len(returned),
        "returned_precision": len(exact) / len(returned) if returned else None,
        "coverage": len(returned) / len(readable) if readable else None,
        "correct_fields": {
            field: sum(record["field_correct"][field] for record in readable)
            for field in FIELDS
        },
        "safe_negatives": sum(record["reading"] is None for record in negatives),
        "median_ms": statistics.median(times) if times else None,
        "p95_ms": percentile(times, 0.95),
        "model_load_ms": model_load_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, nargs="+")
    parser.add_argument("--refusals", type=Path, nargs="*", default=[])
    parser.add_argument("--bundle-manifest", type=Path, nargs="+")
    parser.add_argument("--view", choices=("original", "rowCrop", "rectified"), default="original")
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.view == "original":
        if not args.truth:
            parser.error("--truth is required for the original view")
        cases = load_frozen(args.truth, False) + load_frozen(args.refusals, True)
    else:
        if not args.bundle_manifest:
            parser.error("--bundle-manifest is required for a derived view")
        cases = load_bundles(args.bundle_manifest, args.view)
    if args.limit is not None:
        cases = cases[: args.limit]

    import easyocr

    load_started = time.perf_counter()
    reader = easyocr.Reader(
        ["en"],
        gpu=True,
        model_storage_directory=str(args.model_dir) if args.model_dir else None,
        download_enabled=True,
    )
    model_load_ms = (time.perf_counter() - load_started) * 1000

    records = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        if case["available"]:
            raw = reader.readtext(
                str(case["image"]),
                detail=1,
                paragraph=False,
                decoder="greedy",
                allowlist="0123456789",
            )
            reading, method, rows = geometry_reading(raw, args.minimum_confidence)
        else:
            raw, reading, method, rows = [], None, "view-unavailable", []
        elapsed_ms = (time.perf_counter() - started) * 1000
        expected = None if case["expected_refusal"] else {field: case[field] for field in FIELDS}
        records.append(
            {
                "id": case["id"],
                "expected_refusal": case["expected_refusal"],
                "available": case["available"],
                "expected": expected,
                "reading": reading,
                "recognized": [
                    {"text": text, "confidence": float(confidence)}
                    for _box, text, confidence in raw
                ],
                "assembled_rows": rows,
                "parse_method": method,
                "exact_triplet": reading == expected if expected else False,
                "field_correct": {
                    field: bool(reading and expected and reading[field] == expected[field])
                    for field in FIELDS
                },
                "elapsed_ms": elapsed_ms,
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(cases)}",
                    "id": case["id"],
                    "reading": reading,
                    "recognized": [text for _box, text, _confidence in raw],
                    "elapsed_ms": round(elapsed_ms, 1),
                }
            ),
            flush=True,
        )

    summary = summarize(records, model_load_ms)
    report = {
        "system": "EasyOCR 1.7.2 English detector/recognizer",
        "view": args.view,
        "allowlist": "digits only",
        "minimum_confidence": args.minimum_confidence,
        "parser": "geometry rows, then exactly one bounded triplet; no correction",
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.aggregate_output:
        args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_output.write_text(
            json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2)
            + "\n"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
