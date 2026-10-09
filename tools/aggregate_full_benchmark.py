#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Aggregate v2 and v3 reports without publishing per-image records.

The input reports remain ignored because they contain predictions and reference
readings. The generated report contains aggregate metrics and input checksums
only. Combining consumed suites increases the regression sample but does not
create new independent held-out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from benchmark import summarize, wilson


FIELDS = ("sys", "dia", "pulse")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 2)


def ensure_unique(records: list[dict], label: str) -> None:
    keys = [(record.get("role", "readable"), record["id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate case ids in {label}")


def refusal_summary(records: list[dict]) -> dict:
    safe = sum(record["prediction"] is None for record in records)
    accepted = sum(record["accepted"] for record in records)
    times = [record["elapsed_ms"] for record in records]
    return {
        "images": len(records),
        "safe_refusals": safe,
        "safe_refusal_rate": safe / len(records) if records else None,
        "safe_refusal_rate_wilson_95": wilson(safe, len(records)),
        "returned_readings": len(records) - safe,
        "unsafe_candidate_outputs": accepted,
        "latency_ms": {
            "median": percentile(times, 0.5),
            "p95": percentile(times, 0.95),
        },
    }


def aggregate_hearth(paths: list[Path]) -> tuple[dict, dict[str, dict[str, bool]]]:
    reports = [load(path) for path in paths]
    variants = set(reports[0]["pipelines"])
    if any(set(report["pipelines"]) != variants for report in reports[1:]):
        raise ValueError("Hearth reports do not contain identical pipeline sets")
    output = {}
    exact_by_variant = {}
    for name in sorted(variants):
        readable = sum(
            (report["pipelines"][name]["readableRecords"] for report in reports), []
        )
        refusals = sum(
            (report["pipelines"][name]["refusalRecords"] for report in reports), []
        )
        ensure_unique(readable + refusals, f"Hearth {name}")
        summary = summarize(readable)
        accepted = [record for record in readable if record["accepted"]]
        summary["exact_accepted_triplets"] = sum(
            record["exact_triplet"] is True for record in accepted
        )
        output[name] = {
            "readable": summary,
            "refusal": refusal_summary(refusals),
            "mean_passes": round(
                sum(record["passes"] for record in readable + refusals)
                / len(readable + refusals),
                3,
            ),
        }
        exact_by_variant[name] = {
            record["id"]: record["exact_triplet"] is True for record in readable
        }
    return output, exact_by_variant


def mlkit_summary(records: list[dict]) -> dict:
    readable = [record for record in records if record["role"] == "readable"]
    refusals = [record for record in records if record["role"] == "refusal"]
    available = [record for record in readable if record.get("available", True)]
    accepted = [record for record in readable if record["appAccepted"]]
    complete = [record for record in readable if record["complete"]]
    exact = sum(record["exactTriplet"] is True for record in readable)
    exact_accepted = sum(record["exactTriplet"] is True for record in accepted)
    exact_complete = sum(record["exactTriplet"] is True for record in complete)
    return {
        "readable_images": len(readable),
        "available_images": len(available),
        "exact_triplets": exact,
        "exact_accuracy": exact / len(readable) if readable else None,
        "exact_accuracy_wilson_95": wilson(exact, len(readable)),
        "app_accepted": len(accepted),
        "exact_app_accepted": exact_accepted,
        "app_accepted_precision": exact_accepted / len(accepted) if accepted else None,
        "app_accepted_coverage": len(accepted) / len(readable) if readable else None,
        "complete_outputs": len(complete),
        "exact_complete_outputs": exact_complete,
        "complete_output_precision": exact_complete / len(complete) if complete else None,
        "fields_correct": {
            field: sum(record["fields"][field] is True for record in readable)
            for field in FIELDS
        },
        "refusal_images": len(refusals),
        "safe_refusals": sum(not record["appAccepted"] for record in refusals),
        "strict_no_reading": sum(record["prediction"] is None for record in refusals),
        "latency_ms": {
            "median": percentile([record["elapsedMs"] for record in available], 0.5),
            "p95": percentile([record["elapsedMs"] for record in available], 0.95),
        },
        "source_preprocess_ms": {
            "median": percentile([record.get("preprocessMs", 0) for record in available], 0.5),
            "p95": percentile([record.get("preprocessMs", 0) for record in available], 0.95),
        },
    }


def aggregate_mlkit(paths: list[Path]) -> dict:
    reports = [load(path) for path in paths]
    variants = set(reports[0]["variants"])
    if any(set(report["variants"]) != variants for report in reports[1:]):
        raise ValueError("ML Kit reports do not contain identical variant sets")
    output = {}
    for name in sorted(variants):
        records = sum(
            (report["variants"][name]["records"] for report in reports), []
        )
        ensure_unique(records, f"ML Kit {name}")
        output[name] = mlkit_summary(records)
    return output


def aggregate_gemini(paths: list[Path]) -> tuple[dict, dict[str, bool]]:
    reports = [load(path) for path in paths]
    records = []
    for report in reports:
        for record in report["records"]:
            copied = dict(record)
            copied.setdefault("role", "readable")
            records.append(copied)
    ensure_unique(records, "Gemini")
    readable = [record for record in records if record["role"] == "readable"]
    refusals = [record for record in records if record["role"] == "refusal"]
    accepted = [record for record in readable if record["appAccepted"]]
    complete = [record for record in readable if record["complete"]]
    exact = sum(record["exactTriplet"] is True for record in readable)
    exact_accepted = sum(record["exactTriplet"] is True for record in accepted)
    exact_complete = sum(record["exactTriplet"] is True for record in complete)
    strict_no_reading = sum(
        record.get("prediction") is None
        or all(record["prediction"].get(field) is None for field in FIELDS)
        for record in refusals
    )
    usage_keys = ("promptTokens", "candidateTokens", "thoughtsTokens", "totalTokens")
    summary = {
        "readable_images": len(readable),
        "exact_triplets": exact,
        "exact_accuracy": exact / len(readable),
        "exact_accuracy_wilson_95": wilson(exact, len(readable)),
        "app_accepted": len(accepted),
        "exact_app_accepted": exact_accepted,
        "app_accepted_precision": exact_accepted / len(accepted) if accepted else None,
        "app_accepted_coverage": len(accepted) / len(readable),
        "complete_outputs": len(complete),
        "exact_complete_outputs": exact_complete,
        "complete_output_precision": exact_complete / len(complete) if complete else None,
        "fields_correct": {
            field: sum(record["fields"][field] is True for record in readable)
            for field in FIELDS
        },
        "refusal_images": len(refusals),
        "safe_by_app_rule": sum(not record["appAccepted"] for record in refusals),
        "strict_no_reading": strict_no_reading,
        "latency_ms": {
            "median": percentile([record["elapsedMs"] for record in records], 0.5),
            "p95": percentile([record["elapsedMs"] for record in records], 0.95),
        },
        "usage": {
            key: sum(report["summary"]["usage"][key] for report in reports)
            for key in usage_keys
        },
        "errors": sum(report["summary"]["errors"] for report in reports),
        "retries": sum(report["summary"]["retries"] for report in reports),
    }
    return summary, {
        record["id"]: record["exactTriplet"] is True for record in readable
    }


def paired(exact_left: dict[str, bool], exact_right: dict[str, bool]) -> dict:
    if set(exact_left) != set(exact_right):
        raise ValueError("Paired systems do not contain the same readable case ids")
    return {
        "both_exact": sum(exact_left[key] and exact_right[key] for key in exact_left),
        "left_only_exact": sum(exact_left[key] and not exact_right[key] for key in exact_left),
        "right_only_exact": sum(not exact_left[key] and exact_right[key] for key in exact_left),
        "neither_exact": sum(not exact_left[key] and not exact_right[key] for key in exact_left),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hearth", type=Path, nargs="+",
        default=[
            ROOT / "prototype/reports/ablation_v1.json",
            ROOT / "prototype/reports/ablation_v3.json",
        ],
    )
    parser.add_argument(
        "--mlkit", type=Path, nargs="+",
        default=[
            ROOT / "prototype/reports/mlkit_external_v2_a52.json",
            ROOT / "prototype/reports/mlkit_external_v3_a52.json",
        ],
    )
    parser.add_argument(
        "--gemini", type=Path, nargs="+",
        default=[
            ROOT / "prototype/reports/gemini-external-v2.json",
            ROOT / "prototype/reports/gemini-external-v3.json",
        ],
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "prototype/reports/full_consumed_benchmark.json",
    )
    args = parser.parse_args()

    hearth, hearth_exact = aggregate_hearth(args.hearth)
    mlkit = aggregate_mlkit(args.mlkit)
    gemini, gemini_exact = aggregate_gemini(args.gemini)
    first_hearth = next(iter(hearth.values()))
    readable_images = first_hearth["readable"]["images"]
    refusal_images = first_hearth["refusal"]["images"]
    report = {
        "benchmark": "external-v2 + external-v3 combined consumed regression suite",
        "independent_held_out_claim": False,
        "selection_allowed_for_independent_claim": False,
        "readable_images": readable_images,
        "refusal_images": refusal_images,
        "latency_note": "Combined observations from separate v2 and v3 runs in different execution environments.",
        "input_sha256": {
            "hearth": {path.name: sha256(path) for path in args.hearth},
            "mlkit": {path.name: sha256(path) for path in args.mlkit},
            "gemini": {path.name: sha256(path) for path in args.gemini},
        },
        "hearth": hearth,
        "mlkit": mlkit,
        "gemini": gemini,
        "paired_hearth_vs_current": {
            name: paired(exact, hearth_exact["current-staged"])
            for name, exact in hearth_exact.items()
        },
        "paired_hearth_vs_gemini": {
            name: paired(exact, gemini_exact) for name, exact in hearth_exact.items()
        },
    }
    if any(
        pipeline["readable"]["images"] != report["readable_images"]
        or pipeline["refusal"]["images"] != report["refusal_images"]
        for pipeline in hearth.values()
    ):
        raise ValueError("Unexpected Hearth combined case counts")
    if (
        gemini["readable_images"] != readable_images
        or gemini["refusal_images"] != refusal_images
    ):
        raise ValueError("Unexpected Gemini combined case counts")
    if any(
        variant["readable_images"] != readable_images
        or variant["refusal_images"] != refusal_images
        for variant in mlkit.values()
    ):
        raise ValueError("Unexpected ML Kit combined case counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
