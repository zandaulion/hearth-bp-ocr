# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Shared complete-reading metrics; errors and refusals count as misses."""
import math

FIELDS = ("sys", "dia", "pulse")


def wilson(successes, total):
    if not total:
        return None
    z = 1.96
    p = successes / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total)) / d
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [round(center-half, 4), round(center+half, 4)]


def summarize(records):
    complete = [r for r in records if all(r["expected"].get(k) is not None for k in FIELDS)]
    accepted = [r for r in complete if r["accepted"]]
    correct = sum(r["exact_triplet"] is True for r in complete)
    correct_accepted = sum(r["exact_triplet"] is True for r in accepted)
    times = sorted(r["elapsed_ms"] for r in records)
    fields = {}
    for k in FIELDS:
        eligible = sum(r["expected"].get(k) is not None for r in records)
        correct_field = sum(r["field_correct"][k] is True for r in records)
        fields[k] = {"eligible": eligible, "correct": correct_field, "accuracy": correct_field / eligible if eligible else None}
    return {
        "images": len(records),
        "returned_readings": sum(r["prediction"] is not None for r in records),
        "errors": sum(r["error"] is not None for r in records),
        "complete_ground_truth_images": len(complete),
        "correct_triplets": correct,
        "exact_triplet_accuracy": correct / len(complete) if complete else None,
        "exact_triplet_accuracy_wilson_95": wilson(correct, len(complete)),
        "accepted_triplets": len(accepted),
        "accepted_coverage": len(accepted) / len(complete) if complete else None,
        "accepted_precision": correct_accepted / len(accepted) if accepted else None,
        "accepted_precision_wilson_95": wilson(correct_accepted, len(accepted)),
        "fields": fields,
        "latency_ms": {"median": times[len(times)//2] if times else None, "p95": times[min(len(times)-1, math.ceil(.95*len(times))-1)] if times else None},
    }
