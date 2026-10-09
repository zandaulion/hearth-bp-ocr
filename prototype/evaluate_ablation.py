# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Measure OCR pipeline changes independently on the consumed external-v2 suite.

This is an experiment runner, not a release selector. The suite is permanently
test-only and must not be used to tune thresholds, preprocessing parameters or
model weights. Detailed records are written below the ignored reports folder.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import cv2
import numpy as np

from benchmark import summarize, wilson
from prototype.adaptive import normalize_light, row_region
from prototype.detector import Detector, DigitRecognizer
from prototype.digit_model import digit_tensor
from prototype.evaluate_external_v2 import portrait_fallback
from prototype.reading import is_plausible_reading


FIELDS = ("sys", "dia", "pulse")


def checked_image(item):
    path = ROOT / item["local_file"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise ValueError(f"Frozen image checksum mismatch for {item['id']}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read {item['local_file']}")
    return image


def result_key(result):
    reading = result.get("reading")
    if not is_plausible_reading(reading):
        return None
    return tuple(reading[field] for field in FIELDS)


def select_unopposed_consensus(original, views, method, minimum_votes=2):
    """Select one reading only when no plausible transformed view contradicts it."""
    plausible = [(name, result) for name, result in views if result_key(result)]
    if not plausible:
        return original
    counts = Counter(result_key(result) for _, result in plausible)
    if len(counts) != 1:
        return original
    key, votes = counts.most_common(1)[0]
    if votes < minimum_votes:
        return original
    supporters = [(name, result) for name, result in plausible if result_key(result) == key]
    selected = dict(supporters[0][1])
    any_review = any(result["status"] != "candidate" for _, result in supporters)
    selected["status"] = "review" if any_review else "candidate"
    selected["score"] = min(result["score"] for _, result in supporters)
    selected["reasons"] = (
        ["Experimental views agree, but at least one view has low confidence"]
        if any_review else []
    )
    selected["ablationFallback"] = {
        "method": method,
        "supportingViews": [name for name, _ in supporters],
        "votes": votes,
    }
    return selected


def current_pipeline(detector, image, min_score, accept_score):
    full = detector.read(image, min_score, accept_score)
    selected = full
    passes = 1
    stage = "full"
    if not is_plausible_reading(selected["reading"]):
        selected, portrait_views = portrait_fallback(
            detector, image, full, min_score, accept_score
        )
        passes += len(portrait_views)
        if selected is not full:
            stage = "portrait"
    if not is_plausible_reading(selected["reading"]):
        rect = row_region(full["detections"], image.shape[1], image.shape[0])
        if rect:
            x, y, width, height = rect
            crop = image[y:y + height, x:x + width]
            views = [
                (f"local-{fraction:.2f}", detector.read(
                    normalize_light(crop, fraction), min_score, accept_score
                ))
                for fraction in (0.04, 0.05)
            ]
            passes += len(views)
            adaptive = select_unopposed_consensus(full, views, "current-local-normalization")
            if adaptive is not full:
                selected = adaptive
                stage = "adaptive"
    return selected, {"stage": stage, "passes": passes}


def expanded_normalization_fallback(detector, image, original, min_score, accept_score):
    rect = row_region(original["detections"], image.shape[1], image.shape[0])
    if not rect:
        return original, [], None
    x, y, width, height = rect
    crop = image[y:y + height, x:x + width]
    views = []
    for fraction in (0.03, 0.04, 0.05, 0.06):
        views.append((
            f"local-{fraction:.2f}",
            detector.read(normalize_light(crop, fraction), min_score, accept_score),
        ))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for clip in (1.5, 2.5):
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
        views.append((
            f"clahe-{clip:.1f}",
            detector.read(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), min_score, accept_score),
        ))
    selected = select_unopposed_consensus(
        original, views, "expanded-normalization-consensus"
    )
    return selected, views, rect


def order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    totals = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(totals)]
    ordered[2] = points[np.argmax(totals)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def warp_quad(image, points):
    top_left, top_right, bottom_right, bottom_left = order_quad(points)
    width = round(max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    ))
    height = round(max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    ))
    if width < 80 or height < 80:
        return None
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left]), destination
    )
    return cv2.warpPerspective(image, matrix, (width, height))


def display_candidates(image, detections, limit=3):
    """Return plausible display/body quadrilaterals without using truth labels."""
    height, width = image.shape[:2]
    scale = min(1.0, 1280.0 / max(height, width))
    working = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 110)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    row_centers = []
    for detection in detections:
        if detection["class"] == "row" and detection["score"] >= 0.2:
            x1, y1, x2, y2 = detection["box"]
            row_centers.append((scale * (x1 + x2) / 2, scale * (y1 + y2) / 2))
    image_area = working.shape[0] * working.shape[1]
    scored = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximate = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approximate) != 4 or not cv2.isContourConvex(approximate):
            continue
        area = abs(cv2.contourArea(approximate))
        fraction = area / max(image_area, 1)
        if not 0.025 <= fraction <= 0.9:
            continue
        rect = cv2.minAreaRect(approximate)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area <= 0 or area / rect_area < 0.72:
            continue
        points = approximate.reshape(4, 2)
        ordered = order_quad(points)
        quad_width = max(
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[3]),
        )
        quad_height = max(
            np.linalg.norm(ordered[3] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
        )
        aspect = quad_width / max(quad_height, 1)
        if not 0.45 <= aspect <= 3.2:
            continue
        contained = sum(
            cv2.pointPolygonTest(approximate, center, False) >= 0
            for center in row_centers
        )
        center = points.mean(axis=0)
        center_distance = np.linalg.norm(
            center - np.array([working.shape[1] / 2, working.shape[0] / 2])
        ) / max(working.shape[:2])
        score = 3.0 * contained + 1.5 * fraction - 0.4 * center_distance
        scored.append((score, points.astype(np.float32) / scale))
    scored.sort(key=lambda entry: entry[0], reverse=True)
    unique = []
    for score, points in scored:
        bounds = cv2.boundingRect(points.astype(np.int32))
        if any(_rect_iou(bounds, old_bounds) > 0.9 for _, _, old_bounds in unique):
            continue
        unique.append((score, points, bounds))
        if len(unique) == limit:
            break
    return [(score, points) for score, points, _ in unique]


def _rect_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / max(union, 1)


def rectified_fallback(detector, image, original, min_score, accept_score):
    views = []
    candidates = display_candidates(image, original["detections"])
    for index, (_, points) in enumerate(candidates):
        warped = warp_quad(image, points)
        if warped is None:
            continue
        views.append((
            f"quad-{index + 1}-original",
            detector.read(warped, min_score, accept_score),
        ))
        normalized = normalize_light(warped, 0.04)
        views.append((
            f"quad-{index + 1}-normalized",
            detector.read(normalized, min_score, accept_score),
        ))
    selected = select_unopposed_consensus(
        original, views, "display-rectification-consensus"
    )
    return selected, views, len(candidates)


class EnsembleDigitRecognizer(DigitRecognizer):
    """Average digit probabilities across small crop/pixel variants."""

    def _variants(self, image, box):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        box_width, box_height = x2 - x1, y2 - y1
        crops = []
        for padding in (0.0, 0.04, 0.08):
            px, py = padding * box_width, padding * box_height
            left = max(0, math.floor(x1 - px))
            top = max(0, math.floor(y1 - py))
            right = min(width, math.ceil(x2 + px))
            bottom = min(height, math.ceil(y2 + py))
            crop = image[top:bottom, left:right]
            if crop.size:
                crops.append(crop)
        if crops:
            gray = cv2.cvtColor(crops[0], cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
            crops.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
        return crops

    def refine(self, image, detections):
        digits = [detection for detection in detections if detection["class"] != "row"]
        kept = [detection for detection in detections if detection["class"] == "row"]
        tensors = []
        owners = []
        for index, detection in enumerate(digits):
            for crop in self._variants(image, detection["box"]):
                tensors.append(digit_tensor(crop))
                owners.append(index)
        if not tensors:
            return kept
        logits = self.session.run(
            None, {self.input: np.stack(tensors).astype(np.float32)}
        )[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        grouped = defaultdict(list)
        for owner, probability in zip(owners, probabilities):
            grouped[owner].append(probability)
        for index, detection in enumerate(digits):
            if index not in grouped:
                continue
            probability = np.mean(grouped[index], axis=0)
            class_id = int(probability.argmax())
            confidence = float(probability[class_id])
            if class_id == 10 or confidence < self.min_score:
                continue
            kept.append({
                **detection,
                "class": str(class_id),
                "detector_class": detection["class"],
                "recognition_score": confidence,
                "score": min(detection["score"], confidence),
            })
        return kept


def rescue_pipeline(primary, rescue, image, min_score, accept_score):
    original, details = current_pipeline(primary, image, min_score, accept_score)
    if is_plausible_reading(original["reading"]):
        return original, details
    rescued, rescue_details = current_pipeline(rescue, image, min_score, accept_score)
    combined = {
        "stage": "digit-ensemble-rescue" if is_plausible_reading(rescued["reading"])
        else details["stage"],
        "passes": details["passes"] + rescue_details["passes"],
    }
    return (rescued if is_plausible_reading(rescued["reading"]) else original), combined


def run_pipeline(name, detector, image, min_score, accept_score):
    if name == "full-frame":
        return detector.read(image, min_score, accept_score), {"stage": "full", "passes": 1}

    if name == "current-staged":
        return current_pipeline(detector, image, min_score, accept_score)

    if name == "expanded-normalization":
        full = detector.read(image, min_score, accept_score)
        current, details = current_pipeline_after_full(
            detector, image, full, min_score, accept_score
        )
        if is_plausible_reading(current["reading"]):
            return current, details
        selected, views, _ = expanded_normalization_fallback(
            detector, image, full, min_score, accept_score
        )
        return selected, {
            "stage": "expanded-normalization" if selected is not full else details["stage"],
            "passes": details["passes"] + len(views),
        }

    if name == "display-rectification":
        full = detector.read(image, min_score, accept_score)
        current, details = current_pipeline_after_full(
            detector, image, full, min_score, accept_score
        )
        if is_plausible_reading(current["reading"]):
            return current, details
        selected, views, candidates = rectified_fallback(
            detector, image, full, min_score, accept_score
        )
        return selected, {
            "stage": "display-rectification" if selected is not full else details["stage"],
            "passes": details["passes"] + len(views),
            "displayCandidates": candidates,
        }

    if name == "cumulative-experimental":
        full = detector.read(image, min_score, accept_score)
        current, details = current_pipeline_after_full(
            detector, image, full, min_score, accept_score
        )
        if is_plausible_reading(current["reading"]):
            return current, details
        normalized, views, _ = expanded_normalization_fallback(
            detector, image, full, min_score, accept_score
        )
        details["passes"] += len(views)
        if normalized is not full:
            details["stage"] = "expanded-normalization"
            return normalized, details
        rectified, rectified_views, candidates = rectified_fallback(
            detector, image, full, min_score, accept_score
        )
        details["passes"] += len(rectified_views)
        details["displayCandidates"] = candidates
        if rectified is not full:
            details["stage"] = "display-rectification"
            return rectified, details
        return full, details

    raise ValueError(f"Unknown pipeline {name}")


def current_pipeline_after_full(detector, image, full, min_score, accept_score):
    selected = full
    passes = 1
    stage = "full"
    if not is_plausible_reading(selected["reading"]):
        selected, portrait_views = portrait_fallback(
            detector, image, full, min_score, accept_score
        )
        passes += len(portrait_views)
        if selected is not full:
            stage = "portrait"
    if not is_plausible_reading(selected["reading"]):
        rect = row_region(full["detections"], image.shape[1], image.shape[0])
        if rect:
            x, y, width, height = rect
            crop = image[y:y + height, x:x + width]
            views = [
                (f"local-{fraction:.2f}", detector.read(
                    normalize_light(crop, fraction), min_score, accept_score
                ))
                for fraction in (0.04, 0.05)
            ]
            passes += len(views)
            adaptive = select_unopposed_consensus(full, views, "current-local-normalization")
            if adaptive is not full:
                selected = adaptive
                stage = "adaptive"
    return selected, {"stage": stage, "passes": passes}


def readable_record(item, result, details, elapsed_ms):
    expected = {field: item[field] for field in FIELDS}
    prediction = result["reading"]
    fields = {
        field: prediction is not None and prediction[field] == expected[field]
        for field in FIELDS
    }
    return {
        "id": item["id"],
        "source": item["source"],
        "expected": expected,
        "prediction": prediction,
        "accepted": result["status"] == "candidate",
        "exact_triplet": all(fields.values()),
        "field_correct": fields,
        "status": result["status"],
        "score": result["score"],
        "stage": details["stage"],
        "passes": details["passes"],
        "elapsed_ms": round(elapsed_ms, 2),
        "error": None,
    }


def refusal_record(item, result, details, elapsed_ms):
    return {
        "id": item["id"],
        "prediction": result["reading"],
        "accepted": result["status"] == "candidate",
        "status": result["status"],
        "score": result["score"],
        "stage": details["stage"],
        "passes": details["passes"],
        "elapsed_ms": round(elapsed_ms, 2),
    }


def refusal_summary(records):
    total = len(records)
    safe = sum(record["prediction"] is None for record in records)
    accepted = sum(record["accepted"] for record in records)
    times = sorted(record["elapsed_ms"] for record in records)
    return {
        "images": total,
        "safe_refusals": safe,
        "safe_refusal_rate": safe / total if total else None,
        "safe_refusal_rate_wilson_95": wilson(safe, total),
        "returned_readings": total - safe,
        "unsafe_candidate_outputs": accepted,
        "latency_ms": {
            "median": times[len(times) // 2] if times else None,
            "p95": times[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)] if times else None,
        },
    }


def main():
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
        default=ROOT / "prototype/reports/ablation_v1.json",
    )
    parser.add_argument(
        "--pipelines", nargs="+",
        default=[
            "full-frame",
            "current-staged",
            "expanded-normalization",
            "display-rectification",
            "cumulative-experimental",
            "digit-ensemble-current",
            "digit-ensemble-rescue",
        ],
    )
    args = parser.parse_args()

    model_dir = ROOT / "prototype/web/models"
    config = json.loads((model_dir / "config.json").read_text())
    detector_path = model_dir / "bp-detector.onnx"
    digits_path = model_dir / "bp-digits.onnx"
    standard = Detector(detector_path, digit_model=digits_path)
    ensemble = Detector(detector_path, digit_model=None)
    ensemble.recognizer = EnsembleDigitRecognizer(
        digits_path, min_score=config["digitMinScore"]
    )
    truth = json.loads(args.truth.read_text())
    refusals = json.loads(args.refusals.read_text())
    image_cache = {
        item["id"]: checked_image(item)
        for item in truth + refusals
    }
    pipelines = {}
    for requested_name in args.pipelines:
        pipeline_name = requested_name
        detector = standard
        implementation_name = requested_name
        if requested_name == "digit-ensemble-current":
            detector = ensemble
            implementation_name = "current-staged"
        readable_records = []
        refusal_records = []
        for item in truth:
            started = time.perf_counter()
            if requested_name == "digit-ensemble-rescue":
                result, details = rescue_pipeline(
                    standard, ensemble, image_cache[item["id"]],
                    config["minScore"], config["acceptScore"],
                )
            else:
                result, details = run_pipeline(
                    implementation_name, detector, image_cache[item["id"]],
                    config["minScore"], config["acceptScore"],
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            readable_records.append(readable_record(item, result, details, elapsed_ms))
        for item in refusals:
            started = time.perf_counter()
            if requested_name == "digit-ensemble-rescue":
                result, details = rescue_pipeline(
                    standard, ensemble, image_cache[item["id"]],
                    config["minScore"], config["acceptScore"],
                )
            else:
                result, details = run_pipeline(
                    implementation_name, detector, image_cache[item["id"]],
                    config["minScore"], config["acceptScore"],
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            refusal_records.append(refusal_record(item, result, details, elapsed_ms))
        summary = summarize(readable_records)
        refusal = refusal_summary(refusal_records)
        pipelines[pipeline_name] = {
            "readableSummary": summary,
            "refusalSummary": refusal,
            "stageCounts": dict(Counter(
                record["stage"] for record in readable_records + refusal_records
            )),
            "meanPasses": round(float(np.mean([
                record["passes"] for record in readable_records + refusal_records
            ])), 3),
            "readableRecords": readable_records,
            "refusalRecords": refusal_records,
        }
        print(json.dumps({
            "pipeline": pipeline_name,
            "exact": summary["correct_triplets"],
            "candidates": summary["accepted_triplets"],
            "candidatePrecision": summary["accepted_precision"],
            "safeRefusals": refusal["safe_refusals"],
            "medianMs": summary["latency_ms"]["median"],
            "p95Ms": summary["latency_ms"]["p95"],
            "meanPasses": pipelines[pipeline_name]["meanPasses"],
            "stageCounts": pipelines[pipeline_name]["stageCounts"],
        }), flush=True)

    report = {
        "benchmark": "external-v2 consumed regression suite",
        "independentHeldOutClaim": False,
        "selectionAllowed": False,
        "config": config,
        "detectorSha256": hashlib.sha256(detector_path.read_bytes()).hexdigest(),
        "digitsSha256": hashlib.sha256(digits_path.read_bytes()).hexdigest(),
        "truthSha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
        "refusalsSha256": hashlib.sha256(args.refusals.read_bytes()).hexdigest(),
        "pipelines": pipelines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
