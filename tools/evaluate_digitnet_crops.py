#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Evaluate DigitNet separately from row assembly on frozen public test images.

Two inputs are measured:

* source annotation crops, which isolate recognition from detection; and
* detector crops matched to source boxes, which approximate the crops DigitNet
  receives in the deployed pipeline.

The external suites are already consumed. Results from this script are useful
for diagnosis and regression selection, not as new held-out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototype.detector import decode_output, letterbox
from prototype.digit_model import digit_tensor
from prototype.reading import iou


SOURCE_CLASS_NAMES = {
    "roboflow_naphop_external_v2": ("0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9"),
    "ega_v5": tuple(str(value) for value in range(10)),
    "datacluster_v2": (*tuple(str(value) for value in range(10)), "bp-monitor"),
}


def source_class_names(image_path: Path) -> tuple[str, ...]:
    for directory, names in SOURCE_CLASS_NAMES.items():
        if directory in image_path.parts:
            return names
    raise ValueError(f"No source class mapping for {image_path}")


def annotation_digits(image_path: Path, width: int, height: int) -> list[dict]:
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []
    names = source_class_names(image_path)
    digits = []
    for line in label_path.read_text().splitlines():
        class_id, center_x, center_y, box_width, box_height = map(float, line.split())
        name = names[int(class_id)]
        if name not in tuple(str(value) for value in range(10)):
            continue
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        x2 = (center_x + box_width / 2) * width
        y2 = (center_y + box_height / 2) * height
        digits.append({"class": int(name), "box": [x1, y1, x2, y2]})
    return digits


def crop_box(image: np.ndarray, box: list[float], padding: float = 0.0) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, math.floor(x1 - pad_x))
    top = max(0, math.floor(y1 - pad_y))
    right = min(width, math.ceil(x2 + pad_x))
    bottom = min(height, math.ceil(y2 + pad_y))
    crop = image[top:bottom, left:right]
    return crop if crop.size else None


def crop_variants(image: np.ndarray, box: list[float], variant: str) -> list[np.ndarray]:
    if variant.startswith("pad-"):
        padding = float(variant.removeprefix("pad-"))
        crop = crop_box(image, box, padding)
        return [crop] if crop is not None else []
    if variant == "clahe":
        crop = crop_box(image, box)
        if crop is None:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return [enhanced]
    if variant == "ensemble":
        crops = [crop_box(image, box, padding) for padding in (0.0, 0.04, 0.08)]
        crops = [crop for crop in crops if crop is not None]
        base = crop_box(image, box)
        if base is not None:
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            crops.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray))
        return crops
    raise ValueError(f"Unknown crop variant: {variant}")


class DigitModel:
    def __init__(self, path: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def probabilities(
        self, image: np.ndarray, boxes: list[list[float]], variant: str
    ) -> list[np.ndarray | None]:
        tensors = []
        owners = []
        for owner, box in enumerate(boxes):
            for crop in crop_variants(image, box, variant):
                tensors.append(digit_tensor(crop))
                owners.append(owner)
        if not tensors:
            return [None] * len(boxes)
        logits = self.session.run(
            None, {self.input_name: np.stack(tensors).astype(np.float32)}
        )[0]
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        grouped: dict[int, list[np.ndarray]] = defaultdict(list)
        for owner, probability in zip(owners, probabilities):
            grouped[owner].append(probability)
        return [
            np.mean(grouped[index], axis=0) if index in grouped else None
            for index in range(len(boxes))
        ]


class RawDetector:
    def __init__(self, path: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.size = int(self.input.shape[-1])

    def digit_boxes(self, image: np.ndarray, minimum_score: float) -> list[list[float]]:
        tensor, transform = letterbox(image, self.size)
        output = self.session.run(None, {self.input.name: tensor})[0]
        detections = decode_output(output, transform, minimum_score)
        return [detection["box"] for detection in detections if detection["class"] != "row"]


def greedy_matches(
    truth: list[dict], predictions: list[list[float]], minimum_iou: float
) -> list[tuple[int, int, float]]:
    pairs = sorted(
        (
            (iou(item["box"], box), truth_index, prediction_index)
            for truth_index, item in enumerate(truth)
            for prediction_index, box in enumerate(predictions)
        ),
        reverse=True,
    )
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches = []
    for overlap, truth_index, prediction_index in pairs:
        if overlap < minimum_iou:
            break
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, overlap))
    return matches


def new_metrics() -> dict:
    return {
        "digits": 0,
        "top1_correct": 0,
        "accepted": 0,
        "accepted_correct": 0,
        "images": 0,
        "all_top1_correct_images": 0,
        "all_accepted_correct_images": 0,
        "confusion": [[0] * 11 for _ in range(10)],
    }


def add_image_metrics(
    metrics: dict,
    targets: list[int],
    probabilities: list[np.ndarray | None],
    minimum_score: float,
    expected_count: int,
) -> None:
    metrics["images"] += 1
    complete = bool(targets) and len(targets) == expected_count
    image_top1 = complete
    image_accepted = complete
    for target, probability in zip(targets, probabilities):
        metrics["digits"] += 1
        if probability is None:
            image_top1 = False
            image_accepted = False
            continue
        prediction = int(probability.argmax())
        score = float(probability[prediction])
        correct = prediction == target
        accepted = prediction != 10 and score >= minimum_score
        metrics["confusion"][target][prediction] += 1
        metrics["top1_correct"] += int(correct)
        metrics["accepted"] += int(accepted)
        metrics["accepted_correct"] += int(accepted and correct)
        image_top1 &= correct
        image_accepted &= accepted and correct
    metrics["all_top1_correct_images"] += int(image_top1)
    metrics["all_accepted_correct_images"] += int(image_accepted)


def finish_metrics(metrics: dict) -> dict:
    digits = metrics["digits"]
    accepted = metrics["accepted"]
    images = metrics["images"]
    return {
        **metrics,
        "top1_accuracy": metrics["top1_correct"] / digits if digits else None,
        "accepted_precision": metrics["accepted_correct"] / accepted if accepted else None,
        "accepted_correct_coverage": metrics["accepted_correct"] / digits if digits else None,
        "all_top1_correct_image_rate": metrics["all_top1_correct_images"] / images if images else None,
        "all_accepted_correct_image_rate": metrics["all_accepted_correct_images"] / images if images else None,
    }


def load_cases(manifests: list[Path]) -> list[dict]:
    cases = []
    for manifest in manifests:
        for item in json.loads(manifest.read_text()):
            image_path = ROOT / item["local_file"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Checksum mismatch for {item['id']}")
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            if label_path.exists():
                cases.append({"id": item["id"], "image": image_path})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, nargs="+", required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--digits", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digit-min-score", type=float, default=0.5)
    parser.add_argument("--detector-min-score", type=float, default=0.2)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["pad-0.0", "pad-0.04", "pad-0.08", "clahe", "ensemble"],
    )
    args = parser.parse_args()

    cases = load_cases(args.truth)
    detector = RawDetector(args.detector)
    stem_counts = defaultdict(int)
    for path in args.digits:
        stem_counts[path.stem] += 1
    models = {
        (path.parent.name if stem_counts[path.stem] > 1 else path.stem): DigitModel(path)
        for path in args.digits
    }
    results = {
        name: {
            variant: {"annotation_crops": new_metrics(), "detector_crops": new_metrics()}
            for variant in args.variants
        }
        for name in models
    }
    detection = {
        "images": 0,
        "truth_digits": 0,
        "predicted_digit_boxes": 0,
        "matched_truth_digits": 0,
        "all_truth_digits_matched_images": 0,
        "match_iou": args.match_iou,
    }

    for index, case in enumerate(cases, 1):
        image = cv2.imread(str(case["image"]))
        if image is None:
            raise ValueError(f"Could not read {case['image']}")
        height, width = image.shape[:2]
        truth = annotation_digits(case["image"], width, height)
        predicted_boxes = detector.digit_boxes(image, args.detector_min_score)
        matches = greedy_matches(truth, predicted_boxes, args.match_iou)
        matched_truth = [truth[truth_index] for truth_index, _, _ in matches]
        matched_boxes = [predicted_boxes[prediction_index] for _, prediction_index, _ in matches]

        detection["images"] += 1
        detection["truth_digits"] += len(truth)
        detection["predicted_digit_boxes"] += len(predicted_boxes)
        detection["matched_truth_digits"] += len(matches)
        detection["all_truth_digits_matched_images"] += int(len(matches) == len(truth))

        for name, model in models.items():
            for variant in args.variants:
                annotation_probabilities = model.probabilities(
                    image, [item["box"] for item in truth], variant
                )
                add_image_metrics(
                    results[name][variant]["annotation_crops"],
                    [item["class"] for item in truth],
                    annotation_probabilities,
                    args.digit_min_score,
                    len(truth),
                )
                detector_probabilities = model.probabilities(image, matched_boxes, variant)
                add_image_metrics(
                    results[name][variant]["detector_crops"],
                    [item["class"] for item in matched_truth],
                    detector_probabilities,
                    args.digit_min_score,
                    len(truth),
                )
        print(json.dumps({"progress": f"{index}/{len(cases)}", "id": case["id"]}), flush=True)

    detection["digit_recall"] = (
        detection["matched_truth_digits"] / detection["truth_digits"]
        if detection["truth_digits"]
        else None
    )
    detection["all_digits_matched_image_rate"] = (
        detection["all_truth_digits_matched_images"] / detection["images"]
        if detection["images"]
        else None
    )
    report = {
        "benchmark": "consumed external public suites; diagnostic only",
        "independentHeldOutClaim": False,
        "selectionAllowed": False,
        "casesWithSourceDigitBoxes": len(cases),
        "digitMinScore": args.digit_min_score,
        "detectorMinScore": args.detector_min_score,
        "detection": detection,
        "models": {
            name: {
                variant: {
                    crop_type: finish_metrics(metrics)
                    for crop_type, metrics in crop_types.items()
                }
                for variant, crop_types in variants.items()
            }
            for name, variants in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
