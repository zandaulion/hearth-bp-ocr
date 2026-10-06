# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Diagnostic only: classify digits using ANNOTATED boxes (not end-to-end OCR).

Training uses the official training split only. The validation split is used
to evaluate the classifier; the test split is intentionally untouched.
"""
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from dataset_tools import DATA, ROOT, annotations, rows_from_annotations
from benchmark import wilson

HOG = cv2.HOGDescriptor((32, 48), (16, 16), (8, 8), (8, 8), 9)


def crop_digit(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1*w)), max(0, int(y1*h))
    x2, y2 = min(w, int(np.ceil(x2*w))), min(h, int(np.ceil(y2*h)))
    return image[y1:y2, x1:x2]


def feature(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 48), interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(gray)
    hog = HOG.compute(normalized).ravel()
    pixels = cv2.resize(normalized, (16, 24)).astype(np.float32).ravel()/255
    # HOG plus a weak pixel template and original aspect ratio.
    return np.concatenate((hog, pixels * .2, [crop.shape[1]/crop.shape[0]]))


def load(split):
    x, y, metadata = [], [], []
    for p in sorted((DATA / split / "images").glob("*.jpg")):
        image = cv2.imread(str(p))
        for a in annotations(p):
            if a["class"] == "10":
                continue
            crop = crop_digit(image, a["box"])
            x.append(feature(crop))
            y.append(int(a["class"]))
            metadata.append({"file": p.name, "box": a["box"]})
    return np.asarray(x, dtype=np.float32), np.asarray(y), metadata


def main():
    started = time.perf_counter()
    x, y, _ = load("train")
    vx, vy, vm = load("valid")
    # Fixed hyperparameters chosen before evaluating validation results.
    classifier = SVC(C=10, gamma="scale", class_weight="balanced")
    classifier.fit(x, y)
    prediction = classifier.predict(vx)
    image_correct = {}
    for label, pred, meta in zip(vy, prediction, vm):
        image_correct.setdefault(meta["file"], True)
        image_correct[meta["file"]] &= bool(label == pred)
    result = {
        "experiment": "Annotated digit crops: HOG + RBF SVM; NOT autonomous reading",
        "training_images": 270, "training_digits": len(y), "validation_images": len(image_correct), "validation_digits": len(vy),
        "training_class_counts": dict(Counter(map(str, y))),
        "learned_digit_accuracy": float(np.mean(prediction == vy)),
        "all_annotated_digits_correct_images": sum(image_correct.values()),
        "all_annotated_digits_correct_image_rate": sum(image_correct.values())/len(image_correct),
        "all_annotated_digits_correct_image_wilson_95": wilson(sum(image_correct.values()), len(image_correct)),
        "confusion_matrix_rows_truth_cols_prediction_0_to_9": confusion_matrix(vy, prediction, labels=list(range(10))).tolist(),
        "errors": [{**m, "expected": int(t), "prediction": int(p)} for t,p,m in zip(vy, prediction, vm) if t != p],
        "test_split_used": False,
        "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
        "elapsed_seconds": round(time.perf_counter()-started, 2),
        "limitations": ["Digit locations come from annotations", "No screen detection, row detection or rejection evaluated", "Official splits may contain visually similar stock photos", "No phone runtime benchmark", "Per-digit accuracy is not all-three-readings accuracy"],
    }
    out = ROOT / "evaluation/artifacts"
    joblib.dump(classifier, out / "digit_svm.joblib")
    (out / "digit_experiment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("errors", "confusion_matrix_rows_truth_cols_prediction_0_to_9")}, indent=2))


if __name__ == "__main__":
    main()
