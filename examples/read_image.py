# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Single-pass Python integration example; not the full browser fallback pipeline."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from prototype.detector import Detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--models', type=Path, default=ROOT / 'prototype/web/models')
    args = parser.parse_args()
    config = json.loads((args.models / 'config.json').read_text())
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error('Unable to decode the supplied image')
    detector = Detector(args.models / 'bp-detector.onnx',
                        digit_model=args.models / 'bp-digits.onnx')
    detector.recognizer.min_score = config['digitMinScore']
    result = detector.read(image, config['minScore'], config['acceptScore'])
    print(json.dumps({'pipeline': 'single-pass-python', 'version': config['version'],
                      'result': result}, indent=2))


if __name__ == '__main__':
    main()
