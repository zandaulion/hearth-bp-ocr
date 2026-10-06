# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Historical development experiment: two agreeing crops after a portrait refusal.

No model/threshold changes. Portrait canvases are synthetic transformations of
existing images, not new independent evidence. This script does not implement
the later adaptive2 browser pipeline. See docs/DEVELOPMENT.md for required inputs.
"""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
import cv2
import numpy as np
from prototype.detector import Detector
from benchmark import summarize

CROPS = [('center-wide', (.15, .25, .7, .6)), ('center-tight', (.25, .32, .6, .5))]


def portrait_canvas(image):
    h, w = image.shape[:2]
    scale = min(648 / w, 768 / h)
    nw, nh = math.floor(w * scale + .5), math.floor(h * scale + .5)
    canvas = np.full((1920, 1080, 3), 114, dtype=np.uint8)
    x, y = (1080 - nw) // 2, (1920 - nh) // 2
    canvas[y:y+nh, x:x+nw] = cv2.resize(image, (nw, nh))
    return canvas


def record(item, result, elapsed):
    expected = {k:item.get(k) for k in ('sys', 'dia', 'pulse')}
    prediction = result['reading']
    fields = {k:None if v is None else prediction is not None and prediction[k] == v for k, v in expected.items()}
    return {'id':item['id'], 'local_file':item['local_file'], 'expected':expected,
            'prediction':prediction, 'accepted':result['status'] == 'candidate',
            'result':result, 'error':None, 'elapsed_ms':round(elapsed * 1000, 2),
            'field_correct':fields,
            'exact_triplet':all(fields.values()) if all(v is not None for v in expected.values()) else None}


def main():
    config = json.loads((ROOT / 'prototype/web/models/config.json').read_text())
    detector = Detector(ROOT / 'prototype/web/models/bp-detector.onnx',
                        digit_model=ROOT / 'prototype/web/models/bp-digits.onnx')
    datasets = [('validation', 'evaluation/artifacts/roboflow_valid_truth.json'),
                ('consumed-test-regression', 'evaluation/artifacts/roboflow_test_truth.json'),
                ('curated-development', 'evaluation/curated_reviewed_ground_truth.json')]
    report = {'experiment':'Portrait refusal fallback; require two candidate crops with identical readings',
              'production_changed':False, 'config':config, 'normalized_crops':CROPS,
              'independent_evidence':False, 'datasets':[]}
    for role, truth in datasets:
        items = json.loads((ROOT / truth).read_text(encoding='utf-8'))
        for mode in ('original', 'synthetic-centered-portrait'):
            baseline, fallback, attempts = [], [], []
            for item in items:
                image = cv2.imread(str(ROOT / item['local_file']))
                if image is None:
                    raise ValueError(f"Unreadable image: {item['local_file']}")
                if mode != 'original':
                    image = portrait_canvas(image)
                started = time.perf_counter()
                full = detector.read(image, config['minScore'], config['acceptScore'])
                baseline.append(record(item, full, time.perf_counter()-started))
                selected, replays = full, []
                h, w = image.shape[:2]
                if full['reading'] is None and h >= 1.2*w:
                    for name, fractions in CROPS:
                        x, y, cw, ch = [math.floor(v * (w if i % 2 == 0 else h) + .5) for i, v in enumerate(fractions)]
                        crop = detector.read(image[y:y+ch, x:x+cw], config['minScore'], config['acceptScore'])
                        replays.append({'crop':name, 'rect':[x,y,cw,ch], 'result':crop})
                    if all(c['result']['status'] == 'candidate' for c in replays) and replays[0]['result']['reading'] == replays[1]['result']['reading']:
                        selected = replays[0]['result']
                fallback.append(record(item, selected, time.perf_counter()-started))
                if replays:
                    attempts.append({'id':item['id'], 'replays':replays, 'recovered':selected is not full})
            group = {'dataset_role':role, 'truth':truth, 'mode':mode,
                     'baseline':summarize(baseline), 'fallback':summarize(fallback),
                     'baseline_records':baseline, 'fallback_records':fallback, 'attempts':attempts}
            report['datasets'].append(group)
            print(json.dumps({'role':role, 'mode':mode,
                              'baseline_correct':group['baseline']['correct_triplets'],
                              'fallback_correct':group['fallback']['correct_triplets'],
                              'baseline_accepted':group['baseline']['accepted_triplets'],
                              'fallback_accepted':group['fallback']['accepted_triplets'],
                              'fallback_precision':group['fallback']['accepted_precision']}), flush=True)
    target = ROOT / 'prototype/reports/crop_fallback_experiment.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
