# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Tests for the classifier-only DigitNet diagnostic."""

import unittest

import numpy as np

from tools.evaluate_digitnet_crops import add_image_metrics, greedy_matches, new_metrics


class DigitNetCropDiagnosticTests(unittest.TestCase):
    def test_greedy_matching_is_one_to_one(self):
        truth = [
            {"box": [0, 0, 10, 10]},
            {"box": [20, 0, 30, 10]},
        ]
        predictions = [[1, 0, 11, 10], [20, 0, 30, 10]]
        matches = greedy_matches(truth, predictions, 0.3)
        self.assertEqual({(truth_index, prediction_index) for truth_index, prediction_index, _ in matches}, {(0, 0), (1, 1)})

    def test_partial_detector_match_is_not_a_complete_image(self):
        metrics = new_metrics()
        probability = np.zeros(11, dtype=np.float32)
        probability[4] = 1.0
        add_image_metrics(metrics, [4], [probability], 0.5, expected_count=2)
        self.assertEqual(metrics["top1_correct"], 1)
        self.assertEqual(metrics["all_top1_correct_images"], 0)
        self.assertEqual(metrics["all_accepted_correct_images"], 0)


if __name__ == "__main__":
    unittest.main()
