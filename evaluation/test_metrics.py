# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Guard the benchmark against counting refusals as successes or unknowns as errors."""
import unittest
from benchmark import summarize, wilson
from dataset_tools import CLASS_NAMES, rows_from_annotations


def record(expected, prediction, accepted=False):
    fields = {k: None if v is None else prediction is not None and prediction[k] == v for k,v in expected.items()}
    return {"expected": expected, "prediction": prediction, "accepted": accepted,
            "field_correct": fields, "exact_triplet": all(fields.values()) if all(v is not None for v in expected.values()) else None,
            "error": "unreadable" if prediction is None else None, "elapsed_ms": 1}


class MetricsTests(unittest.TestCase):
    def test_accuracy_precision_and_coverage_have_distinct_denominators(self):
        expected = {"sys": 120, "dia": 80, "pulse": 70}
        bad = {**expected, "pulse": 71}
        stats = summarize([record(expected, expected, True), record(expected, bad, True), record(expected, None)])
        self.assertEqual(stats["exact_triplet_accuracy"], 1/3)
        self.assertEqual(stats["accepted_precision"], .5)
        self.assertEqual(stats["accepted_coverage"], 2/3)
        self.assertEqual(stats["errors"], 1)

    def test_missing_pulse_does_not_enter_triplet_denominator(self):
        expected = {"sys": 149, "dia": 76, "pulse": None}
        stats = summarize([record(expected, {"sys":149,"dia":76,"pulse":999}, True)])
        self.assertEqual(stats["fields"]["sys"]["accuracy"], 1)
        self.assertIsNone(stats["fields"]["pulse"]["accuracy"])
        self.assertEqual(stats["complete_ground_truth_images"], 0)
        self.assertIsNone(stats["accepted_precision"])

    def test_no_acceptance_is_not_perfect_precision(self):
        stats = summarize([record({"sys":120,"dia":80,"pulse":70}, None)])
        self.assertIsNone(stats["accepted_precision"])
        self.assertEqual(stats["exact_triplet_accuracy"], 0)

    def test_tiny_sample_cannot_prove_ninety_percent(self):
        self.assertLess(wilson(8,8)[0], .9)
        self.assertGreater(wilson(100,100)[0], .9)

    def test_class_ids_are_not_digit_values(self):
        self.assertEqual(CLASS_NAMES[2], "10")
        self.assertEqual(CLASS_NAMES[3], "2")

    def test_ambiguous_digit_assignment_is_rejected(self):
        row = {"class":"10","box":[0,0,1,1],"center":[.5,.5]}
        digit = {"class":"8","box":[.4,.4,.6,.6],"center":[.5,.5]}
        _, _, _, valid = rows_from_annotations([row,row,row,digit])
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
