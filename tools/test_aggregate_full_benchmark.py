# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Unit tests for aggregate-only consumed benchmark metrics."""

import unittest

from aggregate_full_benchmark import mlkit_summary, paired, percentile


class AggregateFullBenchmarkTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2)
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 4)
        self.assertIsNone(percentile([], 0.5))

    def test_paired_outcomes(self):
        left = {"a": True, "b": True, "c": False, "d": False}
        right = {"a": True, "b": False, "c": True, "d": False}
        self.assertEqual(paired(left, right), {
            "both_exact": 1,
            "left_only_exact": 1,
            "right_only_exact": 1,
            "neither_exact": 1,
        })

    def test_mlkit_summary_uses_app_acceptance_for_safe_refusal(self):
        # Synthetic readings only; these are not captured health records.
        records = [
            {
                "id": "readable-1", "role": "readable", "available": True,
                "appAccepted": True, "complete": True, "exactTriplet": True,
                "fields": {"sys": True, "dia": True, "pulse": True},
                "prediction": {"sys": 111, "dia": 77, "pulse": 66},
                "elapsedMs": 10.0, "preprocessMs": 0.0,
            },
            {
                "id": "readable-2", "role": "readable", "available": False,
                "appAccepted": False, "complete": False, "exactTriplet": False,
                "fields": {"sys": False, "dia": False, "pulse": False},
                "prediction": None, "elapsedMs": 0.0, "preprocessMs": 0.0,
            },
            {
                "id": "refusal-1", "role": "refusal", "available": True,
                "appAccepted": False, "complete": False, "prediction": None,
                "elapsedMs": 5.0, "preprocessMs": 0.0,
            },
        ]
        summary = mlkit_summary(records)
        self.assertEqual(summary["exact_triplets"], 1)
        self.assertEqual(summary["app_accepted"], 1)
        self.assertEqual(summary["safe_refusals"], 1)
        self.assertEqual(summary["latency_ms"]["median"], 10.0)


if __name__ == "__main__":
    unittest.main()
