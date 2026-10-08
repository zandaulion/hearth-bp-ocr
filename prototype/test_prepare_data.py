# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Ensure known external regression sets cannot enter training manifests."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototype.prepare_data import reject_test_only_paths


class PrepareDataTests(unittest.TestCase):
    def test_public_training_path_is_allowed(self):
        reject_test_only_paths([
            "/repo/dataset/roboflow_bp_display/train/images/example.jpg"
        ])

    def test_external_test_path_is_rejected(self):
        with self.assertRaises(RuntimeError):
            reject_test_only_paths([
                "/repo/dataset/external_test_v2/example.jpg"
            ])

    def test_external_candidate_path_is_rejected(self):
        with self.assertRaises(RuntimeError):
            reject_test_only_paths([
                "/repo/dataset/wikimedia_bp_test_v2_candidates/images/example.jpg"
            ])


if __name__ == "__main__":
    unittest.main()
