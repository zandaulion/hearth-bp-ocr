# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Unit tests for model-blind external-v3 label parsing."""

from pathlib import Path
import tempfile
import unittest

import build_external_test_v3
from build_external_test_v3 import parse_class_names, reading_from_digit_boxes


def digit(value, x, y, height=0.1):
    return {"class": str(value), "x": x, "y": y, "width": 0.05, "height": height}


class ExternalV3BuilderTests(unittest.TestCase):
    # All readings and geometry below are synthetic test fixtures.
    def test_reads_key_from_ignored_repo_env_without_printing_it(self):
        original_root = build_external_test_v3.ROOT
        original_value = build_external_test_v3.os.environ.pop("ROBOFLOW_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                build_external_test_v3.ROOT = Path(directory)
                (Path(directory) / ".env").write_text("ROBOFLOW_API_KEY=test-secret\n")
                self.assertEqual(build_external_test_v3.roboflow_api_key(), "test-secret")
        finally:
            build_external_test_v3.ROOT = original_root
            if original_value is not None:
                build_external_test_v3.os.environ["ROBOFLOW_API_KEY"] = original_value

    def test_parses_roboflow_block_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.yaml"
            path.write_text("names:\n  - '0'\n  - '1'\n  - bp-monitor\nnc: 3\n")
            self.assertEqual(parse_class_names(path), ["0", "1", "bp-monitor"])

    def test_reconstructs_three_rows(self):
        boxes = [
            digit(1, .3, .2), digit(2, .4, .2), digit(0, .5, .2),
            digit(8, .35, .5), digit(0, .45, .5),
            digit(7, .35, .8), digit(1, .45, .8),
            {"class": "bp-monitor", "x": .5, "y": .5, "width": .9, "height": .9},
        ]
        reading, reason = reading_from_digit_boxes(boxes)
        self.assertEqual(reason, "candidate")
        self.assertEqual(reading, {"sys": 120, "dia": 80, "pulse": 71})

    def test_rejects_extra_date_digits(self):
        boxes = [
            digit(1, .3, .2), digit(2, .4, .2), digit(0, .5, .2),
            digit(8, .35, .5), digit(0, .45, .5),
            digit(7, .35, .8), digit(1, .45, .8),
            digit(2, .8, .92), digit(4, .9, .92), digit(5, .95, .92),
        ]
        reading, reason = reading_from_digit_boxes(boxes)
        self.assertIsNone(reading)
        self.assertEqual(reason, "digit_count_10")


if __name__ == "__main__":
    unittest.main()
