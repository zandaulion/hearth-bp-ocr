import unittest

import numpy as np
import torch

from prototype.digit_model import SegmentAwareDigitNet, SegmentDecoderDigitNet
from prototype.train_digits import synthetic_digits


class SyntheticDigitsTest(unittest.TestCase):
    def test_generates_balanced_uint8_crops(self):
        np.random.seed(7)
        crops, labels = synthetic_digits(3)

        self.assertEqual(len(crops), 30)
        self.assertEqual(labels.tolist(), [digit for digit in range(10) for _ in range(3)])
        self.assertTrue(all(crop.shape == (48, 32) for crop in crops))
        self.assertTrue(all(crop.dtype == np.uint8 for crop in crops))

    def test_zero_count_is_empty_integer_labels(self):
        crops, labels = synthetic_digits(0)

        self.assertEqual(crops, [])
        self.assertEqual(labels.dtype.kind, "i")

    def test_segment_aware_model_keeps_deployed_output_shape(self):
        model = SegmentAwareDigitNet().eval()
        inputs = torch.zeros(2, 1, 48, 32)
        with torch.inference_mode():
            digits = model(inputs)
            training_digits, segments = model.forward_with_segments(inputs)
        self.assertEqual(tuple(digits.shape), (2, 11))
        self.assertEqual(tuple(training_digits.shape), (2, 11))
        self.assertEqual(tuple(segments.shape), (2, 7))

    def test_segment_decoder_keeps_deployed_output_shape(self):
        model = SegmentDecoderDigitNet().eval()
        inputs = torch.zeros(2, 1, 48, 32)
        with torch.inference_mode():
            digits = model(inputs)
            training_digits, segments = model.forward_with_segments(inputs)
        self.assertEqual(tuple(digits.shape), (2, 11))
        self.assertEqual(tuple(training_digits.shape), (2, 11))
        self.assertEqual(tuple(segments.shape), (2, 7))

    def test_segment_decoder_is_finite_for_confident_predictions(self):
        model = SegmentDecoderDigitNet().eval()
        segments = torch.tensor([[100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0]])
        foreground = torch.tensor([[100.0]])
        with torch.inference_mode():
            logits = model._decode(segments, foreground)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
