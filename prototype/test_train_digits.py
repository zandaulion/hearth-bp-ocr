import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
