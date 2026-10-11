import unittest

from tools.evaluate_easyocr import geometry_reading


def box(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


class GeometryReadingTest(unittest.TestCase):
    def test_three_rows(self):
        result, method, _rows = geometry_reading(
            [
                (box(10, 10, 70, 30), "120", 0.9),
                (box(12, 50, 60, 70), "80", 0.8),
                (box(11, 90, 60, 110), "70", 0.7),
            ],
            0.0,
        )
        self.assertEqual(result, {"sys": 120, "dia": 80, "pulse": 70})
        self.assertEqual(method, "unique-bounded-row-stack")

    def test_split_digits_are_joined(self):
        result, _method, _rows = geometry_reading(
            [
                (box(10, 10, 20, 30), "1", 0.9),
                (box(22, 10, 32, 30), "2", 0.9),
                (box(34, 10, 44, 30), "0", 0.9),
                (box(10, 50, 30, 70), "80", 0.9),
                (box(10, 90, 30, 110), "70", 0.9),
            ],
            0.0,
        )
        self.assertEqual(result, {"sys": 120, "dia": 80, "pulse": 70})

    def test_low_confidence_is_filtered(self):
        result, _method, _rows = geometry_reading(
            [
                (box(10, 10, 70, 30), "120", 0.9),
                (box(10, 50, 70, 70), "80", 0.1),
                (box(10, 90, 70, 110), "70", 0.9),
            ],
            0.5,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
