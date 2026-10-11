import unittest

from tools.evaluate_teleocr import parse_reading


class ParseReadingTest(unittest.TestCase):
    def test_labeled_output(self):
        self.assertEqual(
            parse_reading("SYS=120 DIA=80 PULSE=70"),
            ({"sys": 120, "dia": 80, "pulse": 70}, "labels"),
        )

    def test_unlabeled_unique_sequence(self):
        self.assertEqual(
            parse_reading("120\n80\n70"),
            ({"sys": 120, "dia": 80, "pulse": 70}, "unique-bounded-sequence"),
        )

    def test_ambiguous_sequence_refuses(self):
        self.assertEqual(parse_reading("120 80 70 65")[0], None)

    def test_implausible_labels_refuse_without_correction(self):
        self.assertEqual(parse_reading("SYS 12 DIA 80 PULSE 70")[0], None)


if __name__ == "__main__":
    unittest.main()
