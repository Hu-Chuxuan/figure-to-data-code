import cv2
import pandas as pd
import unittest
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
from SubplotConstructor import Axis
from Subplot import Histogram

class TestHistogram(unittest.TestCase):
    def test_simple(self):
        image = cv2.imread("plots/P-55-O1.png")

        x_axis = Axis(line_r=1060, line_c_lo=239, line_c_hi=1507, direction="x")
        x_axis.ticks = [350, 524, 699, 873, 1048, 1222, 1396]
        y_axis = Axis(line_r=239, line_c_lo=162, line_c_hi=1060, direction="y")
        y_axis.ticks = [184, 354, 524, 694, 864, 1034]

        x_axis.set_labels(["<12", "12-16", "16-20", "20-24", "24-28", "28-32", ">32"])
        y_axis.set_labels([125, 100, 75, 50, 25, 0])
        subplot = Histogram(x_axis, y_axis, subplot_value="Figure 1", has_error_bars=False, value_direction="y")

        subplot.estimate(image)
        self.assertEqual(len(subplot.curves), 7)
        df = subplot.to_value().sort_values(by="Type-1")
        gt = pd.read_csv("gt/P-55-O1.csv").sort_values(by="Type-1")
        
        # TODO: Move the comparison into the Continuous class
        diffs = []
        for i in range(len(df)):
            self.assertEqual(df["Type-1"][i], gt["Type-1"][i])
            diffs.append(abs(df["Value"][i] - gt["Value"][i]) / gt["Value"][i])
        self.assertLessEqual(sum(diffs) / len(diffs), 0.05)
        self.assertLessEqual(max(diffs), 0.1)

if __name__ == "__main__":
    unittest.main()