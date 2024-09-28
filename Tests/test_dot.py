import cv2
import pandas as pd
import unittest
import sys, os

sys.path.append(os.path.abspath(".."))
from SubplotConstructor import Axis
from Subplot import DotPlot

class TestDotPlot(unittest.TestCase):
    def test_simple(self):
        image = cv2.imread("plots/P-61-O2.png")

        # These should be called by SubplotConstructor
        x_axis = Axis(line_r=862, line_c_lo=409, line_c_hi=1292, direction="x")
        x_axis.ticks = [432, 622, 811, 1001, 1091]
        y_axis = Axis(line_r=409, line_c_lo=9, line_c_hi=862, direction="y")
        y_axis.ticks = [189, 311, 678, 801]

        # These should be called by the LLM
        x_axis.set_labels([0, 0.05, 0.1, 0.15, 0.2])
        y_axis.set_labels(["Low Electoral Integrity - No historey of Conflict", "Low Electoral Integrity - History of Conflict", "High Electoral Integrity - No historey of Conflict", "High Electoral Integrity - History of Conflict"])
        subplot = DotPlot(x_axis, y_axis, subplot_value="Figure 2", has_error_bars=True, value_direction="x")

        # Estimate the coordinates of the dots
        subplot.estimate(image)
        # Since this is the simplest case, we do not need to use the LLM to organize the data points
        self.assertEqual(len(subplot.curves), 4)
        # Convert the coordinates to the values
        df = subplot.to_value().sort_values(by="Type-1")
        # Calculate the difference between the actual and expected values
        gt = pd.read_csv("gt/P-61-O2.csv").sort_values(by="Type-1")
        
        # TODO: Move the comparison into the Continuous class
        value_diffs, err_diffs = [], []
        for i in range(len(df)):
            self.assertEqual(df["Type-1"][i], gt["Type-1"][i])
            value_diffs.append(abs(df["Value"][i] - gt["Value"][i]) / gt["Value"][i])
            err_diffs.append(abs(df["Error"][i] - gt["Error"][i]) / gt["Error"][i])
        
        # Two criteria for the values and errors: average difference < 5%, max difference < 10%
        self.assertLessEqual(sum(value_diffs) / len(value_diffs), 0.05)
        self.assertLessEqual(max(value_diffs), 0.1)
        self.assertLessEqual(sum(err_diffs) / len(err_diffs), 0.05)
        self.assertLessEqual(max(err_diffs), 0.1)

if __name__ == "__main__":
    unittest.main()