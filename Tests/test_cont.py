import cv2
import pandas as pd
import unittest
import sys, os
from scipy.interpolate import interp1d
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))
from SubplotConstructor import Axis
from Subplot import Continuous

class TestContinuous(unittest.TestCase):
    def test_simple(self):
        image = cv2.imread("plots/P-6-O8.png")

        x_axis = Axis(line_r=777, line_c_lo=90, line_c_hi=1056, direction="x")
        x_axis.ticks = [119, 262, 404, 547, 690, 832, 975]
        y_axis = Axis(line_r=90, line_c_lo=17, line_c_hi=777, direction="y")
        y_axis.ticks = [96, 168, 244, 320, 396, 472, 548, 624, 700]

        x_axis.set_labels([1985, 1990, 1995, 2000, 2005, 2010, 2015])
        y_axis.set_labels([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        subplot = Continuous(x_axis, y_axis, subplot_value="Figure 1", has_error_bars=False, value_direction="y")

        subplot.estimate(image)
        df = subplot.to_value()
        gt = pd.read_csv("gt/P-6-O8.csv")

        # TODO: Move the comparison into the Continuous class
        estimate_interp = interp1d(df["Type-1"], df["Value"], kind="cubic", fill_value="extrapolate")
        gt_interp = interp1d(gt["Type-1"], gt["Value"], kind="cubic", fill_value="extrapolate")
        x_common = np.linspace(min(min(df["Type-1"]), min(gt["Type-1"])), max(max(df["Type-1"]), max(gt["Type-1"])), num=100)
        estimate_y = estimate_interp(x_common)
        gt_y = gt_interp(x_common)
        diffs = []
        matched = 0
        for i in range(len(x_common)):
            diffs.append(abs(estimate_y[i] - gt_y[i]) / abs(gt_y[i]))
            if diffs[-1] < 0.05:
                matched += 1
        
        # Three criteria to pass the test: 
        self.assertLessEqual(sum(diffs) / len(diffs), 0.05) # Avg difference < 5%
        self.assertLessEqual(max(diffs), 0.1)               # Max difference < 10%
        self.assertGreaterEqual(matched / len(diffs), 0.9)  # At least 90% of the points are within 5% difference

if __name__ == "__main__":
    unittest.main()