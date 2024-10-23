import pandas as pd
import unittest
import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from TableEvaluator import evaluate_table, is_match, TableIterator

class TestTableEvaluator(unittest.TestCase):
    def assertMatch(self, value1, value2, same_inc, total_inc):
        same, total = is_match(value1, value2)
        self.assertEqual(same, same_inc)
        self.assertEqual(total, total_inc)

    def test_is_match(self):
        empty = [np.nan, "", None]
        invalid = ["1 million", "invalid number", "1.5 is smaller than 2.0"]
        number = [1, 3.14, 13.14, -3.14, -1]
        def gen_numbers(num):
            return [
                [num, str(num), np.float64(num)], # no significance
                [f"{num}*", f"{num}\dagger", f"  {num}*", f"{num}*   ", str(num)+"^{*}", str(num)+"^{ * }"], # 1
                [f"{num}**", f"{num}\dagger\dagger", f"  {num}**", f"{num}**  ", str(num)+"{**}"], # 2
                [f"{num}***  ", f"{num}\dagger\dagger\dagger", f"  {num}\dagger\dagger\dagger", f"{num}***  ", str(num)+"_{ ***}"], # 3
                [f"{num} - {num+1}", f"[{num}, {num+1}]", f"[{num}; {num+1}]", f"{num} to {num+1}"],
                [f"{num} - {num+2}", f"[{num}, {num+2}]", f"[{num}; {num+2}]", f"{num} to {num+2}"],
                [f"{num-1} - {num+1}", f"[{num-1}, {num+1}]", f"[{num-1}; {num+1}]", f"{num-1} to {num+1}"],
            ]

        # Test gt empty
        for gt in empty:
            for pred in empty:
                # Both empty should be considered as same
                self.assertMatch(pred, gt, 1, 1)
            for pred_num in number:
                pred_valid = gen_numbers(pred_num)
                for pred_list in pred_valid:
                    for pred in pred_list:
                        # empty should not be matched with any number
                        self.assertMatch(pred, gt, 0, 1)
            for pred in invalid:
                # empty should not be matched with any invalid value
                self.assertMatch(pred, gt, 0, 1)
        
        # Test valid values
        for gt_num in number:
            gt_numbers = gen_numbers(gt_num)
            for sign_level, gt_list in enumerate(gt_numbers):
                for gt in gt_list:
                    for pred in empty:
                        # A numerical data points should not be matched with empty
                        self.assertMatch(pred, gt, 0, 1)
                    for pred_num in number:
                        pred_valid = gen_numbers(pred_num)
                        for sign_level_pred, pred_list in enumerate(pred_valid):
                            for pred in pred_list:
                                # A numerical data points can only match with the same number and significance level
                                if sign_level == sign_level_pred and gt_num == pred_num:
                                    self.assertMatch(pred, gt, 1, 1)
                                else:
                                    self.assertMatch(pred, gt, 0, 1)
                    for pred in invalid:
                        # A numerical data points should not be matched with invalid values
                        self.assertMatch(pred, gt, 0, 1)
        # Test invalid values
        for gt in invalid:
            for pred in empty:
                # An invalid value should not be matched with empty
                self.assertMatch(pred, gt, 0, 1)
            for pred_num in number:
                pred_valid = gen_numbers(pred_num)
                for sign_level_pred, pred_list in enumerate(pred_valid):
                    for pred in pred_list:
                        # An invalid value should not be matched with numerical data points
                        self.assertMatch(pred, gt, 0, 1)
            for pred in invalid:
                # If both are invalid, should not count it at all
                self.assertMatch(pred, gt, 0, 0)

    def test_table_iterator_col(self):
        col_df = pd.DataFrame({
            "Empty Column 0": ["First row", "Second row", "Third row"],
            "DV 1": [1, 2, 3],
            "DV 1 (SE)": [0.1, 0.2, 0.3],
            "DV 2": [4, 5, 6],
            "DV 2 (SE)": [0.4, 0.5, 0.6],
            "DV 2 (p-value)": [0.04, 0.05, 0.06],
            "DV 3": [7, 8, 9]
        })
        col_iter = TableIterator(col_df)
        col_group = [[0], [1, 2], [3, 4, 5], [6]]
        self.assertEqual(col_iter.col_group, col_group)
        row_group = [[0], [1], [2]]
        self.assertEqual(col_iter.row_group, row_group)
        col_order = [
            "First row", 1, 0.1, 4, 0.4, 0.04, 7, 
            "Second row", 2, 0.2, 5, 0.5, 0.05, 8, 
            "Third row", 3, 0.3, 6, 0.6, 0.06, 9
        ]
        for i, col in enumerate(col_iter):
            self.assertEqual(col, col_order[i])

    def test_table_iterator_row(self):
        row_df = pd.DataFrame({
            "Empty Column 0": ["First row", "First row (SE)", "First row (p-value)", "Second row", "Second row (SE)", "Second row (p-value)", "Third row", "Third row (SE)", "Third row (p-value)"],
            "DV 1": [1, 0.1, "", 2, 0.2, "", 3, 0.3, ""],
            "DV 2": [4, 0.4, 0.04, 5, 0.5, 0.05, 6, 0.6, 0.06],
            "DV 3": [7, "", "", 8, "", "", 9, "", ""]
        })
        row_iter = TableIterator(row_df)
        row_group = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.assertEqual(row_iter.row_group, row_group)
        col_group = [[0], [1], [2], [3]]
        self.assertEqual(row_iter.col_group, col_group)
        row_order = [
            "First row", 1, 0.1, 4, 0.4, 0.04, 7, 
            "Second row", 2, 0.2, 5, 0.5, 0.05, 8, 
            "Third row", 3, 0.3, 6, 0.6, 0.06, 9
        ]
        for i, row in enumerate(row_iter):
            self.assertEqual(row, row_order[i])
    
    def DISABLE_test_eval_table_simple(self):
        pass

    def test_table_cmp_both(self):
        row_df = pd.read_csv("Tests/evaluator/T-10-O1_pred.csv")
        col_df = pd.read_csv("Tests/evaluator/T-10-O1_gt.csv")
        perf = evaluate_table([row_df], [col_df])
        self.assertEqual(perf, 1)
    
    def DISABLE_test_table_std_in_row(self):
        self.assertEqual(1, 2)
    
    def DISABLE_test_table_range_value(self):
        self.assertEqual(1, 2)

if __name__ == "__main__":
    unittest.main()
