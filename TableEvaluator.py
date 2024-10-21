import re
import pandas as pd
import numpy as np

from Exceptions import WrongCSVNumberError, FormatError, YELLOW, BLUE, MAGENTA, RESET
from PlotEvaluator import parse_range

def is_repeat(value, repeat):
    if len(repeat) == 0 or len(value) % len(repeat) != 0:
        return False
    return value == repeat * (len(value) // len(repeat))

def parse_digit_from_sig(value):
    # find the number at the beginning of the string
    value = value.strip()
    match = re.match(r'^-?\d*\.?\d*', value)
    if not match:
        return None, None
    digit = match.group(0)
    if len(digit) == 0:
        return None, None
    value = value[len(digit):]
    digit = float(digit)
    if len(value) == 0:
        return digit, 0
    
    # find repeat in value[len(digit):]
    repeat = ""
    if "{" in value and "}" in value:
        value = value[value.find("{")+1:value.find("}")]
    for ch in value:
        if not is_repeat(value, repeat):
            repeat += ch
        else:
            break
    repeat_cnt = len(value) // len(repeat)
    if repeat_cnt == 1:
        # if a space is in the repeat, or it contain both digits and letters, we do not consider it as a valid repeat
        if " " in repeat or (re.search(r'\d', repeat) and re.search(r'\D', repeat)):
            return None, None
    return digit, repeat_cnt

def to_float(value):
    if type(value) == str:
        digit, repeat = parse_digit_from_sig(value)
        if digit is None:
            return parse_range(value)
        return digit, repeat
    return value, 0

def is_empty(value):
    if type(value) == str:
        return value == ""
    return value == None or np.isnan(value)

def is_match(pred, gt):
    if is_empty(gt):
        return 1 if is_empty(pred) else 0, 1
    pred_1, pred_2 = to_float(pred)
    gt_1, gt_2 = to_float(gt)
    if gt_1 == None:
        return 0, 0
    if pred_1 == gt_1 and pred_2 == gt_2:
        return 1, 1
    return 0, 1

class TableIterator:
    def __init__(self, df):
        self.df = df
        self.cur_row, self.cur_col, self.main_row = 0, 0, 0
    
    def next(self):
        pass

'''
@raise:
    WrongCSVNumberError: the number of predicted CSV files is different from the number of ground truth CSV files
    FormatError: the number of columns in the predicted data and the ground truth data are different
'''
def evaluate_table(pred_dfs, gt_dfs):
    if len(pred_dfs) != len(gt_dfs):
        raise WrongCSVNumberError(len(pred_dfs), len(gt_dfs))
    
    same = 0
    total = 0
    
    for df_ptr in range(len(pred_dfs)):
        pred_df = pred_dfs[df_ptr]
        gt_df = gt_dfs[df_ptr]

        if len(pred_df.columns) != len(gt_df.columns):
            raise FormatError(f"The predicted CSV file has {len(pred_df.columns)} columns while the ground truth CSV file has {len(gt_df.columns)} columns.")

        for i in range(len(gt_df.columns)):
            for j in range(len(gt_df)):
                if type(gt_df.iloc[j][i]) != str:
                    total += 1
                else:
                    # TODO: Is pred = nan when gt is not handled?
                    # TODO: Handle one of them is empty and the other is not
                    gt_digit, gt_repeat = parse_digit_from_sig(gt_df.iloc[j][i])
                    if gt_digit is None:
                        print("Not a data point ", gt_df.columns[i], j, gt_df.iloc[j][i])
                        continue

                    if type(pred_df.iloc[j][i]) != str:
                        pred_digit = pred_df.iloc[j][i]
                        pred_repeat = 0
                    else:
                        pred_digit, pred_repeat = parse_digit_from_sig(pred_df.iloc[j][i])
                    total += 1
                    if pred_digit is None:
                        print(MAGENTA + "Type mismatch: ", gt_df.columns[i], j, pred_df.iloc[j][i], gt_df.iloc[j][i], type(pred_df.iloc[j][i]), type(gt_df.iloc[j][i]), RESET)
                        continue
                    if pred_digit == gt_digit and pred_repeat == gt_repeat:
                        same += 1
                    else:
                        print(BLUE + "Value mismatch: ", gt_df.columns[i], j, pred_df.iloc[j][i], gt_df.iloc[j][i], pred_digit, gt_digit, pred_repeat, gt_repeat, RESET)
    print("same: ", same, "total: ", total)
    return same / total
