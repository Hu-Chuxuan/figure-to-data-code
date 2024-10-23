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
        value = value[value.find("{")+1:value.find("}")].strip()
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
            start, end = parse_range(value)
            return start, end, "range"
        return digit, repeat, "digit"
    return value, 0, "digit"

def is_empty(value):
    if type(value) == str:
        return value == ""
    return value == None or np.isnan(value)

def is_match(pred, gt):
    if is_empty(gt):
        return 1 if is_empty(pred) else 0, 1
    pred_1, pred_2, pred_type = to_float(pred)
    gt_1, gt_2, gt_type = to_float(gt)
    if gt_1 == None:
        if is_empty(pred) or pred_1 != None:
            return 0, 1
        return 0, 0
    if pred_1 == gt_1 and pred_2 == gt_2 and pred_type == gt_type:
        return 1, 1
    return 0, 1

class TableIterator:
    def is_split_stat(self, prev, cur):
        return type(prev) == str and type(cur) == str and \
               cur[:len(prev)] == prev and re.match(r"\(.*\)", cur[len(prev):].strip())
    
    def __init__(self, df):
        self.df = df

        self.col_group = []
        self.row_group = []
        i = 0
        while i < len(df.columns):
            self.col_group.append([i])
            idx = i
            i += 1
            while i < len(df.columns) and self.is_split_stat(df.columns[idx], df.columns[i]):
                self.col_group[-1].append(i)
                i += 1
        i = 0
        while i < len(df):
            self.row_group.append([i])
            idx = i
            i += 1
            while i < len(df) and self.is_split_stat(df.iloc[idx][0], df.iloc[i][0]):
                self.row_group[-1].append(i)
                i += 1

        self.row_group_ptr, self.col_group_ptr = 0, 0
        self.cur_row_in, self.cur_col_in = -1, 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.row_group_ptr >= len(self.row_group):
            raise StopIteration
        
        while True:
            self.cur_row_in += 1
            if self.cur_row_in >= len(self.row_group[self.row_group_ptr]):
                self.cur_row_in = 0
                self.cur_col_in += 1
            if self.cur_col_in >= len(self.col_group[self.col_group_ptr]):
                self.cur_col_in = 0
                self.col_group_ptr += 1
            if self.col_group_ptr >= len(self.col_group):
                self.col_group_ptr = 0
                self.row_group_ptr += 1
            if self.row_group_ptr >= len(self.row_group):
                raise StopIteration
            row_idx = self.row_group[self.row_group_ptr][self.cur_row_in]
            col_idx = self.col_group[self.col_group_ptr][self.cur_col_in]
            if (self.cur_row_in != 0 or self.cur_col_in != 0) and is_empty(self.df.iloc[row_idx][col_idx]):
                continue
            if self.cur_row_in != 0 and col_idx == 0:
                continue

            return self.df.iloc[row_idx][col_idx]

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

        pred_iter = TableIterator(pred_df)
        gt_iter = TableIterator(gt_df)

        if len(pred_iter.col_group) != len(gt_iter.col_group):
            raise FormatError(f"The predicted CSV file has {len(pred_iter.col_group)} columns while the ground truth CSV file has {len(gt_iter.col_group)} columns.")
        elif len(pred_iter.row_group) != len(gt_iter.row_group):
            raise FormatError(f"The predicted CSV file has {len(pred_iter.row_group)} rows while the ground truth CSV file has {len(gt_iter.row_group)} rows.")
        for pred_val, gt_val in zip(pred_iter, gt_iter):
            s_inc, t_inc = is_match(pred_val, gt_val)
            same += s_inc
            total += t_inc

    print("same: ", same, "total: ", total)
    return same / total
