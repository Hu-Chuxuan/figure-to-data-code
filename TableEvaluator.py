import re
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

from Exceptions import WrongCSVNumberError, FormatError, YELLOW, BLUE, MAGENTA, RESET

MIN_FUZZ_RATIO = 50

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
    if len(digit) == 0 or digit == "-":
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

'''
@description: if the string is in the format of "start - end" where start and end are two numbers, 
              we can parse the start and end values
@params: a string
@return: start and end values, None if the string is not in the format of "start - end"
'''
def parse_range(value):
    if len(value) <= 1:
        return None, None
    bracket_open = ["[", "(", "{", "<"]
    bracket_close = ["]", ")", "}", ">"]
    value = value.strip()
    if value[0] in bracket_open and value[-1] == bracket_close[bracket_open.index(value[0])]:
        value = value[1:-1]
    range_dict = [",", ";", "to", "--"]
    for range_str in range_dict:
        if range_str in value and value.count(range_str) == 1:
            start, end = value.split(range_str)
            start = start.strip()
            end = end.strip()
            try:
                start = float(start)
                end = float(end)
            except:
                return None, None
            return start, end
    if "–" in value:
        value = value.replace("–", "-")
    if "-" in value:
        dash_indices = [i for i in range(len(value)) if value[i] == "-"]
        if dash_indices[0] == 0:
            if len(dash_indices) == 1:
                return None, None
            split_idx = dash_indices[1]
        else:
            split_idx = dash_indices[0]
        start = value[:split_idx].strip()
        end = value[split_idx+1:].strip()
        try:
            start = float(start)
            end = float(end)
            return start, end
        except:
            return None, None
    return None, None

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
        return 1 if is_empty(pred) else 0
    gt_1, gt_2, gt_type = to_float(gt)
    if gt_1 == None:
        if is_empty(pred):
            return 0
        if fuzz.ratio(str(pred), gt) >= MIN_FUZZ_RATIO:
            return 1
        return 0
    pred_1, pred_2, pred_type = to_float(pred)
    if pred_1 == gt_1 and pred_2 == gt_2 and pred_type == gt_type:
        return 1
    return 0

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
        for i, col in enumerate(gt_df.columns):
            if "Empty Column" in col or "Unnamed" in col:
                if "Empty Column" in pred_df.columns[i] or "Unnamed" in pred_df.columns[i]:
                    same += 1
            else:
                gt = col
                pred = pred_df.columns[i]
                same += is_match(pred, gt)
            total += 1
        for gt_row in range(len(gt_df)):
            for gt_col in range(len(gt_df.columns)):
                if gt_row >= len(pred_df):
                    total += 1
                    continue
                pred_value = pred_df.iloc[gt_row, gt_col]
                gt_value = gt_df.iloc[gt_row, gt_col]
                same += is_match(pred_value, gt_value)
                total += 1

    return {"Table accuracy": same / total}
