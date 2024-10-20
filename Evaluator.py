import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

class WrongCSVNumberError(Exception):
    def __init__(self, pred_num, gt_num):
        super().__init__(YELLOW + f"The number of predicted CSV files is {pred_num} when the ground truth has {gt_num} CSV files." + RESET)

class FormatError(Exception):
    def __init__(self, msg):
        super().__init__(YELLOW + msg + RESET)

'''
@description: if the string is in the format of "start - end" where start and end are two numbers, 
              we can parse the start and end values
@params: a string
@return: start and end values, None if the string is not in the format of "start - end"
'''
def parse_range(value):
    if "-" in value and value.count("-") == 1:
        start, end = value.split("-")
        start = start.strip()
        end = end.strip()
        try:
            start = float(start)
            end = float(end)
        except:
            return None, None
        return start, end
    return None, None

'''
@description: calculate the score of the pair of two values
@return: the score of the pair. np.inf if the two values are completely different strings
'''
def pair_score(pred, gt):
    if type(pred) != type(gt):
        if type(gt) != str:
            try:
                pred = float(pred)
            except:
                pred = str(pred)
                gt = str(gt)
        else:
            pred = str(pred)
    
    if type(gt) == str:
        gt_start, gt_end = parse_range(gt)
        if gt_start is not None and gt_end is not None:
            pred_start, pred_end = parse_range(pred)
            if pred_start is None and pred_end is None:
                return np.inf, []
            return min(abs(gt_start - pred_start), abs(gt_end - pred_end)), [(gt_start, pred_start), (gt_end, pred_end)]
        return 0 if gt == pred else np.inf, []
    if gt == None:
        if pred == None:
            return 0, []
        return np.inf, []
    return abs(gt - pred), [(gt, pred)]

'''
@params:
    pred_df: DataFrame, predicted data
    gt_df: DataFrame, ground truth data
@return:
    pred: List, predicted values or error bars
    gt: List, ground truth values or error bars
'''
def pair_data_points(pred_curve, gt_curve):
    pred_x, gt_x = [], []
    pred_value, gt_value = [], []
    if "err" in gt_curve.keys():
        pred_error, gt_error = [], []
    else:
        pred_error, gt_error = None, None
    
    pred_paired = [False] * len(pred_curve["x"])
    for i in range(len(gt_curve["x"])):
        best_score = np.inf
        best_j = np.inf
        best_x_pairs = []
        for j in range(len(pred_curve["x"])):
            if pred_paired[j]:
                continue
            score, x_pairs = pair_score(pred_curve["x"][j], gt_curve["x"][i])
            if score < best_score:
                best_score = score
                best_j = j
                best_x_pairs = x_pairs
        if best_score < np.inf:
            if len(best_x_pairs) > 0:
                for gt, pred in best_x_pairs:
                    pred_x.append(pred)
                    gt_x.append(gt)
            pred_value.append(pred_curve["y"][best_j])
            gt_value.append(gt_curve["y"][i])
            pred_paired[best_j] = True
            if pred_error is not None:
                pred_error.append(pred_curve["err"][best_j])
                gt_error.append(gt_curve["err"][i])
    return {"x": pred_x, "y": pred_value, "err": pred_error}, {"x": gt_x, "y": gt_value, "err": gt_error}

'''
@description: Separate the data points into different curves based on the subplot and the type-2 value
@params: A DataFrame of the data points
@return: a dictionary of the curves in the format of 
{
    (subplot, type2): {
        "x": [], 
        "y": [], 
        "err": []
    }
}
'''
def separate_curve(df):
    cur_subplot, cur_type2 = None, None
    curves = {}
    for i in range(len(df)):
        if "Subplot Value" in df.columns and cur_subplot != df["Subplot Value"][i]:
            cur_subplot = df["Subplot Value"][i]
        if "Type-2" in df.columns and cur_type2 != df["Type-2"][i]:
            cur_type2 = df["Type-2"][i]
        if (cur_subplot, cur_type2) not in curves:
            curves[(cur_subplot, cur_type2)] = {"x": [], "y": []}
            if "Error Bar Length" in df.columns:
                curves[(cur_subplot, cur_type2)]["err"] = []
        curves[(cur_subplot, cur_type2)]["x"].append(df["Type-1"][i])
        curves[(cur_subplot, cur_type2)]["y"].append(df["Value"][i])
        if "Error Bar Length" in df.columns:
            curves[(cur_subplot, cur_type2)]["err"].append(df["Error Bar Length"][i])
    return curves

'''
@params:
    pred_curves: Dict, (subplot, type2) -> curve
    gt_curves: Dict, (subplot, type2) -> curve
@return:
    subplot_gt2pred: Dict, mapping a subplot value in the ground truth data to a subplot value in the predicted data
    type2_gt2pred: Dict, mapping a type2 value in the ground truth data to a type2 value in the predicted data
'''
def pair_curves(pred_curves, gt_curves):
    paired_gt_curves, paired_pred_curves = [], []
    paired = {}
    for subplot_gt, type2_gt in gt_curves.keys():
        best_score = np.inf
        best_subplot_pred, best_type2_pred = None, None
        for subplot_pred, type2_pred in pred_curves.keys():
            if (subplot_pred, type2_pred) in paired:
                continue
            subplot_score, _ = pair_score(subplot_pred, subplot_gt)
            type2_score, _ = pair_score(type2_pred, type2_gt)
            score = subplot_score + type2_score
            if score < best_score:
                best_score = score
                best_subplot_pred = subplot_pred
                best_type2_pred = type2_pred
        if best_score < np.inf:
            paired[(best_subplot_pred, best_type2_pred)] = (subplot_gt, type2_gt)
            paired_gt_curves.append(gt_curves[(subplot_gt, type2_gt)])
            paired_pred_curves.append(pred_curves[(best_subplot_pred, best_type2_pred)])

    return paired_pred_curves, paired_gt_curves

def cal_perf(pred_values, gt_values):
    print(len(pred_values), len(gt_values))
    pred_values = np.array(pred_values)
    gt_values = np.array(gt_values)
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_values - gt_values))
    # Mean Absolute Percentage Error
    if 0 in gt_values:
        mape = np.nan
    else:
        mape = np.mean(np.abs(pred_values - gt_values) / np.abs(gt_values))
    # Mean Absolute Percentage Error (epsilon = 1e-5)
    mape_eps = np.mean(np.abs(pred_values - gt_values) / (np.abs(gt_values) + 1e-5))
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(np.abs(pred_values - gt_values) / (np.abs(pred_values) + np.abs(gt_values))) * 2
    # Mean Absolute Scaled Error
    mase = np.mean(np.abs(pred_values - gt_values)) / (np.max(gt_values) - np.min(gt_values))
    # R-squared
    r_2 = 1 - np.sum((pred_values - gt_values) ** 2) / np.sum((gt_values - np.mean(gt_values)) ** 2)

    return {"MAE": mae, "MAPE": mape, "MAPE_eps": mape_eps, "SMAPE": smape, "MASE": mase, "R-squared": r_2}

def interpolate(pred_x, pred_y, gt_x, gt_y):
    pred_interp = interp1d(pred_x, pred_y, kind="cubic", fill_value="extrapolate")
    gt_interp = interp1d(gt_x, gt_y, kind="cubic", fill_value="extrapolate")
    x_common = np.linspace(min(min(pred_x), min(gt_x)), max(max(pred_x), max(gt_x)), num=50)
    pred_y_common = pred_interp(x_common)
    gt_y_common = gt_interp(x_common)
    return x_common, pred_y_common, gt_y_common

'''
@raise: 
    WrongCSVNumberError: the number of predicted CSV files is different from the number of ground truth CSV files
    FormatError: the predicted CSV file has an invalid column (valid columns are "Value", "Subplot Value", and "Type-{}")
'''
def evaluate_plot(pred_df, gt_df):
    if len(pred_df) == 1 and len(gt_df) == 1:
        pred_df = pred_df[0]
        gt_df = gt_df[0]
    else:
        raise WrongCSVNumberError(len(pred_df), len(gt_df))
    for col in pred_df.columns:
        if col not in ["Value", "Subplot Value", "Error Bar Length"] and not re.match(r"Type-\d+", col):
            raise FormatError(f"The column {col} in the predicted CSV file is not a valid column.")
    
    pred_curves = separate_curve(pred_df)
    gt_curves = separate_curve(gt_df)

    if len(gt_curves) <= len(pred_curves):
        paired_pred_curves, paired_gt_curves = pair_curves(pred_curves, gt_curves)
    else:
        paired_gt_curves, paired_pred_curves = pair_curves(gt_curves, pred_curves)

    new_pred_x, new_gt_x = [], []
    new_pred_y, new_gt_y, new_pred_err, new_gt_err = [], [], [], []
    discrete_identified = 0
    discrete_generated = sum([len(curve["x"]) for _, curve in pred_curves.items()])
    discrete_total = sum([len(curve["x"]) if len(curve["x"]) < 50 else 0 for _, curve in gt_curves.items()])
    
    for pred_curve, gt_curve in zip(paired_pred_curves, paired_gt_curves):
        if len(gt_curve["x"]) >= 50:
            # This is a continuous plot
            _, pred_y_resample, gt_y_resample = interpolate(pred_curve["x"], pred_curve["y"], gt_curve["x"], gt_curve["y"])
            new_pred_y.extend(pred_y_resample)
            new_gt_y.extend(gt_y_resample)
            discrete_generated -= len(pred_curve["x"])
            if "err" in gt_curve.keys() and len(gt_curve["err"]) > 0:
                _, pred_err_resample, gt_err_resample = interpolate(pred_curve["x"], pred_curve["err"], gt_curve["x"], gt_curve["err"])
                new_pred_err.extend(pred_err_resample)
                new_gt_err.extend(gt_err_resample)
        else:
            # This is a discrete plot
            if len(gt_curve["x"]) <= len(pred_curve["x"]):
                paired_pred_curve, paired_gt_curve = pair_data_points(pred_curve, gt_curve)
            else:
                paired_gt_curve, paired_pred_curve = pair_data_points(gt_curve, pred_curve)
            new_pred_x.extend(paired_pred_curve["x"])
            new_pred_y.extend(paired_pred_curve["y"])
            new_gt_x.extend(paired_gt_curve["x"])
            new_gt_y.extend(paired_gt_curve["y"])
            discrete_identified += len(paired_gt_curve["y"])
            if paired_pred_curve["err"] is not None:
                new_pred_err.extend(paired_pred_curve["err"])
                new_gt_err.extend(paired_gt_curve["err"])
    
    perf = {}
    if len(new_gt_x) > 0:
        perf["X performance"] = cal_perf(new_pred_x, new_gt_x)
    if len(new_gt_y) > 0:
        perf["Value performance"] = cal_perf(new_pred_y, new_gt_y)
    if len(new_gt_err) > 0:
        perf["Error performance"] = cal_perf(new_pred_err, new_gt_err)
    if len(new_gt_x) > 0 or len(new_gt_err) > 0:
        perf["Overall performance"] = cal_perf(new_pred_x + new_pred_y + new_pred_err, new_gt_x + new_gt_y + new_gt_err)
    if discrete_total > 0:
        perf["Identified rate"] = discrete_identified / discrete_total
        if discrete_generated > 0:
            perf["Identified recall"] = discrete_identified / discrete_generated
        else:
            perf["Identified recall"] = np.nan
    return perf

def is_repeat(value, repeat):
    if len(repeat) == 0 or len(value) % len(repeat) != 0:
        return False
    for i in range(1, len(value)//len(repeat)):
        if value != repeat * i:
            return False
    return True

def parse_digit_from_sig(value):
    # find the number at the beginning of the string
    match = re.match(r'^-?\d*\.?\d*', value)
    if not match:
        return None, None
    digit = match.group(0)
    if len(digit) == 0:
        return None, None
    if len(digit) == len(value):
        return digit, 0
    
    # find repeat in value[len(digit):]
    repeat = ""
    value = value[len(digit):]
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
                    if type(pred_df.iloc[j][i]) != str and (pred_df.iloc[j][i] == gt_df.iloc[j][i] or (np.isnan(pred_df.iloc[j][i]) and np.isnan(gt_df.iloc[j][i]))):
                        same += 1
                    else:
                        try:
                            pred_digit = float(pred_df.iloc[j][i])
                            if pred_digit == gt_df.iloc[j][i]:
                                same += 1
                            else:
                                print(BLUE + "Value mismatch: ", gt_df.columns[i], j, pred_df.iloc[j][i], gt_df.iloc[j][i], type(pred_df.iloc[j][i]), type(gt_df.iloc[j][i]), RESET)
                        except:
                            print(MAGENTA + "Type mismatch: ", gt_df.columns[i], j, pred_df.iloc[j][i], gt_df.iloc[j][i], type(pred_df.iloc[j][i]), type(gt_df.iloc[j][i]), RESET)
                    total += 1
                else:
                    gt_digit, gt_repeat = parse_digit_from_sig(gt_df.iloc[j][i])
                    if type(pred_df.iloc[j][i]) != str:
                        pred_digit = pred_df.iloc[j][i]
                        pred_repeat = 0
                    else:
                        pred_digit, pred_repeat = parse_digit_from_sig(pred_df.iloc[j][i])
                    try:
                        gt_digit = float(gt_digit)
                    except:
                        print("Not a data point ", gt_df.columns[i], j, gt_df.iloc[j][i])
                        continue
                    total += 1
                    try:
                        pred_digit = float(pred_digit)
                    except:
                        print(MAGENTA + "Type mismatch: ", gt_df.columns[i], j, pred_digit, gt_digit, pred_repeat, gt_repeat, RESET)
                        continue
                    if pred_digit == gt_digit and pred_repeat == gt_repeat:
                        same += 1
                    else:
                        print(BLUE + "Value mismatch: ", gt_df.columns[i], j, pred_df.iloc[j][i], gt_df.iloc[j][i], pred_digit, gt_digit, pred_repeat, gt_repeat, RESET)
    print("same: ", same, "total: ", total)
    return same / total

if __name__ == "__main__":
    pred = [pd.read_csv("Baseline/output/P-49-O6.csv")]
    gt = [pd.read_csv("/Users/zhangliyun/Documents/UIUC/figure/figure-to-data/49/P-49-O6.csv")]
    perf = evaluate_plot(pred, gt)
    print(perf)