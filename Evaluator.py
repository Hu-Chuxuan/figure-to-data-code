import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

def is_pair(pred_row, gt_row):
    if "Subplot Value" in gt_row and pred_row["Subplot Value"] != gt_row["Subplot Value"]:
        return False
    if pred_row["Type-1"] != gt_row["Type-1"]:
        return False
    if "Type-2" in gt_row and pred_row["Type-2"] != gt_row["Type-2"]:
        return False
    return True

'''
@params:
    pred_df: DataFrame, predicted data
    gt_df: DataFrame, ground truth data
@return:
    pred: List, predicted values or error bars
    gt: List, ground truth values or error bars

(pred, gt) are the paired data points. They might have less data points than the original data.
'''
def pair_data_points(pred_df, gt_df):
    pred_value, gt_value = [], []
    if "Error Bar Length" in gt_df.columns:
        pred_error, gt_error = [], []
    else:
        pred_error, gt_error = None, None
    
    pred_paired = [False] * len(pred_df)
    for i in range(len(gt_df)):
        for j in range(len(pred_df)):
            if not pred_paired[j] and is_pair(pred_df.iloc[j], gt_df.iloc[i]):
                pred_value.append(pred_df["Value"][j])
                pred_error.append(pred_df["Error Bar Length"][j])
                gt_value.append(gt_df["Value"][i])
                gt_error.append(gt_df["Error Bar Length"][i])
                pred_paired[j] = True
                break
    return pred_value, gt_value, pred_error, gt_error

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

def evaluate_discrete_plot(pred_df, gt_df):
    if len(pred_df) == 1 and len(gt_df) == 1:
        pred_df = pred_df[0]
        gt_df = gt_df[0]
    else:
        raise ValueError("The number of predicted data and the ground truth data is more than 1.")
    # Check they have the same header
    if len(pred_df.columns) != len(gt_df.columns):
        raise ValueError("The number of columns in the predicted data and the ground truth data is different.")
    gt_header = sorted(gt_df.columns)
    pred_header = sorted(pred_df.columns)
    for i in range(len(gt_header)):
        if gt_header[i] != pred_header[i]:
            raise ValueError("The header of the predicted data and the ground truth data is different.")
    
    pred_value, gt_value, pred_error, gt_error = pair_data_points(pred_df, gt_df)
    perf = {}
    perf["Value performance"] = cal_perf(pred_value, gt_value)
    if pred_error is not None:
        perf["Error performance"] = cal_perf(pred_error, gt_error)
        perf["Overall performance"] = cal_perf(pred_value + pred_error, gt_value + gt_error)
    perf["Identified rate"] = len(gt_value) / len(gt_df)
    perf["Identified recall"] = len(gt_value) / len(pred_value)
    
    return perf

def separate_curve(df):
    cur_subplot, cur_type2 = None, None
    curves = {}
    for i in range(len(df)):
        if "Subplot Value" in df.columns and cur_subplot != df["Subplot Value"][i]:
            cur_subplot = df["Subplot Value"][i]
        if "Type-2" in df.columns and cur_type2 != df["Type-2"][i]:
            cur_type2 = df["Type-2"][i]
        if (cur_subplot, cur_type2) not in curves:
            curves[(cur_subplot, cur_type2)] = {"x": [], "y": [], "err": []}
            if "Error Bar Length" in df.columns:
                curves[(cur_subplot, cur_type2)]["err"] = []
        curves[(cur_subplot, cur_type2)]["x"].append(df["Type-1"][i])
        curves[(cur_subplot, cur_type2)]["y"].append(df["Value"][i])
        if "Error Bar Length" in df.columns:
            curves[(cur_subplot, cur_type2)]["err"].append(df["Error Bar Length"][i])
    return curves

def interpolate(pred_x, pred_y, gt_x, gt_y):
    pred_interp = interp1d(pred_x, pred_y, kind="cubic", fill_value="extrapolate")
    gt_interp = interp1d(gt_x, gt_y, kind="cubic", fill_value="extrapolate")
    x_common = np.linspace(min(min(pred_x), min(gt_x)), max(max(pred_x), max(gt_x)), num=50)
    pred_y_common = pred_interp(x_common)
    gt_y_common = gt_interp(x_common)
    return x_common, pred_y_common, gt_y_common

def pair_string(pred_str, gt_str):
    gt2pred = {}
    paired = [False] * len(pred_str)
    for gt in gt_str:
        for i, pred in enumerate(pred_str):
            if not paired[i] and ((gt == None and pred == None) or (gt != None and pred != None and gt.lower() == pred.lower())):
                gt2pred[gt] = pred
                paired[i] = True
                break
    return gt2pred

def pair_curves(pred_curves, gt_curves):
    if len(pred_curves) != len(gt_curves):
        raise ValueError("The number of curves in the predicted data and the ground truth data is different.")
    gt_subplots = set([subplot for subplot, _ in gt_curves.keys()])
    gt_type2s = set([type2 for _, type2 in gt_curves.keys()])
    pred_subplots = set([subplot for subplot, _ in pred_curves.keys()])
    pred_type2s = set([type2 for _, type2 in pred_curves.keys()])
    
    subplot_gt2pred = pair_string(pred_subplots, gt_subplots)
    type2_gt2pred = pair_string(pred_type2s, gt_type2s)

    return subplot_gt2pred, type2_gt2pred

def evaluate_continuous_plot(pred_df, gt_df):
    pred_curves = separate_curve(pred_df)
    gt_curves = separate_curve(gt_df)
    print(len(pred_curves), len(gt_curves))
    print(pred_curves.keys(), gt_curves.keys())
    
    subplot_gt2pred, type2_gt2pred = pair_curves(pred_curves, gt_curves)

    new_pred_y, new_gt_y, new_pred_err, new_gt_err = [], [], [], []
    for subplot, type2 in gt_curves.keys():
        gt_key = (subplot, type2)
        pred_key = (subplot_gt2pred[subplot], type2_gt2pred[type2])
        _, pred_y_resample, gt_y_resample = interpolate(pred_curves[pred_key]["x"], pred_curves[pred_key]["y"], gt_curves[gt_key]["x"], gt_curves[gt_key]["y"])
        new_pred_y.extend(pred_y_resample)
        new_gt_y.extend(gt_y_resample)
        if len(gt_curves[gt_key]["err"]) > 0:
            _, pred_err_resample, gt_err_resample = interpolate(pred_curves[pred_key]["x"], pred_curves[pred_key]["err"], gt_curves[gt_key]["x"], gt_curves[gt_key]["err"])
            new_pred_err.extend(pred_err_resample)
            new_gt_err.extend(gt_err_resample)
    
    perf = {}
    perf["Value performance"] = cal_perf(new_pred_y, new_gt_y)
    if len(new_gt_err) > 0:
        perf["Error performance"] = cal_perf(new_pred_err, new_gt_err)
        perf["Overall performance"] = cal_perf(new_pred_y + new_pred_err, new_gt_y + new_gt_err)
    return perf

def is_repeat(value, repeat):
    if len(repeat) == 0 or len(value) % len(repeat) != 0:
        return False
    for i in range(1, len(value)//len(repeat)):
        if value != repeat * i:
            return False
    return True

def parse_digit_from_sig(value):
    digit = ""
    for ch in value:
        if ch == ".":
            if "." not in digit:
                digit += "."
            else:
                break
        elif ch.isdigit():
            digit += ch
        elif ch == "-" and len(digit) == 0:
            digit += "-"
        else:
            break
    if len(digit) == 0:
        return None, None
    if len(digit) == len(value):
        return digit, 0
    # find repeat in value[len(digit):]
    repeat = ""
    value = value[len(digit):]
    if "{" in value and "}" in value:
        repeat = value[value.find("{")+1:value.find("}")]
    for ch in value:
        if not is_repeat(value, repeat):
            repeat += ch
        else:
            break
    return digit, len(value) // len(repeat)

def evaluate_table(pred_dfs, gt_dfs):
    if len(pred_dfs) != len(gt_dfs):
        raise ValueError("The number of predicted data and the ground truth data is different.")
    
    same = 0
    total = 0
    
    for df_ptr in range(len(pred_dfs)):
        pred_df = pred_dfs[df_ptr]
        gt_df = gt_dfs[df_ptr]

        for i in range(len(gt_df.columns)):
            for j in range(len(gt_df)):
                if type(gt_df.iloc[j][i]) != str:
                    if type(pred_df.iloc[j][i]) != str and (pred_df.iloc[j][i] == gt_df.iloc[j][i] or (np.isnan(pred_df.iloc[j][i]) and np.isnan(gt_df.iloc[j][i]))):
                        same += 1
                    else:
                        print("Mismatch: ", gt_df.columns[i], j, pred_df.iloc[j][i], gt_df.iloc[j][i], type(pred_df.iloc[j][i]), type(gt_df.iloc[j][i]))
                    total += 1
                else:
                    gt_digit, gt_repeat = parse_digit_from_sig(gt_df.iloc[j][i])
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
                        print("Mismatch after parsing: ", gt_df.columns[i], j, pred_digit, gt_digit, pred_repeat, gt_repeat)
                        continue
                    if pred_digit == gt_digit and pred_repeat == gt_repeat:
                        same += 1
    print("same: ", same, "total: ", total)
    return same / total
