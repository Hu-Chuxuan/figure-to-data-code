import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from Exceptions import WrongCSVNumberError, FormatError, YELLOW, BLUE, MAGENTA, RESET
from TableEvaluator import parse_range

def process(value):
    if type(value) == np.ndarray:
        return value
    if value is None or (type(value) == float and np.isnan(value)):
        return None
    if type(value) == str:
        if len(value) == 0:
            return None
        value = value.strip()
        start, end= parse_range(value)
        if start is not None and end is not None:
            return np.array([start, end], dtype=np.float64)
    try:
        value = float(value)
    except:
        pass
    return value

class Curve:
    def __setattr__(self, name: str, value: list) -> None:
        if value is None or len(value) == 0:
            super().__setattr__(name, None)
            return
        parsed_values = []
        any_str = False
        for i in range(len(value)):
            parsed_values.append(process(value[i]))
            if type(parsed_values[-1]) != type(parsed_values[0]):
                any_str = True
                break
        if not any_str:
            array = np.array(parsed_values, dtype=np.float64)
            super().__setattr__(name, array)
        else:
            array = []
            for value in parsed_values:
                if isinstance(value, np.ndarray):
                    assert value.shape == (2,)
                    str_val = f"{value[0]}-{value[1]}"
                else:
                    str_val = str(value)
                array.append(str_val)
            super().__setattr__(name, array)

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None
    
    def __len__(self):
        if self.y is None:
            return 0
        return len(self.y)

    def __init__(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)

    def select(self, indices):
        if indices is None:
            self.x = None
            self.y = None
            self.err = None
        if self.x is not None:
            if type(self.x) == np.ndarray and self.x.dtype != object:
                self.x = self.x[indices]
            else:
                self.x = None
        if self.y is not None:
            if type(self.y) == np.ndarray and self.y.dtype != object:
                self.y = self.y[indices]
            else:
                self.y = None
        if self.err is not None:
            if type(self.err) == np.ndarray and self.err.dtype != object:
                self.err = self.err[indices]
            else:
                self.err = None
    
    def getter(self, attributes):
        res = []
        for attr in attributes:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    if len(val.shape) == 1:
                        val = val.reshape(-1, 1)
                    res.append(val)
        return np.concatenate(res, axis=1)
    
    def interpolate(self, x_common):
        y = interp1d(self.x, self.y, kind="cubic", fill_value="extrapolate")
        self.y = y(x_common)
        if self.err is not None:
            err = interp1d(self.x, self.err, kind="cubic", fill_value="extrapolate")
            self.err = err(x_common)
        self.x = None

def cost_fn(pred, gt):
    return np.sum(np.abs(gt - pred))

def is_empty(value):
    if type(value) == str:
        return value == ""
    return value is None or (type(value) == float and np.isnan(value))

'''
@description: calculate the score of the pair of two values
@return: the score of the pair. np.inf if the two values are completely different strings
'''
def pair_score(pred, gt):
    if type(gt) == str:
        if type(pred) == np.ndarray:
            return np.inf
        return 0 if gt == pred else np.inf
    if is_empty(pred) and is_empty(gt):
        return 0
    if is_empty(pred) or is_empty(gt):
        return np.inf
    if type(pred) != type(gt):
        return np.inf
    return cost_fn(pred, gt)

'''
@params:
    pred_df: DataFrame, predicted data
    gt_df: DataFrame, ground truth data
@return:
    pred: List, predicted values or error bars
    gt: List, ground truth values or error bars
'''
def pair_data_points(pred_curve: Curve, gt_curve: Curve):
    cost_matrix = np.zeros((len(pred_curve), len(gt_curve)))
    for i in range(len(pred_curve)):
        for j in range(len(gt_curve)):
            cost_matrix[i, j] = pair_score(pred_curve.x[i], gt_curve.x[j])
    try:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
    except:
        row_idx = None
        col_idx = None
    pred_curve.select(row_idx)
    gt_curve.select(col_idx)

    return pred_curve, gt_curve

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
            if isinstance(cur_subplot, float) and np.isnan(cur_subplot):
                cur_subplot = None
        if "Type-2" in df.columns and cur_type2 != df["Type-2"][i]:
            cur_type2 = df["Type-2"][i]
            if isinstance(cur_type2, float) and np.isnan(cur_type2):
                cur_type2 = None
        if (cur_subplot, cur_type2) not in curves:
            curves[(cur_subplot, cur_type2)] = {"x": [], "y": [], "err": []}
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
    paired_curves: Dict, subplot_gt -> List of curve pairs (pred_curve, gt_curve)
'''
def pair_curves(pred_curves, gt_curves):
    paired_curves = {}
    paired = {}
    # pair_trace = ""
    for subplot_gt, type2_gt in gt_curves.keys():
        best_score = np.inf
        best_subplot_pred, best_type2_pred = None, None
        for subplot_pred, type2_pred in pred_curves.keys():
            if (subplot_pred, type2_pred) in paired:
                continue
            subplot_score = pair_score(process(subplot_pred), process(subplot_gt))
            type2_score = pair_score(process(type2_pred), process(type2_gt))
            score = subplot_score + type2_score
            if score < best_score:
                best_score = score
                best_subplot_pred = subplot_pred
                best_type2_pred = type2_pred
        if best_score < np.inf:
            paired[(best_subplot_pred, best_type2_pred)] = (subplot_gt, type2_gt)
            pred_curve = pred_curves[(best_subplot_pred, best_type2_pred)]
            # pair_trace += f"Pairing ({subplot_gt}, {best_subplot_pred}) and ({type2_gt}, {best_type2_pred})\n"
        else:
            pred_curve = {}
            # pair_trace += f"Curve ({subplot_gt}, {type2_gt}) cannot be paired with any curve in prediction\n"
        if subplot_gt not in paired_curves:
            paired_curves[subplot_gt] = []
        gt_curve = gt_curves[(subplot_gt, type2_gt)]
        paired_curves[subplot_gt].append((Curve(pred_curve), Curve(gt_curve)))
    
    # pair_trace += "Unpaired curves in prediction:\n"
    for subplot_pred, type2_pred in pred_curves.keys():
        if (subplot_pred, type2_pred) not in paired:
            pred_curve = pred_curves[(subplot_pred, type2_pred)]
            if subplot_pred not in paired_curves:
                paired_curves[subplot_pred] = []
            paired_curves[subplot_pred].append((Curve(pred_curve), Curve({})))
            # pair_trace += f"({subplot_pred}, {type2_pred}), "
    # pair_trace += "\n"

    return paired_curves

def merge_perf(perf_list):
    overall_perf = {}
    for perf in perf_list:
        for key in perf.keys():
            if key in overall_perf:
                continue
            if type(perf[key]) == dict:
                l = []
                for i in range(len(perf_list)):
                    if key in perf_list[i]:
                        l.append(perf_list[i][key])
                overall_perf[key] = merge_perf(l)
            else:
                res = []
                for i in range(len(perf_list)):
                    if key in perf_list[i] and not np.isnan(perf_list[i][key]):
                        res.append(perf_list[i][key])
                if len(res) == 0:
                    overall_perf[key] = np.nan
                else:
                    overall_perf[key] = np.mean(res)
    return overall_perf

'''
@params: curves_in_subplot: Dict, subplot -> List of {"pred": Curve, "gt": Curve}
'''
def cal_perf(curves_in_subplot):
    curve_level_perf = {}
    for subplot, curves in curves_in_subplot.items():
        curve_level_perf[subplot] = []
        for curve in curves:
            perf = {}
            if len(curve["gt"]) != 0 and len(curve["pred"]) != 0:
                if curve["gt"].x is not None:
                    perf["X performance"] = cal_metrics(curve["pred"].x, curve["gt"].x)
                perf["Value performance"] = cal_metrics(curve["pred"].y, curve["gt"].y)
                if curve["gt"].err is not None:
                    perf["Error performance"] = cal_metrics(curve["pred"].err, curve["gt"].err)
                if curve["gt"].x is not None or curve["gt"].err is not None:
                    perf["Overall performance"] = cal_metrics(
                        curve["pred"].getter(["x", "y", "err"]),
                        curve["gt"].getter(["x", "y", "err"])
                    )
            if "gt_len" in curve:
                if curve["gt_len"] == 0:
                    perf["DP accuracy"] = 0
                else:
                    perf["DP accuracy"] = min(len(curve["gt"]), len(curve["pred"])) / curve["gt_len"]
            if "pred_len" in curve:
                if curve["pred_len"] == 0:
                    perf["DP recall"] = 0
                else:
                    perf["DP recall"] = min(len(curve["gt"]), len(curve["pred"])) / curve["pred_len"]
            curve_level_perf[subplot].append(perf)
    subplot_level_perf = []
    for subplot, curve_perf in curve_level_perf.items():
        subplot_level_perf.append(merge_perf(curve_perf))
    return merge_perf(subplot_level_perf)

def cal_metrics(pred_values, gt_values):
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
    smape = np.mean(np.abs(pred_values - gt_values) / (np.abs(pred_values) + np.abs(gt_values) + 1e-5)) * 2
    # Mean Absolute Scaled Error
    mase = np.mean(np.abs(pred_values - gt_values)) / (np.max(gt_values) - np.min(gt_values))
    # R-squared
    r_2 = 1 - np.sum((pred_values - gt_values) ** 2) / np.sum((gt_values - np.mean(gt_values)) ** 2)

    return {"MAE": mae, "MAPE": mape, "MAPE_eps": mape_eps, "SMAPE": smape, "MASE": mase, "R-squared": r_2}

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
        paired_curves = pair_curves(pred_curves, gt_curves)
        pred_idx, gt_idx = 0, 1
    else:
        paired_curves = pair_curves(gt_curves, pred_curves)
        pred_idx, gt_idx = 1, 0

    if len(paired_curves) == 0:
        perf = {"DP accuracy": 0, "DP recall": 0, "Curve accuracy": 0, "Curve recall": 0}
        return perf

    paired_curve_num = 0
    for curves in paired_curves.values():
        for curve in curves:
            if len(curve[0]) != 0 and len(curve[1]) != 0:
                paired_curve_num += 1

    curves_in_subplot = {}
    for subplot_gt, curve_pairs in paired_curves.items():
        curves = []
        for curve_pair in curve_pairs:
            pred_curve, gt_curve = curve_pair[pred_idx], curve_pair[gt_idx]
            if len(pred_curve) == 0 or len(gt_curve) == 0:
                continue
            if len(gt_curve) >= 50:
                # This is a continuous plot
                x_common = np.linspace(
                    min(min(pred_curve.x), min(gt_curve.x)), 
                    max(max(pred_curve.x), max(gt_curve.x)), num=50)
                pred_curve.interpolate(x_common)
                gt_curve.interpolate(x_common)
                curves.append({"pred": pred_curve, "gt": gt_curve})
            else:
                # This is a discrete plot
                gt_len = len(gt_curve)
                pred_len = len(pred_curve)
                if gt_len <= pred_len:
                    paired_pred_curve, paired_gt_curve = pair_data_points(pred_curve, gt_curve)
                else:
                    paired_gt_curve, paired_pred_curve = pair_data_points(gt_curve, pred_curve)
                curves.append({
                    "pred": paired_pred_curve, "gt": paired_gt_curve,
                    "gt_len": gt_len, "pred_len": pred_len
                })
        curves_in_subplot[subplot_gt] = curves
    perf = cal_perf(curves_in_subplot)
    perf["Curve accuracy"] = paired_curve_num / len(gt_curves)
    if len(pred_curves) == 0:
        perf["Curve recall"] = 0
    else:
        perf["Curve recall"] = paired_curve_num / len(pred_curves)
    return perf
