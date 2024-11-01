import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from Exceptions import WrongCSVNumberError, FormatError, YELLOW, BLUE, MAGENTA, RESET
from TableEvaluator import parse_range

USE_MASE = False
DEBUG = False

'''
Potential types of outputs: float, ndarray, str, None
'''
def process(value):
    if type(value) == np.ndarray:
        return value
    if type(value) == tuple:
        return tuple([process(val) for val in value])
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
        any_str = any_str or type(parsed_values[0]) == str
        if name != "x":
            array = np.array(parsed_values, dtype=np.float64)
            super().__setattr__(name, array)
        else:
            super().__setattr__(name, parsed_values)
    
    def get_clean_attr(self, attr):
        if self.__getattribute__(attr) is not None:
            new_attr = []
            for x in self.__getattribute__(attr):
                if isinstance(x, np.ndarray):
                    new_attr.extend(x.copy().flatten().astype(float))
                elif isinstance(x, str) or x is None:
                    continue
                elif isinstance(x, tuple) or isinstance(x, list):
                    for val in x:
                        new_attr.append(float(val))
                else:
                    new_attr.append(float(x))
            if len(new_attr) == 0:
                new_attr = None
            else:
                new_attr = np.array(new_attr, dtype=np.float64)
            return new_attr
        return None

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
            self.x = [self.x[i] for i in indices]
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

def str_cost_fn(pred, gt):
    pred = pred.strip().lower()
    gt = gt.strip().lower()
    if pred == gt:
        return 0
    if len(pred) <= len(gt) and gt.endswith(pred):
        return len(gt) - len(pred)
    if len(gt) <= len(pred) and pred.endswith(gt):
        return len(pred) - len(gt)
    # remove puctuations with 0.5 cost each chacacter
    pred_ptr, gt_ptr = 0, 0
    cost = 0
    while pred_ptr < len(pred) or gt_ptr < len(gt):
        if pred_ptr < len(pred) and gt_ptr < len(gt) and pred[pred_ptr] == gt[gt_ptr]:
            pred_ptr += 1
            gt_ptr += 1
        elif pred_ptr < len(pred) and not pred[pred_ptr].isalpha() and not pred[pred_ptr].isdigit():
            pred_ptr += 1
            cost += 0.5
        elif gt_ptr < len(gt) and not gt[gt_ptr].isalpha() and not gt[gt_ptr].isdigit():
            gt_ptr += 1
            cost += 0.5
        else:
            return np.inf
    return cost

'''
@description: calculate the score of the pair of two values
@return: the score of the pair. np.inf if the two values are completely different strings
'''
def pair_score(pred, gt):
    if is_empty(pred) and is_empty(gt):
        return 0
    if is_empty(pred) or is_empty(gt):
        return np.inf
    if type(pred) != type(gt):
        return np.inf
    if type(gt) == str:
        return str_cost_fn(pred, gt)
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
    row_candidates = []
    col_candidates = []
    for i in range(len(pred_curve)):
        for j in range(len(gt_curve)):
            if pair_score(pred_curve.x[i], gt_curve.x[j]) != np.inf:
                row_candidates.append(i)
                col_candidates.append(j)
    row_candidates = sorted(list(set(row_candidates)))
    col_candidates = sorted(list(set(col_candidates)))

    if USE_MASE and type(gt_curve.x[0]) != str:
        x_matrix = np.zeros((len(row_candidates), len(col_candidates)))
        for i in range(len(row_candidates)):
            for j in range(len(col_candidates)):
                x_matrix[i, j] = pair_score(pred_curve.x[row_candidates[i]], gt_curve.x[col_candidates[j]])
            gt_x = gt_curve.get_clean_attr("x")
            x_matrix /= np.max(gt_x) - np.min(gt_x)

        y_matrix = np.zeros((len(row_candidates), len(col_candidates)))
        for i in range(len(row_candidates)):
            for j in range(len(col_candidates)):
                y_matrix[i, j] = pair_score(pred_curve.y[row_candidates[i]], gt_curve.y[col_candidates[j]])
        y_matrix /= np.max(gt_curve.y) - np.min(gt_curve.y)

        cost_matrix = x_matrix + y_matrix
        if pred_curve.err is not None and gt_curve.err is not None:
            err_matrix = np.zeros((len(row_candidates), len(col_candidates)))
            for i in range(len(row_candidates)):
                for j in range(len(col_candidates)):
                    err_matrix[i, j] = pair_score(pred_curve.err[row_candidates[i]], gt_curve.err[col_candidates[j]])
            err_matrix /= np.max(gt_curve.err) - np.min(gt_curve.err)
            cost_matrix += err_matrix
    else:
        cost_matrix = np.zeros((len(row_candidates), len(col_candidates)))
        for i in range(len(row_candidates)):
            for j in range(len(col_candidates)):
                cost_matrix[i, j] = pair_score(pred_curve.x[row_candidates[i]], gt_curve.x[col_candidates[j]])
    try:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
    except:
        row_idx = None
        col_idx = None
    if row_idx is not None:
        for i in range(len(row_idx)):
            row_idx[i] = row_candidates[row_idx[i]]
        for i in range(len(col_idx)):
            col_idx[i] = col_candidates[col_idx[i]]
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
    num = 0
    for i in range(len(df)):
        if "Subplot Value" in df.columns and cur_subplot != df["Subplot Value"][i]:
            cur_subplot = df["Subplot Value"][i]
            if isinstance(cur_subplot, float) and np.isnan(cur_subplot):
                cur_subplot = None
        if "Type-2" in df.columns and cur_type2 != df["Type-2"][i]:
            cur_type2 = df["Type-2"][i]
            if isinstance(cur_type2, float) and np.isnan(cur_type2):
                cur_type2 = None
        if cur_subplot not in curves:
            curves[cur_subplot] = {}
        if cur_type2 not in curves[cur_subplot]:
            curves[cur_subplot][cur_type2] = {"x": [], "y": [], "err": []}
            num += 1
        curves[cur_subplot][cur_type2]["x"].append(df["Type-1"][i])
        curves[cur_subplot][cur_type2]["y"].append(df["Value"][i])
        if "Error Bar Length" in df.columns:
            curves[cur_subplot][cur_type2]["err"].append(df["Error Bar Length"][i])
        elif "Error Bar Length 1" in df.columns and "Error Bar Length 2" in df.columns:
            curves[cur_subplot][cur_type2]["err"].append((df["Error Bar Length 1"][i], df["Error Bar Length 2"][i]))
    return curves, num

'''
@params:
    pred_curves: Dict, (subplot, type2) -> curve
    gt_curves: Dict, (subplot, type2) -> curve
@return:
    paired_curves: Dict, subplot_gt -> List of curve pairs (pred_curve, gt_curve)
'''
def pair_curves(pred_curves, gt_curves):
    unpaired_gt_subplot = []
    unpaired_pred_subplot = []

    paired_curves = {}
    paired_subplots = {}
    for subplot_gt in gt_curves.keys():
        best_score = np.inf
        best_subplot_pred = None
        for subplot_pred in pred_curves.keys():
            if subplot_pred in paired_subplots:
                continue
            score = pair_score(subplot_pred, subplot_gt)
            if score < best_score:
                best_score = score
                best_subplot_pred = subplot_pred
        if best_score < np.inf:
            paired_subplots[best_subplot_pred] = subplot_gt
        else:
            if subplot_gt not in unpaired_gt_subplot:
                unpaired_gt_subplot.append(subplot_gt)
            paired_curves[subplot_gt] = []
            for type2_gt in gt_curves[subplot_gt].keys():
                gt_curve = gt_curves[subplot_gt][type2_gt]
                pred_curve = {}
                paired_curves[subplot_gt].append((Curve(pred_curve), Curve(gt_curve)))
    
    for subplot_pred in pred_curves.keys():
        if subplot_pred not in paired_subplots:
            unpaired_pred_subplot.append(subplot_pred)
            paired_curves[subplot_pred] = []
            for type2_pred in pred_curves[subplot_pred].keys():
                pred_curve = pred_curves[subplot_pred][type2_pred]
                gt_curve = {}
                paired_curves[subplot_pred].append((Curve(pred_curve), Curve(gt_curve)))
    
    for subplot_pred, subplot_gt in paired_subplots.items():
        paired = {}
        unpaired_gt_curve, unpaired_pred_curve = [], []
        if len(gt_curves[subplot_gt]) == 1 and len(pred_curves[subplot_pred]) == 1:
            paired_curves[subplot_gt] = [
                (Curve(list(pred_curves[subplot_pred].values())[0]), 
                 Curve(list(gt_curves[subplot_gt].values())[0]))
            ]
            continue
        for type2_gt in gt_curves[subplot_gt].keys():
            best_score = np.inf
            best_type2_pred = None
            for type2_pred in pred_curves[subplot_pred].keys():
                if type2_pred in paired:
                    continue
                score = pair_score(process(type2_pred), process(type2_gt))
                if score < best_score:
                    best_score = score
                    best_type2_pred = type2_pred
            if best_score < np.inf:
                paired[best_type2_pred] = type2_gt
                pred_curve = pred_curves[subplot_pred][best_type2_pred]
            else:
                pred_curve = {}
                unpaired_gt_curve.append(type2_gt)
            if subplot_gt not in paired_curves:
                paired_curves[subplot_gt] = []
            gt_curve = gt_curves[subplot_gt][type2_gt]
            paired_curves[subplot_gt].append((Curve(pred_curve), Curve(gt_curve)))
        
        for type2_pred in pred_curves[subplot_pred].keys():
            if type2_pred not in paired:
                unpaired_pred_curve.append(type2_pred)
                gt_curve = {}
                paired_curves[subplot_gt].append((Curve(pred_curves[subplot_pred][type2_pred]), Curve(gt_curve)))
        
        if DEBUG:
            paired = {k: v for k, v in paired.items() if k != v}
            if len(paired) > 0:
                print(BLUE + "Paired type2:", paired, RESET)
            if len(unpaired_gt_curve) > 0:
                print(MAGENTA + "Unpaired gt curve:", unpaired_gt_curve, RESET)
            if len(unpaired_pred_curve) > 0:
                print(MAGENTA + "Unpaired pred curve:", unpaired_pred_curve, RESET)
    if DEBUG:
        paired_subplots = {k: v for k, v in paired_subplots.items() if k != v}
        if len(paired_subplots) > 0:
            print(BLUE + "Paired subplots:", paired_subplots, RESET)
        if len(unpaired_gt_subplot) > 0:
            print(MAGENTA + "Unpaired gt subplot:", unpaired_gt_subplot, RESET)
        if len(unpaired_pred_subplot) > 0:
            print(MAGENTA + "Unpaired pred subplot:", unpaired_pred_subplot, RESET)


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
                pred_x = curve["pred"].get_clean_attr("x")
                gt_x = curve["gt"].get_clean_attr("x")
                if gt_x is not None:
                    perf["X performance"] = cal_metrics(pred_x, gt_x, curve["x scale"], curve["x mean"])
                perf["Value performance"] = cal_metrics(curve["pred"].y, curve["gt"].y, curve["y scale"], curve["y mean"])
                if curve["gt"].err is not None:
                    if curve["pred"].err is None:
                        curve["pred"].err = np.zeros_like(curve["gt"].err)
                    perf["Error performance"] = cal_metrics(curve["pred"].err, curve["gt"].err, curve["err scale"], curve["err mean"])
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

def cal_metrics(pred_values, gt_values, scale, mean):
    perf = {}
    if len(pred_values.shape) == 1 and len(gt_values.shape) == 2:
        pred_values = pred_values.reshape(-1, 1)
    elif len(pred_values.shape) == 2 and len(gt_values.shape) == 1:
        gt_values = gt_values.reshape(-1, 1)
        scale = np.array([scale]).reshape(-1, 1)
        mean = np.array([mean]).reshape(-1, 1)
    # Mean Absolute Error
    perf["MAE"] = np.mean(np.abs(pred_values - gt_values))
    # Mean Absolute Percentage Error
    if 0 not in gt_values:
        perf["MAPE"] = np.mean(np.abs(pred_values - gt_values) / np.abs(gt_values))
    # Mean Absolute Percentage Error (epsilon = 1e-5)
    perf["MAPE_eps"] = np.mean(np.abs(pred_values - gt_values) / (np.abs(gt_values) + 1e-5))
    # Symmetric Mean Absolute Percentage Error
    perf["SMAPE"] = np.mean(np.abs(pred_values - gt_values) / (np.abs(pred_values) + np.abs(gt_values) + 1e-5)) * 2
    # Mean Absolute Scaled Error
    if (len(gt_values.shape) == 1 and scale != 0) or (len(gt_values.shape) > 1 and scale[0] != 0):
        perf["MASE"] = np.mean(np.abs(pred_values - gt_values) / scale)
    # R-squared
    if len(gt_values.shape) == 1:
        r2_denom = np.sum((gt_values - mean) ** 2)
    else:
        r2_denom = np.sum((gt_values - mean) ** 2, axis=0)
    if (len(gt_values.shape) == 1 and r2_denom != 0) or (len(gt_values.shape) > 1 and r2_denom[0] != 0):
        perf["R-squared"] = 1 - np.sum((pred_values - gt_values) ** 2 / r2_denom)

    return perf

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
        if col not in ["Value", "Subplot Value", "Error Bar Length"] and not re.match(r"Type-\d+", col) and not re.match(r"Error Bar Length \d+", col):
            raise FormatError(f"The column {col} in the predicted CSV file is not a valid column.")
    
    pred_curves, pred_num = separate_curve(pred_df)
    gt_curves, gt_num = separate_curve(gt_df)

    # if len(gt_curves) <= len(pred_curves):
    paired_curves = pair_curves(pred_curves, gt_curves)
    pred_idx, gt_idx = 0, 1
    # else:
    #     paired_curves = pair_curves(gt_curves, pred_curves)
    #     pred_idx, gt_idx = 1, 0

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
                curves.append({"pred": pred_curve, "gt": gt_curve, "gt_len": len(gt_curve), "pred_len": len(pred_curve)})
                continue
            gt_x = gt_curve.get_clean_attr("x")
            if len(gt_curve.x) >= 50 and gt_x is not None and len(gt_x) >= 50:
                # This is a continuous plot
                x_common = np.linspace(
                    min(min(pred_curve.x), min(gt_curve.x)), 
                    max(max(pred_curve.x), max(gt_curve.x)), num=50)
                pred_curve.interpolate(x_common)
                gt_curve.interpolate(x_common)
                curves.append({"pred": pred_curve, "gt": gt_curve, "y scale": np.max(gt_curve.y) - np.min(gt_curve.y), "y mean": np.mean(gt_curve.y)})
                if gt_curve.err is not None:
                    if len(gt_curve.err.shape) == 1:
                        curves[-1]["err scale"] = np.max(gt_curve.err) - np.min(gt_curve.err)
                        curves[-1]["err mean"] = np.mean(gt_curve.err)
                    else:
                        curves[-1]["err scale"] = np.max(gt_curve.err, axis=0) - np.min(gt_curve.err, axis=0)
                        curves[-1]["err mean"] = np.mean(gt_curve.err, axis=0)
            else:
                # This is a discrete plot
                curve = {"pred_len": len(pred_curve), "gt_len": len(gt_curve)}
                gt_x = gt_curve.get_clean_attr("x")
                if gt_x is not None:
                    curve["x scale"] = np.max(gt_x) - np.min(gt_x)
                    curve["x mean"] = np.mean(gt_x)
                if gt_curve.y is not None:
                    curve["y scale"] = np.max(gt_curve.y) - np.min(gt_curve.y)
                    curve["y mean"] = np.mean(gt_curve.y)
                if gt_curve.err is not None:
                    if len(gt_curve.err.shape) == 1:
                        curve["err scale"] = np.max(gt_curve.err) - np.min(gt_curve.err)
                        curve["err mean"] = np.mean(gt_curve.err)
                    else:
                        curve["err scale"] = np.max(gt_curve.err, axis=0) - np.min(gt_curve.err, axis=0)
                        curve["err mean"] = np.mean(gt_curve.err, axis=0)
                
                if curve["pred_len"] < curve["gt_len"]:
                    paired_pred_curve, paired_gt_curve = pair_data_points(pred_curve, gt_curve)
                else:
                    paired_gt_curve, paired_pred_curve = pair_data_points(gt_curve, pred_curve)
                curve["pred"] = paired_pred_curve
                curve["gt"] = paired_gt_curve
                curves.append(curve)
        curves_in_subplot[subplot_gt] = curves
    perf = cal_perf(curves_in_subplot)
    perf["Curve accuracy"] = paired_curve_num / gt_num
    if len(pred_curves) == 0:
        perf["Curve recall"] = 0
    else:
        perf["Curve recall"] = paired_curve_num / pred_num
    return perf
