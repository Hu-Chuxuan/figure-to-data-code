import numpy as np

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
                pred_value.append(pred_df.iloc[j])
                gt_value.append(gt_df.iloc[i])
                pred_paired[j] = True
                break
    return pred_value, gt_value, pred_error, gt_error

def cal_perf(pred_values, gt_values):
    pred_values = np.array(pred_values)
    gt_values = np.array(gt_values)
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_values - gt_values))
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs(pred_values - gt_values) / gt_values)
    # Mean Absolute Percentage Error (epsilon = 1e-5)
    mape_eps = np.mean(np.abs(pred_values - gt_values) / (gt_values + 1e-5))
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(np.abs(pred_values - gt_values) / (np.abs(pred_values) + np.abs(gt_values))) * 2
    # Mean Absolute Scaled Error
    mase = np.mean(np.abs(pred_values - gt_values)) / (np.max(gt_values) - np.min(gt_values))
    # R-squared
    r_2 = 1 - np.sum((pred_values - gt_values) ** 2) / np.sum((gt_values - np.mean(gt_values)) ** 2)

    return {"MAE": mae, "MAPE": mape, "MAPE_eps": mape_eps, "SMAPE": smape, "MASE": mase, "R-squared": r_2}

def evaluate_plot(pred_df, gt_df):
    pred, gt = pair_data_points(pred_df, gt_df)
    overall_perf = cal_perf(pred, gt)
    if "Error Bar Length" in gt_df.columns:
        overall_perf["Identified rate"] = len(pred) / (len(gt_df) * 2)
    else:
        overall_perf["Identified rate"] = len(pred) / len(gt_df)
    return overall_perf