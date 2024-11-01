import pandas as pd
import unittest
import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from PlotEvaluator import evaluate_plot, cal_perf, merge_perf, cal_metrics, pair_score, pair_curves, pair_data_points, Curve, process

class TestPlotEvaluator(unittest.TestCase):
    def assertSameDict(self, pred, gt):
        self.assertEqual(set(pred.keys()), set(gt.keys()))
        for key in gt.keys():
            if key not in pred:
                self.fail(f"Key {key} not in pred")
            if type(gt[key]) == dict:
                self.assertSameDict(pred[key], gt[key])
            elif np.isnan(gt[key]):
                self.assertTrue(np.isnan(pred[key]))
            else:
                self.assertAlmostEqual(pred[key], gt[key])

class TestPlotEvaluatorCalPerf(TestPlotEvaluator):
    def test_merge_perf(self):
        perf_list = [
            {
                "Value performance": {"MAE": 0.1, "MSE": 0.01},
                "X performance": {"MAE": 0.2, "MSE": 0.02},
                "Error performance": {"MAE": 0.3, "MSE": 0.03},
                "Overall performance": {"MAE": 0.4, "MSE": 0.04},
                "DP accuracy": 0.5,
                "DP recall": 0.6
            },
            {
                "Value performance": {"MAE": 0.11, "MSE": 0.011},
                "X performance": {"MAE": 0.22, "MSE": 0.022},
                "Error performance": {"MAE": 0.33, "MSE": 0.033},
                "Overall performance": {"MAE": 0.44, "MSE": 0.044},
                "DP accuracy": 0.55,
                "DP recall": 0.66
            },
            {
                "Value performance": {"MAE": 0.111, "MSE": 0.0111},
                "X performance": {"MAE": 0.222, "MSE": 0.0222},
                "Error performance": {"MAE": 0.333, "MSE": 0.0333},
                "Overall performance": {"MAE": 0.444, "MSE": 0.0444},
                "DP accuracy": 0.555,
                "DP recall": 0.666
            },
        ]
        ans = {
            "Value performance": {"MAE": 0.107, "MSE": 0.0107},
            "X performance": {"MAE": 0.214, "MSE": 0.0214},
            "Error performance": {"MAE": 0.321, "MSE": 0.0321},
            "Overall performance": {"MAE": 0.428, "MSE": 0.0428},
            "DP accuracy": 0.535,
            "DP recall": 0.642
        }
        self.assertSameDict(merge_perf(perf_list), ans)
    
    def test_merge_perf_missing_parts(self):
        perf_list = [
            {
                "Value performance": {"MAE": 0.1, "MSE": 0.01},
                "X performance": {"MAE": 0.2, "MSE": 0.02},
                "Error performance": {"MAE": 0.3, "MSE": 0.03},
                "Overall performance": {"MAE": 0.4, "MSE": 0.04},
                "DP accuracy": 0.5,
                "DP recall": 0.6
            },
            {
                "Value performance": {"MAE": 0.11, "MSE": 0.011},
                "X performance": {"MAE": 0.22, "MSE": 0.022},
                "Overall performance": {"MAE": 0.44, "MSE": 0.044},
                "DP recall": 0.66
            },
            {
                "Value performance": {"MAE": 0.111, "MSE": 0.0111},
                "Error performance": {"MAE": 0.333, "MSE": 0.0333},
                "Overall performance": {"MAE": 0.444, "MSE": 0.0444},
                "DP accuracy": 0.555,
            },
        ]
        ans = {
            "Value performance": {"MAE": 0.107, "MSE": 0.0107},
            "X performance": {"MAE": 0.21, "MSE": 0.021},
            "Error performance": {"MAE": 0.3165, "MSE": 0.03165},
            "Overall performance": {"MAE": 0.428, "MSE": 0.0428},
            "DP accuracy": 0.5275,
            "DP recall": 0.63
        }
        self.assertSameDict(merge_perf(perf_list), ans)

    def test_merge_perf_nan(self):
        perf_list = [
            {
                "Value performance": {"MAE": 0.1, "MSE": 0.01},
                "X performance": {"MAE": 0.2, "MSE": 0.02},
                "Error performance": {"MAE": 0.3, "MSE": 0.03},
                "Overall performance": {"MAE": 0.4, "MSE": 0.04},
                "DP accuracy": np.nan,
                "DP recall": 0.6
            },
            {
                "Value performance": {"MAE": 0.11, "MSE": 0.011},
                "X performance": {"MAE": np.nan, "MSE": 0.022},
                "Error performance": {"MAE": 0.33, "MSE": 0.033},
                "Overall performance": {"MAE": 0.44, "MSE": np.nan},
                "DP accuracy": np.nan,
                "DP recall": np.nan
            },
            {
                "Value performance": {"MAE": np.nan, "MSE": 0.0111},
                "X performance": {"MAE": 0.222, "MSE": 0.0222},
                "Error performance": {"MAE": 0.333, "MSE": np.nan},
                "Overall performance": {"MAE": 0.444, "MSE": np.nan},
                "DP accuracy": np.nan,
                "DP recall": 0.666
            },
        ]
        ans = {
            "Value performance": {"MAE": 0.105, "MSE": 0.0107},
            "X performance": {"MAE": 0.211, "MSE": 0.0214},
            "Error performance": {"MAE": 0.321, "MSE": 0.0315},
            "Overall performance": {"MAE": 0.428, "MSE": 0.04},
            "DP accuracy": np.nan,
            "DP recall": 0.633
        }
        self.assertSameDict(merge_perf(perf_list), ans)

    def test_cal_perf(self):
        pred_value = np.array([-0.15, -0.1, -0.05, 0.0, 0.1, 0.05])
        pred_err = np.array([0.1, 0.08, 0.12, 0.07, 0.09, 0.06])
        gt_value = np.array([-0.1957403651115618, -0.3600405679513184, -0.061866125760649, -0.1369168356997971, 0.0740365111561865, 0.0801217038539553])
        gt_err = np.array([0.2251521298174442, 0.2718052738336714, 0.1176470588235294, 0.1440162271805274, 0.2393509127789046, 0.2900608519269776])
        curves_in_subplot = {
            "1": [
                { 
                    "pred": Curve({"y": pred_value, "err": pred_err}), "gt": Curve({"y": gt_value, "err": gt_err}), 
                    "gt_len": 10, "pred_len": 6,
                    "y scale": np.max(gt_value) - np.min(gt_value), "y mean": np.mean(gt_value),
                    "err scale": np.max(gt_err) - np.min(gt_err), "err mean": np.mean(gt_err)
                }
            ]
        }
        ans = {
            "Value performance": cal_metrics(pred_value, gt_value, np.max(gt_value) - np.min(gt_value), np.mean(gt_value)),
            "Error performance": cal_metrics(pred_err, gt_err, np.max(gt_err) - np.min(gt_err), np.mean(gt_err)),
            # "Overall performance": cal_metrics(
            #     np.concatenate([pred_value.reshape(-1, 1), pred_err.reshape(-1, 1)], axis=1), 
            #     np.concatenate([gt_value.reshape(-1, 1), gt_err.reshape(-1, 1)], axis=1)),
            "DP accuracy": 0.6,
            "DP recall": 1.0
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_multi_curve(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]
        x_scales = [np.max(gt_x[0]) - np.min(gt_x[0]), np.max(gt_x[1]) - np.min(gt_x[1])]
        x_means = [np.mean(gt_x[0]), np.mean(gt_x[1])]
        y_scales = [np.max(gt_value[0]) - np.min(gt_value[0]), np.max(gt_value[1]) - np.min(gt_value[1])]
        y_means = [np.mean(gt_value[0]), np.mean(gt_value[1])]

        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x[0], "y": gt_value[0]}), 
                    "pred": Curve({"x": pred_x[0], "y": pred_value[0]}), 
                    "gt_len": 5, "pred_len": 5,
                    "x scale": x_scales[0], "x mean": x_means[0],
                    "y scale": y_scales[0], "y mean": y_means[0],
                },
                { 
                    "gt": Curve({"x": gt_x[1], "y": gt_value[1]}), 
                    "pred": Curve({"x": pred_x[1], "y": pred_value[1]}), 
                    "gt_len": 10, "pred_len": 8,
                    "x scale": x_scales[1], "x mean": x_means[1],
                    "y scale": y_scales[1], "y mean": y_means[1],
                }
            ]
        }
        first_curve = {
            "X performance": cal_metrics(np.array(pred_x[0]), np.array(gt_x[0]), x_scales[0], x_means[0]),
            "Value performance": cal_metrics(np.array(pred_value[0]), np.array(gt_value[0]), y_scales[0], y_means[0]),
            # "Overall performance": cal_metrics(np.array(pred_x[0] + pred_value[0]), np.array(gt_x[0] + gt_value[0])),
            "DP accuracy": 1.0,
            "DP recall": 1.0
        }
        second_curve = {
            "X performance": cal_metrics(np.array(pred_x[1]), np.array(gt_x[1]), x_scales[1], x_means[1]),
            "Value performance": cal_metrics(np.array(pred_value[1]), np.array(gt_value[1]), y_scales[1], y_means[1]),
            # "Overall performance": cal_metrics(np.array(pred_x[1] + pred_value[1]), np.array(gt_x[1] + gt_value[1])),
            "DP accuracy": 0.5,
            "DP recall": 0.625
        }
        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in first_curve["X performance"].keys():
            merged_x[key] = (first_curve["X performance"][key] + second_curve["X performance"][key]) / 2
            merged_v[key] = (first_curve["Value performance"][key] + second_curve["Value performance"][key]) / 2
            # merged_overall[key] = (first_curve["Overall performance"][key] + second_curve["Overall performance"][key]) / 2
            # 0.6956066789457732 0.8339262120382456
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            # "Overall performance": merged_overall,
            "DP accuracy": (first_curve["DP accuracy"] + second_curve["DP accuracy"]) / 2,
            "DP recall": (first_curve["DP recall"] + second_curve["DP recall"]) / 2
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_multi_subplot(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]
        x_scale = [np.max(gt_x[0]) - np.min(gt_x[0]), np.max(gt_x[1]) - np.min(gt_x[1])]
        x_mean = [np.mean(gt_x[0]), np.mean(gt_x[1])]
        y_scale = [np.max(gt_value[0]) - np.min(gt_value[0]), np.max(gt_value[1]) - np.min(gt_value[1])]
        y_mean = [np.mean(gt_value[0]), np.mean(gt_value[1])]

        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x[0], "y": gt_value[0]}), "pred": Curve({"x": pred_x[0], "y": pred_value[0]}), 
                    "gt_len": 5, "pred_len": 5, 
                    "x scale": x_scale[0], "x mean": x_mean[0],
                    "y scale": y_scale[0], "y mean": y_mean[0],
                },
            ],
            "2": [
                {
                    "gt": Curve({"x": gt_x[1], "y": gt_value[1]}), "pred": Curve({"x": pred_x[1], "y": pred_value[1]}), 
                    "gt_len": 10, "pred_len": 8, 
                    "x scale": x_scale[1], "x mean": x_mean[1],
                    "y scale": y_scale[1], "y mean": y_mean[1],
                }
            ]
        }
        first_curve = {
            "X performance": cal_metrics(np.array(pred_x[0]), np.array(gt_x[0]), x_scale[0], x_mean[0]),
            "Value performance": cal_metrics(np.array(pred_value[0]), np.array(gt_value[0]), y_scale[0], y_mean[0]),
            # "Overall performance": cal_metrics(np.array(pred_x[0] + pred_value[0]), np.array(gt_x[0] + gt_value[0])), 
            "DP accuracy": 1.0,
            "DP recall": 1.0
        }
        second_curve = {
            "X performance": cal_metrics(np.array(pred_x[1]), np.array(gt_x[1]), x_scale[1], x_mean[1]),
            "Value performance": cal_metrics(np.array(pred_value[1]), np.array(gt_value[1]), y_scale[1], y_mean[1]),
            # "Overall performance": cal_metrics(np.array(pred_x[1] + pred_value[1]), np.array(gt_x[1] + gt_value[1])),
            "DP accuracy": 0.5,
            "DP recall": 0.625
        }
        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in first_curve["X performance"].keys():
            merged_x[key] = (first_curve["X performance"][key] + second_curve["X performance"][key]) / 2
            merged_v[key] = (first_curve["Value performance"][key] + second_curve["Value performance"][key]) / 2
            # merged_overall[key] = (first_curve["Overall performance"][key] + second_curve["Overall performance"][key]) / 2
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            # "Overall performance": merged_overall,
            "DP accuracy": (first_curve["DP accuracy"] + second_curve["DP accuracy"]) / 2,
            "DP recall": (first_curve["DP recall"] + second_curve["DP recall"]) / 2
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_multi_subplot_multi_curve(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15], [-0.15, -0.1, -0.05, 0.0]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327], [-0.1957403651115618, -0.3600405679513184, -0.061866125760649, -0.1369168356997971]]
        pred_err = [0.1, 0.08, 0.12, 0.07]
        gt_err = [0.2251521298174442, 0.2718052738336714, 0.1176470588235294, 0.1440162271805274]
        x_scales = [np.max(gt_x[0]) - np.min(gt_x[0]), np.max(gt_x[1]) - np.min(gt_x[1])]
        x_means = [np.mean(gt_x[0]), np.mean(gt_x[1])]
        y_scales = [np.max(gt_value[0]) - np.min(gt_value[0]), np.max(gt_value[1]) - np.min(gt_value[1]), np.max(gt_value[2]) - np.min(gt_value[2])]
        y_means = [np.mean(gt_value[0]), np.mean(gt_value[1]), np.mean(gt_value[2])]
        err_scale = np.max(gt_err) - np.min(gt_err)
        err_mean = np.mean(gt_err)

        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x[0], "y": gt_value[0]}), 
                    "pred": Curve({"x": pred_x[0], "y": pred_value[0]}), 
                    "gt_len": 5, "pred_len": 5,
                    "x scale": x_scales[0], "x mean": x_means[0],
                    "y scale": y_scales[0], "y mean": y_means[0],
                },
                {
                    "gt": Curve({"x": gt_x[1], "y": gt_value[1]}), 
                    "pred": Curve({"x": pred_x[1], "y": pred_value[1]}), 
                    "gt_len": 10, "pred_len": 8,
                    "x scale": x_scales[1], "x mean": x_means[1],
                    "y scale": y_scales[1], "y mean": y_means[1],
                }
            ], 
            "2": [
                { 
                    "gt": Curve({"y": gt_value[2], "err": gt_err}), 
                    "pred": Curve({"y": pred_value[2], "err": pred_err}), 
                    "gt_len": 16, "pred_len": 20,
                    "y scale": np.max(gt_value[2]) - np.min(gt_value[2]), "y mean": np.mean(gt_value[2]),
                    "err scale": err_scale, "err mean": err_mean
                }
            ]
        }

        curves = [{}, {}, {}]
        curves[0]["X performance"] = cal_metrics(np.array(pred_x[0]), np.array(gt_x[0]), x_scales[0], x_means[0])
        curves[0]["Value performance"] = cal_metrics(np.array(pred_value[0]), np.array(gt_value[0]), y_scales[0], y_means[0])
        # curves[0]["Overall performance"] = cal_metrics(np.array(pred_x[0] + pred_value[0]), np.array(gt_x[0] + gt_value[0]))
        curves[0]["DP accuracy"] = 1.0
        curves[0]["DP recall"] = 1.0

        curves[1]["X performance"] = cal_metrics(np.array(pred_x[1]), np.array(gt_x[1]), x_scales[1], x_means[1])
        curves[1]["Value performance"] = cal_metrics(np.array(pred_value[1]), np.array(gt_value[1]), y_scales[1], y_means[1])
        # curves[1]["Overall performance"] = cal_metrics(np.array(pred_x[1] + pred_value[1]), np.array(gt_x[1] + gt_value[1]))
        curves[1]["DP accuracy"] = 0.5
        curves[1]["DP recall"] = 0.625

        curves[2]["Value performance"] = cal_metrics(np.array(pred_value[2]), np.array(gt_value[2]), y_scales[2], y_means[2])
        curves[2]["Error performance"] = cal_metrics(np.array(pred_err), np.array(gt_err), err_scale, err_mean)
        # curves[2]["Overall performance"] = cal_metrics(np.array(pred_value[2] + pred_err), np.array(gt_value[2] + gt_err))
        curves[2]["DP accuracy"] = 4/16
        curves[2]["DP recall"] = 4/20

        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in curves[0]["X performance"].keys():
            merged_x[key] = (curves[0]["X performance"][key] + curves[1]["X performance"][key]) / 2
            merged_v[key] = ((curves[0]["Value performance"][key] + curves[1]["Value performance"][key])/2 + curves[2]["Value performance"][key]) / 2
            # merged_overall[key] = ((curves[0]["Overall performance"][key] + curves[1]["Overall performance"][key])/2 + curves[2]["Overall performance"][key]) / 2
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            # "Overall performance": merged_overall,
            "Error performance": curves[2]["Error performance"],
            "DP accuracy": ((curves[0]["DP accuracy"] + curves[1]["DP accuracy"])/2 + curves[2]["DP accuracy"]) / 2,
            "DP recall": ((curves[0]["DP recall"] + curves[1]["DP recall"])/2 + curves[2]["DP recall"]) / 2
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_missing_or_extra_curve(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]
        x_scales = [np.max(gt_x[0]) - np.min(gt_x[0]), np.max(gt_x[1]) - np.min(gt_x[1])]
        x_means = [np.mean(gt_x[0]), np.mean(gt_x[1])]
        y_scales = [np.max(gt_value[0]) - np.min(gt_value[0]), np.max(gt_value[1]) - np.min(gt_value[1])]
        y_means = [np.mean(gt_value[0]), np.mean(gt_value[1])]

        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x[0], "y": gt_value[0]}),
                    "pred": Curve({"x": pred_x[0], "y": pred_value[0]}),
                    "gt_len": 5, "pred_len": 5,
                    "x scale": x_scales[0], "x mean": x_means[0],
                    "y scale": y_scales[0], "y mean": y_means[0],
                },
                {
                    "gt": Curve({"x": gt_x[1], "y": gt_value[1]}),
                    "pred": Curve({"x": pred_x[1], "y": pred_value[1]}),
                    "gt_len": 5, "pred_len": 5,
                    "x scale": x_scales[1], "x mean": x_means[1],
                    "y scale": y_scales[1], "y mean": y_means[1],
                },
                { 
                    "gt": Curve({ "x": gt_x[1], "y": gt_value[1] }), 
                    "pred": Curve({}), "gt_len": 5, "pred_len": 0 
                }
            ]
        }
        curves = [
            {
                "X performance": cal_metrics(np.array(pred_x[0]), np.array(gt_x[0]), x_scales[0], x_means[0]),
                "Value performance": cal_metrics(np.array(pred_value[0]), np.array(gt_value[0]), y_scales[0], y_means[0]),
                # "Overall performance": cal_metrics(np.array(pred_x[0] + pred_value[0]), np.array(gt_x[0] + gt_value[0])),
                "DP accuracy": 1.0,
                "DP recall": 1.0
            },
            {
                "X performance": cal_metrics(np.array(pred_x[1]), np.array(gt_x[1]), x_scales[1], x_means[1]),
                "Value performance": cal_metrics(np.array(pred_value[1]), np.array(gt_value[1]), y_scales[1], y_means[1]),
                # "Overall performance": cal_metrics(np.array(pred_x[1] + pred_value[1]), np.array(gt_x[1] + gt_value[1])),
                "DP accuracy": 1.0,
                "DP recall": 1.0
            },
            {
                "DP accuracy": 0.0,
                "DP recall": 0.0
            }
        ]
        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in curves[0]["X performance"].keys():
            merged_x[key] = (curves[0]["X performance"][key] + curves[1]["X performance"][key]) / 2
            merged_v[key] = (curves[0]["Value performance"][key] + curves[1]["Value performance"][key]) / 2
            # merged_overall[key] = (curves[0]["Overall performance"][key] + curves[1]["Overall performance"][key]) / 2
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            # "Overall performance": merged_overall,
            "DP accuracy": (curves[0]["DP accuracy"] + curves[1]["DP accuracy"]) / 3,
            "DP recall": (curves[0]["DP recall"] + curves[1]["DP recall"]) / 3
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

        curves_in_subplot["1"][-1] = {
            "gt": Curve({}), "pred": Curve({ "x": pred_x[1], "y": pred_value[1] }), 
            "gt_len": 0, "pred_len": 5
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_missing_or_extra_subplot(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]
        x_scales = [np.max(gt_x[0]) - np.min(gt_x[0]), np.max(gt_x[1]) - np.min(gt_x[1])]
        x_means = [np.mean(gt_x[0]), np.mean(gt_x[1])]
        y_scales = [np.max(gt_value[0]) - np.min(gt_value[0]), np.max(gt_value[1]) - np.min(gt_value[1])]
        y_means = [np.mean(gt_value[0]), np.mean(gt_value[1])]

        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x[0], "y": gt_value[0]}),
                    "pred": Curve({"x": pred_x[0], "y": pred_value[0]}),
                    "gt_len": 5, "pred_len": 5,
                    "x scale": x_scales[0], "x mean": x_means[0],
                    "y scale": y_scales[0], "y mean": y_means[0],
                },
                {
                    "gt": Curve({"x": gt_x[1], "y": gt_value[1]}),
                    "pred": Curve({"x": pred_x[1], "y": pred_value[1]}),
                    "gt_len": 5, "pred_len": 5,
                    "x scale": x_scales[1], "x mean": x_means[1],
                    "y scale": y_scales[1], "y mean": y_means[1],
                }
            ],
            "2": [{ 
                "gt": Curve({ "x": gt_x[1], "y": gt_value[1] }),
                "pred": Curve({}), "gt_len": 5, "pred_len": 0 
            }]
        }
        curves = [
            {
                "X performance": cal_metrics(np.array(pred_x[0]), np.array(gt_x[0]), x_scales[0], x_means[0]),
                "Value performance": cal_metrics(np.array(pred_value[0]), np.array(gt_value[0]), y_scales[0], y_means[0]),
                # "Overall performance": cal_metrics(np.array(pred_x[0] + pred_value[0]), np.array(gt_x[0] + gt_value[0])),
                "DP accuracy": 1.0,
                "DP recall": 1.0
            },
            {
                "X performance": cal_metrics(np.array(pred_x[1]), np.array(gt_x[1]), x_scales[1], x_means[1]),
                "Value performance": cal_metrics(np.array(pred_value[1]), np.array(gt_value[1]), y_scales[1], y_means[1]),
                # "Overall performance": cal_metrics(np.array(pred_x[1] + pred_value[1]), np.array(gt_x[1] + gt_value[1])),
                "DP accuracy": 1.0,
                "DP recall": 1.0
            },
            {
                "DP accuracy": 0.0,
                "DP recall": 0.0
            }
        ]
        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in curves[0]["X performance"].keys():
            merged_x[key] = (curves[0]["X performance"][key] + curves[1]["X performance"][key]) / 2
            merged_v[key] = (curves[0]["Value performance"][key] + curves[1]["Value performance"][key]) / 2
            # merged_overall[key] = (curves[0]["Overall performance"][key] + curves[1]["Overall performance"][key]) / 2
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            # "Overall performance": merged_overall,
            "DP accuracy": (curves[0]["DP accuracy"] + curves[1]["DP accuracy"]) / 2 / 2,
            "DP recall": (curves[0]["DP recall"] + curves[1]["DP recall"]) / 2 / 2
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

        curves_in_subplot["2"][0] = {
            "gt": Curve({}), "pred": Curve({ "x": pred_x[1], "y": pred_value[1] }), 
            "gt_len": 0, "pred_len": 5
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

class TestPlotEvaluatorPairing(TestPlotEvaluator):
    def test_pair_score(self):
        names = {
            "str": ["A", "B", "C"],
            "range": ["1 - 2", "1 - 3", "2 - 3", "2 - 4", "3 - 4"],
            "float": [3.14-20, 3.14-10, 3.14, 3.14+10, "-16.86", "-6.86", "3.14", "13.14"],
            "int": [-2, -1, 0, 1, 2, "-2", "-1", "0", "1", "2"],
            "NoneType": [None, np.nan, ""]
        }
        for type_pred in names:
            for type_gt in names:
                for i, name_pred in enumerate(names[type_pred]):
                    for j, name_gt in enumerate(names[type_gt]):
                        data_score = pair_score(process(name_pred), process(name_gt))
                        # curve_score = pair_score(name_pred, name_gt)
                        if type_gt != type_pred and type_gt in ["float", "int"] and type_pred in ["float", "int"]:
                            continue
                        if type_gt != type_pred:
                            self.assertEqual(data_score, np.inf)
                            # self.assertEqual(curve_score, np.inf)
                        elif type_gt == "range":
                            self.assertEqual(data_score, abs(i - j))
                            # self.assertEqual(curve_score, 0 if i == j else np.inf)
                        elif type_gt == "str":
                            self.assertEqual(data_score, 0 if i == j else np.inf)
                            # self.assertEqual(curve_score, 0 if i == j else np.inf)
                        elif type_gt == "float":
                            self.assertAlmostEqual(data_score, 0 if i % 4 == j % 4 else abs(i%4 - j%4)*10)
                            # self.assertAlmostEqual(curve_score, abs(i%4 - j%4)*10 if ((i < 4 and j < 4) or i == j) else np.inf)
                        elif type_gt == "int":
                            self.assertEqual(data_score, 0 if i % 5 == j % 5 else abs(i%5 - j%5))
                            # self.assertEqual(curve_score, abs(i%5 - j%5) if ((i < 5 and j < 5) or i == j) else np.inf)
                        elif type_gt == "NoneType":
                            self.assertEqual(data_score, 0)

    def gen_curves(self, curve_num):
        curves = {}
        for i, num in enumerate(curve_num):
            curves[chr(65+i)] = {}
            for j in range(num):
                curves[chr(65+i)][str(j+1)] = {"y": list(range(i*10+j+1))}
        return curves
    
    def gen_ans(self, curve_num):
        ans = {}
        for i, num in enumerate(curve_num):
            ans[chr(65+i)] = []
            for j in range(num):
                ans[chr(65+i)].append((Curve({"y": list(range(i*10+j+1))}), Curve({"y": list(range(i*10+j+1))})))
        return ans
    
    def assertSameCurvePair(self, pair1, pair2):
        self.assertEqual(sorted(list(pair1.keys())), sorted(list(pair2.keys())))
        for key in pair1.keys():
            self.assertEqual(len(pair1[key]), len(pair2[key]))
            for i in range(len(pair1[key])):
                self.assertEqual(len(pair1[key][i]), len(pair2[key][i]))
    
    def test_pair_curves(self):
        curve_num = [2, 1, 3]
        pred_curves = self.gen_curves(curve_num)
        gt_curves = self.gen_curves(curve_num)
        paired_curves = pair_curves(pred_curves, gt_curves)
        ans = self.gen_ans(curve_num)
        self.assertSameCurvePair(paired_curves, ans)

    def test_pair_missing_curve(self):
        curve_num = [2, 1, 3]
        pred_curves = self.gen_curves(curve_num)
        gt_curves = self.gen_curves(curve_num)
        del pred_curves["C"]["2"]
        paired_curves = pair_curves(pred_curves, gt_curves)

        ans = self.gen_ans(curve_num)
        ans["C"][1] = (Curve({}), Curve({"y": list(range(20, 23))}))
        self.assertSameCurvePair(paired_curves, ans)

    def test_pair_extra_curve(self):
        pred_curves = self.gen_curves([2, 2, 3])
        gt_curves = self.gen_curves([2, 1, 3])
        paired_curves = pair_curves(pred_curves, gt_curves)

        ans = self.gen_ans([2, 2, 3])
        ans["B"][1] = (ans["B"][1][0], Curve({}))
        self.assertSameCurvePair(paired_curves, ans)
    
    def test_pair_missing_subplot(self):
        pred_curves = self.gen_curves([2, 1])
        gt_curves = self.gen_curves([2, 1, 3])
        paired_curves = pair_curves(pred_curves, gt_curves)

        ans = self.gen_ans([2, 1, 3])
        new_c = []
        for _, gt_c in ans["C"]:
            new_c.append((Curve({}), gt_c))
        ans["C"] = new_c
        self.assertSameCurvePair(paired_curves, ans)

    def test_pair_extra_subplot(self):
        pred_curves = self.gen_curves([2, 1, 3])
        gt_curves = self.gen_curves([2, 1])
        paired_curves = pair_curves(pred_curves, gt_curves)

        ans = self.gen_ans([2, 1, 3])
        new_c = []
        for pred_c, _ in ans["C"]:
            new_c.append((pred_c, Curve({})))
        ans["C"] = new_c
        self.assertSameCurvePair(paired_curves, ans)

    def DISABLE_test_pair_data_points(self):
        pass
    
    def DISABLE_test_cross_pairing(self):
        # histogram + multiple panels + no type-2
        pred = pd.read_csv("Tests/evaluator/P-14-O2_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-14-O2_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_x = [(1995, 1996), (1997, 1998), (1999, 2000), (2001, 2002), (2003, 2004), (2005, 2006), (2007, 2008), (2009, 2010), (2011, 2012), (2013, 2014)]
        pred_value = [0.05, 0.03, 0.15, 0.08, 0.22, 0.07, 0.13, 0.06, 0.04, 0.02]
        gt_x = [(1996.0, 1997.0), (1998.0, 1999.0), (1999.0, 2000.0), (2000.0, 2001.0), (2001.0, 2002.), (2003.0, 2004.0), (2004.0, 2005.0), (2007.0, 2008.0), (2008.0, 2009.0), (2009.0, 2010.0), (2012.0, 2013.0)]
        gt_value = [0.0447214076246334, 0.0271260997067449, 0.1590909090909091, 0.05058651026392966, 0.0843108504398827, 0.2155425219941349, 0.0659824046920821, 0.1422287390029326, 0.123900293255132, 0.0645161290322581, 0.0344574780058651]

        pre_curve = Curve({"x": pred_x, "y": pred_value})
        gt_curve = Curve({"x": gt_x, "y": gt_value})
        pred_pair, gt_pair = pair_data_points(pre_curve, gt_curve)

class TestPlotEvaluatorGeneral(TestPlotEvaluator):
    def test_discrete_err(self):
        # dot plot + error bars + multiple curves 
        pred = pd.read_csv("Tests/evaluator/P-2-O1_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-2-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_value = [[-0.15, -0.05, 0.1], [-0.1, 0.0, 0.05]]
        pred_err = [[0.1, 0.12, 0.09], [0.08, 0.07, 0.06]]
        gt_value = [[-0.1957403651115618, -0.061866125760649, 0.0740365111561865], 
                    [-0.3600405679513184, -0.1369168356997971, 0.0801217038539553]]
        gt_err = [[0.2251521298174442, 0.1176470588235294, 0.2393509127789046], 
                  [0.2718052738336714, 0.1440162271805274, 0.2900608519269776]]
        y_scale = [np.max(gt_value[0]) - np.min(gt_value[0]), np.max(gt_value[1]) - np.min(gt_value[1])]
        y_mean = [np.mean(gt_value[0]), np.mean(gt_value[1])]
        err_scale = [np.max(gt_err[0]) - np.min(gt_err[0]), np.max(gt_err[1]) - np.min(gt_err[1])]
        err_mean = [np.mean(gt_err[0]), np.mean(gt_err[1])]

        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"y": gt_value[0], "err": gt_err[0]}),
                    "pred": Curve({"y": pred_value[0], "err": pred_err[0]}),
                    "gt_len": 3, "pred_len": 3,
                    "y scale": y_scale[0], "y mean": y_mean[0],
                    "err scale": err_scale[0], "err mean": err_mean[0]
                },
                {
                    "gt": Curve({"y": gt_value[1], "err": gt_err[1]}),
                    "pred": Curve({"y": pred_value[1], "err": pred_err[1]}),
                    "gt_len": 3, "pred_len": 3, 
                    "y scale": y_scale[1], "y mean": y_mean[1],
                    "err scale": err_scale[1], "err mean": err_mean[1]
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)
    
    def test_hist_range(self):
        # histogram with Type-1 being range + no subplot value + no type-2
        pred = pd.read_csv("Tests/evaluator/P-10-O1_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-10-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])
        
        pred_x = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        pred_value = [75, 75, 85, 95, 80, 20, 5, 0, 5, 5]
        gt_x = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        gt_value = [65.97087378640776, 64.95145631067962, 79.80582524271844, 88.10679611650485, 70.04854368932038, 21.99029126213592, 5.825242718446603, 1.893203883495147, 1.0194174757281471, 3.932038834951456]
        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x, "y": gt_value}),
                    "pred": Curve({"x": pred_x, "y": pred_value}),
                    "gt_len": 10, "pred_len": 10, 
                    "x scale": 1.0, "x mean": np.mean(np.array(gt_x).flatten()),
                    "y scale": np.max(gt_value) - np.min(gt_value), "y mean": np.mean(gt_value)
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)

    def test_cont_simple(self):
        # continuous plot + multiple curves + no subplot valuez
        pred = pd.read_csv("Tests/evaluator/P-20-O1_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-20-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])
        
        pred_x, pred_y = [[], []], [[], []]
        for i in range(len(pred)):
            if pred["Type-2"][i] == "Treatment":
                pred_x[0].append(pred["Type-1"][i])
                pred_y[0].append(pred["Value"][i])
            else:
                pred_x[1].append(pred["Type-1"][i])
                pred_y[1].append(pred["Value"][i])
        gt_x, gt_y = [[], []], [[], []]
        for i in range(len(gt)):
            if gt["Type-2"][i] == "Treatment":
                gt_x[0].append(gt["Type-1"][i])
                gt_y[0].append(gt["Value"][i])
            else:
                gt_x[1].append(gt["Type-1"][i])
                gt_y[1].append(gt["Value"][i])
        pred_curves = [Curve({"x": pred_x[0], "y": pred_y[0]}), Curve({"x": pred_x[1], "y": pred_y[1]})]
        gt_curves = [Curve({"x": gt_x[0], "y": gt_y[0]}), Curve({"x": gt_x[1], "y": gt_y[1]})]
        x_common = [np.linspace(0, 259.99999999999994, num=50), np.linspace(0, 257.49999999999994, num=50)]
        pred_curves[0].interpolate(x_common[0])
        pred_curves[1].interpolate(x_common[1])
        gt_curves[0].interpolate(x_common[0])
        gt_curves[1].interpolate(x_common[1])
        curves_in_subplot = {
            "1": [
                {
                    "gt": gt_curves[0], "pred": pred_curves[0],
                    "y scale": np.max(gt_curves[0].y) - np.min(gt_curves[0].y), "y mean": np.mean(gt_curves[0].y)
                },
                {
                    "gt": gt_curves[1], "pred": pred_curves[1],
                    "y scale": np.max(gt_curves[1].y) - np.min(gt_curves[1].y), "y mean": np.mean(gt_curves[1].y)
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)

    def test_gt_more_than_pred(self):
        # ground truth has more data then prediction 
        pred = pd.read_csv("Tests/evaluator/P-14-O1_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-14-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])
        
        x_scales = [np.max(gt["Type-1"][:18]) - np.min(gt["Type-1"][:18]), np.max(gt["Type-1"][18:]) - np.min(gt["Type-1"][18:])]
        x_means = [np.mean(gt["Type-1"][:18]), np.mean(gt["Type-1"][18:])]
        y_scales = [np.max(gt["Value"][:18]) - np.min(gt["Value"][:18]), np.max(gt["Value"][18:]) - np.min(gt["Value"][18:])]
        y_means = [np.mean(gt["Value"][:18]), np.mean(gt["Value"][18:])]

        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]

        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x[0], "y": gt_value[0]}),
                    "pred": Curve({"x": pred_x[0], "y": pred_value[0]}),
                    "gt_len": 18, "pred_len": 5,
                    "x scale": x_scales[0], "x mean": x_means[0],
                    "y scale": y_scales[0], "y mean": y_means[0],
                },
                {
                    "gt": Curve({"x": gt_x[1], "y": gt_value[1]}),
                    "pred": Curve({"x": pred_x[1], "y": pred_value[1]}),
                    "gt_len": 18, "pred_len": 5,
                    "x scale": x_scales[1], "x mean": x_means[1],
                    "y scale": y_scales[1], "y mean": y_means[1],
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)

    def test_pred_more_than_gt(self):
        # prediction has more data then ground truth + Type-1 need estimation
        pred = pd.read_csv("Tests/evaluator/P-47-O3_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-47-O3_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_x = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        pred_value = [1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
        gt_x = [27.649030247272865, 32.681230168415915, 37.71117703385217, 42.74225042714181, 47.63532415838982, 52.66301796811927, 57.68451587465501, 62.5668875912957, 67.57655695537073, 72.58115694410546, 77.58125082142658, 82.58190796267439]
        gt_value = [0.9199475065616798, 0.9803149606299212, 1.0459317585301835, 1.111548556430446, 1.158792650918635, 1.1850393700787405, 1.1797900262467191, 1.1351706036745406, 1.0433070866141732, 0.9094488188976378, 0.7493438320209974, 0.5761154855643045]
        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"x": gt_x, "y": gt_value}),
                    "pred": Curve({"x": pred_x, "y": pred_value}),
                    "gt_len": 12, "pred_len": 13, 
                    "x scale": np.max(gt_x) - np.min(gt_x), "x mean": np.mean(gt_x),
                    "y scale": np.max(gt_value) - np.min(gt_value), "y mean": np.mean(gt_value)
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)
    
    def test_multi_subplot(self):
        # Type-1 and subplot value are float 
        pred = pd.read_csv("Tests/evaluator/P-49-O6_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-49-O6_gt.csv")
        perf = evaluate_plot([pred], [gt])
        
        pred_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        pred_v = [
            [3.9, 3.6, 3.4, 3.1, 2.9, 2.6, 2.4, 2.1, 1.9, 1.7], 
            [3.9, 3.7, 3.4, 3.2, 2.8, 2.5, 2.3, 2.0, 1.8, 1.6], 
            [3.9, 3.7, 3.5, 3.2, 2.9, 2.6, 2.3, 2.0, 1.8, 1.6]
        ]
        gt_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        gt_v = [
            [3.8055555555555554, 3.321428571428571, 3.182539682539683, 3.003968253968254, 2.7182539682539684, 2.515873015873016, 2.369047619047619, 2.174603174603175, 2.031746031746032, 1.6746031746031744], 
            [3.801587301587301, 3.174603174603175, 3.003968253968254, 2.984126984126984, 2.6468253968253967, 2.4404761904761907, 2.4047619047619047, 2.087301587301587, 1.9603174603174605, 1.6825396825396823], 
            [3.78968253968254, 3.134920634920635, 2.9444444444444446, 2.880952380952381, 2.5396825396825395, 2.384920634920635, 2.325396825396825, 2.0, 1.8968253968253967, 1.6111111111111112]
        ]
        curves_in_subplot = {
            "2012":[{ 
                "gt": Curve({"x": gt_x[0], "y": gt_v[0]}),
                "pred": Curve({"x": pred_x[0], "y": pred_v[0]}),
                "gt_len": 10, "pred_len": 10 ,
                "x scale": np.max(gt_x[0]) - np.min(gt_x[0]), "x mean": np.mean(gt_x[0]),
                "y scale": np.max(gt_v[0]) - np.min(gt_v[0]), "y mean": np.mean(gt_v[0])
            }],
            "2013":[{
                "gt": Curve({"x": gt_x[1], "y": gt_v[1]}),
                "pred": Curve({"x": pred_x[1], "y": pred_v[1]}),
                "gt_len": 10, "pred_len": 10,
                "x scale": np.max(gt_x[1]) - np.min(gt_x[1]), "x mean": np.mean(gt_x[1]),
                "y scale": np.max(gt_v[1]) - np.min(gt_v[1]), "y mean": np.mean(gt_v[1])
            }],
            "2014":[{
                "gt": Curve({"x": gt_x[2], "y": gt_v[2]}),
                "pred": Curve({"x": pred_x[2], "y": pred_v[2]}),
                "gt_len": 10, "pred_len": 10, 
                "x scale": np.max(gt_x[2]) - np.min(gt_x[2]), "x mean": np.mean(gt_x[2]),
                "y scale": np.max(gt_v[2]) - np.min(gt_v[2]), "y mean": np.mean(gt_v[2])
            }]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)

    def test_missing_curve(self):
        pred = pd.read_csv("Tests/evaluator/P-2-O1_pred_missing_curve.csv")
        gt = pd.read_csv("Tests/evaluator/P-2-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_value = [[-0.15, -0.05, 0.1]]
        pred_err = [[0.1, 0.12, 0.09]]
        gt_value = [[-0.1957403651115618, -0.061866125760649, 0.0740365111561865], 
                    [-0.3600405679513184, -0.1369168356997971, 0.0801217038539553]]
        gt_err = [[0.2251521298174442, 0.1176470588235294, 0.2393509127789046], 
                  [0.2718052738336714, 0.1440162271805274, 0.2900608519269776]]
        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"y": gt_value[0], "err": gt_err[0]}),
                    "pred": Curve({"y": pred_value[0], "err": pred_err[0]}),
                    "gt_len": 3, "pred_len": 3, 
                    "y scale": np.max(gt_value[0]) - np.min(gt_value[0]), "y mean": np.mean(gt_value[0]),
                    "err scale": np.max(gt_err[0]) - np.min(gt_err[0]), "err mean": np.mean(gt_err[0])
                },
                {
                    "gt": Curve({ "y": gt_value[1], "err": gt_err[1]}), "pred": Curve({}), 
                    "gt_len": 3, "pred_len": 0 
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 0.5
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)

    def test_extra_curve(self):
        pred = pd.read_csv("Tests/evaluator/P-2-O1_pred_extra_curve.csv")
        gt = pd.read_csv("Tests/evaluator/P-2-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_value = [[-0.15, -0.05, 0.1], [-0.1, 0.0, 0.05]]
        pred_err = [[0.1, 0.12, 0.09], [0.08, 0.07, 0.06]]
        gt_value = [[-0.1957403651115618, -0.061866125760649, 0.0740365111561865], 
                    [-0.3600405679513184, -0.1369168356997971, 0.0801217038539553]]
        gt_err = [[0.2251521298174442, 0.1176470588235294, 0.2393509127789046], 
                  [0.2718052738336714, 0.1440162271805274, 0.2900608519269776]]
        y_scale = [np.max(gt_value[0]) - np.min(gt_value[0]), np.max(gt_value[1]) - np.min(gt_value[1])]
        y_mean = [np.mean(gt_value[0]), np.mean(gt_value[1])]
        err_scale = [np.max(gt_err[0]) - np.min(gt_err[0]), np.max(gt_err[1]) - np.min(gt_err[1])]
        err_mean = [np.mean(gt_err[0]), np.mean(gt_err[1])]
        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"y": gt_value[0], "err": gt_err[0]}),
                    "pred": Curve({"y": pred_value[0], "err": pred_err[0]}),
                    "gt_len": 3, "pred_len": 3,
                    "y scale": y_scale[0], "y mean": y_mean[0],
                    "err scale": err_scale[0], "err mean": err_mean[0]
                },
                {
                    "gt": Curve({"y": gt_value[1], "err": gt_err[1]}),
                    "pred": Curve({"y": pred_value[1], "err": pred_err[1]}),
                    "gt_len": 3, "pred_len": 3,
                    "y scale": y_scale[1], "y mean": y_mean[1],
                    "err scale": err_scale[1], "err mean": err_mean[1]
                },
                {
                    "gt": Curve({}), "pred": Curve({"y": pred_value[1], "err": pred_err[1]}), 
                    "gt_len": 0, "pred_len": 3 
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 2/3
        self.assertSameDict(perf, ans)

    def test_missing_subplot(self):
        pred = pd.read_csv("Tests/evaluator/P-49-O6_pred_missing_subplot.csv")
        gt = pd.read_csv("Tests/evaluator/P-49-O6_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        pred_v = [
            [3.9, 3.6, 3.4, 3.1, 2.9, 2.6, 2.4, 2.1, 1.9, 1.7], 
            [3.9, 3.7, 3.4, 3.2, 2.8, 2.5, 2.3, 2.0, 1.8, 1.6], 
            [3.9, 3.7, 3.5, 3.2, 2.9, 2.6, 2.3, 2.0, 1.8, 1.6]
        ]
        gt_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        gt_v = [
            [3.8055555555555554, 3.321428571428571, 3.182539682539683, 3.003968253968254, 2.7182539682539684, 2.515873015873016, 2.369047619047619, 2.174603174603175, 2.031746031746032, 1.6746031746031744], 
            [3.801587301587301, 3.174603174603175, 3.003968253968254, 2.984126984126984, 2.6468253968253967, 2.4404761904761907, 2.4047619047619047, 2.087301587301587, 1.9603174603174605, 1.6825396825396823]
        ]
        curves_in_subplot = {
            "2012":[{ 
                "gt": Curve({"x": gt_x[0], "y": gt_v[0]}),
                "pred": Curve({"x": pred_x[0], "y": pred_v[0]}),
                "gt_len": 10, "pred_len": 10, 
                "x scale": np.max(gt_x[0]) - np.min(gt_x[0]), "x mean": np.mean(gt_x[0]),
                "y scale": np.max(gt_v[0]) - np.min(gt_v[0]), "y mean": np.mean(gt_v[0])
            }],
            "2013":[{
                "gt": Curve({"x": gt_x[1], "y": gt_v[1]}),
                "pred": Curve({"x": pred_x[1], "y": pred_v[1]}),
                "gt_len": 10, "pred_len": 10, 
                "x scale": np.max(gt_x[1]) - np.min(gt_x[1]), "x mean": np.mean(gt_x[1]),
                "y scale": np.max(gt_v[1]) - np.min(gt_v[1]), "y mean": np.mean(gt_v[1])
            }],
            "2014":[{ 
                "gt": Curve({"x": gt_x[1], "y": gt_v[1]}), "pred": Curve({}), 
                "gt_len": 10, "pred_len": 0
            }]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 2/3
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)

    def test_extra_subplot(self):
        pred = pd.read_csv("Tests/evaluator/P-49-O6_pred_extra_subplot.csv")
        gt = pd.read_csv("Tests/evaluator/P-49-O6_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        pred_v = [
            [3.9, 3.6, 3.4, 3.1, 2.9, 2.6, 2.4, 2.1, 1.9, 1.7], 
            [3.9, 3.7, 3.4, 3.2, 2.8, 2.5, 2.3, 2.0, 1.8, 1.6], 
            [3.9, 3.7, 3.5, 3.2, 2.9, 2.6, 2.3, 2.0, 1.8, 1.6]
        ]
        gt_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        gt_v = [
            [3.8055555555555554, 3.321428571428571, 3.182539682539683, 3.003968253968254, 2.7182539682539684, 2.515873015873016, 2.369047619047619, 2.174603174603175, 2.031746031746032, 1.6746031746031744], 
            [3.801587301587301, 3.174603174603175, 3.003968253968254, 2.984126984126984, 2.6468253968253967, 2.4404761904761907, 2.4047619047619047, 2.087301587301587, 1.9603174603174605, 1.6825396825396823], 
            [3.78968253968254, 3.134920634920635, 2.9444444444444446, 2.880952380952381, 2.5396825396825395, 2.384920634920635, 2.325396825396825, 2.0, 1.8968253968253967, 1.6111111111111112]
        ]
        x_scale = [np.max(gt_x[0]) - np.min(gt_x[0]), np.max(gt_x[1]) - np.min(gt_x[1]), np.max(gt_x[2]) - np.min(gt_x[2])]
        x_mean = [np.mean(gt_x[0]), np.mean(gt_x[1]), np.mean(gt_x[2])]
        y_scale = [np.max(gt_v[0]) - np.min(gt_v[0]), np.max(gt_v[1]) - np.min(gt_v[1]), np.max(gt_v[2]) - np.min(gt_v[2])]
        y_mean = [np.mean(gt_v[0]), np.mean(gt_v[1]), np.mean(gt_v[2])]
        curves_in_subplot = {
            "2012":[{ 
                "gt": Curve({"x": gt_x[0], "y": gt_v[0]}),
                "pred": Curve({"x": pred_x[0], "y": pred_v[0]}),
                "gt_len": 10, "pred_len": 10,
                "x scale": x_scale[0], "x mean": x_mean[0],
                "y scale": y_scale[0], "y mean": y_mean[0]
            }],
            "2013":[{
                "gt": Curve({"x": gt_x[1], "y": gt_v[1]}),
                "pred": Curve({"x": pred_x[1], "y": pred_v[1]}),
                "gt_len": 10, "pred_len": 10,
                "x scale": x_scale[1], "x mean": x_mean[1],
                "y scale": y_scale[1], "y mean": y_mean[1]
            }],
            "2014":[{
                "gt": Curve({"x": gt_x[2], "y": gt_v[2]}),
                "pred": Curve({"x": pred_x[2], "y": pred_v[2]}),
                "gt_len": 10, "pred_len": 10,
                "x scale": x_scale[2], "x mean": x_mean[2],
                "y scale": y_scale[2], "y mean": y_mean[2]
            }],
            "2015":[{ 
                "gt": Curve({}), "pred": Curve({"x": pred_x[2], "y": pred_v[2]}), 
                "gt_len": 0, "pred_len": 10 
            }]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 3/4
        self.assertSameDict(perf, ans)

    def test_pred_missing_err(self):
        pred = [pd.read_csv("Tests/evaluator/P-41-O2_pred.csv")]
        gt = [pd.read_csv("Tests/evaluator/P-41-O2_gt.csv")]
        perf = evaluate_plot(pred, gt)

        pred_value = [[0.13, 0.15, 0.18, 0.9, 0.12, 0.11], [0.35, 0.32, 0.3, 0.28, 0.27, 0.29]]
        gt_value = [[0.1005324813631522, 0.134185303514377, 0.1631522896698615, 0.1005324813631522, 0.1478168264110756, 0.1226837060702875], [0.3608093716719915, 0.3135250266240681, 0.2696485623003194, 0.2636847710330138, 0.3020234291799787, 0.2879659211927582]]
        gt_err = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0298189563365282, 0.0259850905218317, 0.0323748668796592, 0.027689030883919, 0.027689030883919, 0.0255591054313099]]
        pred_err = [[0]*len(gt_err[0]), [0]*len(gt_err[1])]
        curves_in_subplot = {
            "1": [
                {
                    "gt": Curve({"y": gt_value[0], "err": gt_err[0]}),
                    "pred": Curve({"y": pred_value[0], "err": pred_err[0]}),
                    "gt_len": 6, "pred_len": 6,
                    "y scale": np.max(gt_value[0]) - np.min(gt_value[0]), "y mean": np.mean(gt_value[0]),
                    "err scale": np.max(gt_err[0]) - np.min(gt_err[0]), "err mean": np.mean(gt_err[0])
                },
                {
                    "gt": Curve({"y": gt_value[1], "err": gt_err[1]}),
                    "pred": Curve({"y": pred_value[1], "err": pred_err[1]}),
                    "gt_len": 6, "pred_len": 6,
                    "y scale": np.max(gt_value[1]) - np.min(gt_value[1]), "y mean": np.mean(gt_value[1]),
                    "err scale": np.max(gt_err[1]) - np.min(gt_err[1]), "err mean": np.mean(gt_err[1])
                }
            ]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)
    
    def test_err_zero(self):
        pred = [pd.read_csv("Tests/evaluator/P-41-O5_pred.csv")]
        gt = [pd.read_csv("Tests/evaluator/P-41-O5_gt.csv")]
        perf = evaluate_plot(pred, gt)
    
    def test_partial_range(self):
        pred = [pd.read_csv("Tests/evaluator/P-92-O6_pred.csv")]
        gt = [pd.read_csv("Tests/evaluator/P-92-O6_gt.csv")]
        perf = evaluate_plot(pred, gt)

        pred_x = [[0, (1.0, 5.0), (6.0, 10.0), (11.0, 15.0), (16.0, 20.0), (21.0, 30.0), (31.0, 40.0), ">40"], [(0.0, 10.0), (11.0, 20.0), (21.0, 30.0), (31.0, 40.0), (41.0, 50.0), (51.0, 60.0), ">60"]]
        pred_y = [[0.05, 0.1, 0.275, 0.25, 0.15, 0.125, 0.05, 0.025], [0.225, 0.2, 0.175, 0.15, 0.125, 0.075, 0.05]]
        gt_x = [[0, (1.0, 5.0), (6.0, 10.0), (11.0, 15.0), (16.0, 20.0), (21.0, 30.0), (31.0, 40.0), ">40"], [0, (1.0, 10.0), (11.0, 20.0), (21.0, 30.0), (31.0, 40.0), (41.0, 50.0), (51.0, 60.0), ">60"]]
        gt_y = [[0.0014263074484945, 0.0556259904912837, 0.2857369255150555, 0.2714738510301109, 0.0974643423137877, 0.1640253565768621, 0.06513470681458, 0.0532488114104596], [0.0011848341232227, 0.0371248025276461, 0.2342022116903633, 0.2049763033175355, 0.1595576619273301, 0.1248025276461295, 0.0624012638230648, 0.1674565560821485]]
        curves_in_subplot = {
            "1": [{
                "gt": Curve({"x": gt_x[0], "y": gt_y[0]}),
                "pred": Curve({"x": pred_x[0], "y": pred_y[0]}),
                "gt_len": 8, "pred_len": 8,
                "x scale": 40.0, "x mean": 206/13,
                "y scale": np.max(gt_y[0]) - np.min(gt_y[0]), "y mean": np.mean(gt_y[0])
            }], 
            "2": [{
                "gt": Curve({"x": gt_x[1][1:], "y": gt_y[1][1:]}),
                "pred": Curve({"x": pred_x[1], "y": pred_y[1]}),
                "gt_len": len(gt_y[1]), "pred_len": len(pred_y[1]),
                "x scale": 60.0, "x mean": 366/13,
                "y scale": np.max(gt_y[1]) - np.min(gt_y[1]), "y mean": np.mean(gt_y[1])
            }]
        }
        ans = cal_perf(curves_in_subplot)
        ans["Curve accuracy"] = 1
        ans["Curve recall"] = 1
        self.assertSameDict(perf, ans)

if __name__ == "__main__":
    unittest.main()
