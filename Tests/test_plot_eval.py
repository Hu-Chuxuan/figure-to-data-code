import pandas as pd
import unittest
import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from PlotEvaluator import evaluate_plot, cal_perf, merge_perf, cal_metrics, pair_score, pair_curves, pair_data_points

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
                "Identified rate": 0.5,
                "Identified recall": 0.6
            },
            {
                "Value performance": {"MAE": 0.11, "MSE": 0.011},
                "X performance": {"MAE": 0.22, "MSE": 0.022},
                "Error performance": {"MAE": 0.33, "MSE": 0.033},
                "Overall performance": {"MAE": 0.44, "MSE": 0.044},
                "Identified rate": 0.55,
                "Identified recall": 0.66
            },
            {
                "Value performance": {"MAE": 0.111, "MSE": 0.0111},
                "X performance": {"MAE": 0.222, "MSE": 0.0222},
                "Error performance": {"MAE": 0.333, "MSE": 0.0333},
                "Overall performance": {"MAE": 0.444, "MSE": 0.0444},
                "Identified rate": 0.555,
                "Identified recall": 0.666
            },
        ]
        ans = {
            "Value performance": {"MAE": 0.107, "MSE": 0.0107},
            "X performance": {"MAE": 0.214, "MSE": 0.0214},
            "Error performance": {"MAE": 0.321, "MSE": 0.0321},
            "Overall performance": {"MAE": 0.428, "MSE": 0.0428},
            "Identified rate": 0.535,
            "Identified recall": 0.642
        }
        self.assertSameDict(merge_perf(perf_list), ans)
    
    def test_merge_perf_missing_parts(self):
        perf_list = [
            {
                "Value performance": {"MAE": 0.1, "MSE": 0.01},
                "X performance": {"MAE": 0.2, "MSE": 0.02},
                "Error performance": {"MAE": 0.3, "MSE": 0.03},
                "Overall performance": {"MAE": 0.4, "MSE": 0.04},
                "Identified rate": 0.5,
                "Identified recall": 0.6
            },
            {
                "Value performance": {"MAE": 0.11, "MSE": 0.011},
                "X performance": {"MAE": 0.22, "MSE": 0.022},
                "Overall performance": {"MAE": 0.44, "MSE": 0.044},
                "Identified recall": 0.66
            },
            {
                "Value performance": {"MAE": 0.111, "MSE": 0.0111},
                "Error performance": {"MAE": 0.333, "MSE": 0.0333},
                "Overall performance": {"MAE": 0.444, "MSE": 0.0444},
                "Identified rate": 0.555,
            },
        ]
        ans = {
            "Value performance": {"MAE": 0.107, "MSE": 0.0107},
            "X performance": {"MAE": 0.21, "MSE": 0.021},
            "Error performance": {"MAE": 0.3165, "MSE": 0.03165},
            "Overall performance": {"MAE": 0.428, "MSE": 0.0428},
            "Identified rate": 0.5275,
            "Identified recall": 0.63
        }
        self.assertSameDict(merge_perf(perf_list), ans)

    def test_merge_perf_nan(self):
        perf_list = [
            {
                "Value performance": {"MAE": 0.1, "MSE": 0.01},
                "X performance": {"MAE": 0.2, "MSE": 0.02},
                "Error performance": {"MAE": 0.3, "MSE": 0.03},
                "Overall performance": {"MAE": 0.4, "MSE": 0.04},
                "Identified rate": np.nan,
                "Identified recall": 0.6
            },
            {
                "Value performance": {"MAE": 0.11, "MSE": 0.011},
                "X performance": {"MAE": np.nan, "MSE": 0.022},
                "Error performance": {"MAE": 0.33, "MSE": 0.033},
                "Overall performance": {"MAE": 0.44, "MSE": np.nan},
                "Identified rate": np.nan,
                "Identified recall": np.nan
            },
            {
                "Value performance": {"MAE": np.nan, "MSE": 0.0111},
                "X performance": {"MAE": 0.222, "MSE": 0.0222},
                "Error performance": {"MAE": 0.333, "MSE": np.nan},
                "Overall performance": {"MAE": 0.444, "MSE": np.nan},
                "Identified rate": np.nan,
                "Identified recall": 0.666
            },
        ]
        ans = {
            "Value performance": {"MAE": 0.105, "MSE": 0.0107},
            "X performance": {"MAE": 0.211, "MSE": 0.0214},
            "Error performance": {"MAE": 0.321, "MSE": 0.0315},
            "Overall performance": {"MAE": 0.428, "MSE": 0.04},
            "Identified rate": np.nan,
            "Identified recall": 0.633
        }
        self.assertSameDict(merge_perf(perf_list), ans)

    def test_cal_perf(self):
        pred_value = [-0.15, -0.1, -0.05, 0.0, 0.1, 0.05]
        pred_err = [0.1, 0.08, 0.12, 0.07, 0.09, 0.06]
        gt_value = [-0.1957403651115618, -0.3600405679513184, -0.061866125760649, -0.1369168356997971, 0.0740365111561865, 0.0801217038539553]
        gt_err = [0.2251521298174442, 0.2718052738336714, 0.1176470588235294, 0.1440162271805274, 0.2393509127789046, 0.2900608519269776]
        curves_in_subplot = {
            "1": [
                { "x": ([], []), "y": (pred_value, gt_value), "err": (pred_err, gt_err), "gt_len": 10, "pred_len": 6 }
            ]
        }
        ans = {
            "Value performance": cal_metrics(pred_value, gt_value),
            "Error performance": cal_metrics(pred_err, gt_err),
            "Overall performance": cal_metrics(pred_value + pred_err, gt_value + gt_err),
            "Identified rate": 0.6,
            "Identified recall": 1.0
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_multi_curve(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]
        curves_in_subplot = {
            "1": [
                {"x": (pred_x[0], gt_x[0]), "y": (pred_value[0], gt_value[0]), "err": ([], []), "gt_len": 5, "pred_len": 5},
                {"x": (pred_x[1], gt_x[1]), "y": (pred_value[1], gt_value[1]), "err": ([], []), "gt_len": 10, "pred_len": 8}
            ]
        }
        first_curve = {
            "X performance": cal_metrics(pred_x[0], gt_x[0]),
            "Value performance": cal_metrics(pred_value[0], gt_value[0]),
            "Overall performance": cal_metrics(pred_x[0] + pred_value[0], gt_x[0] + gt_value[0]),
            "Identified rate": 1.0,
            "Identified recall": 1.0
        }
        second_curve = {
            "X performance": cal_metrics(pred_x[1], gt_x[1]),
            "Value performance": cal_metrics(pred_value[1], gt_value[1]),
            "Overall performance": cal_metrics(pred_x[1] + pred_value[1], gt_x[1] + gt_value[1]),
            "Identified rate": 0.5,
            "Identified recall": 0.625
        }
        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in first_curve["X performance"].keys():
            merged_x[key] = (first_curve["X performance"][key] + second_curve["X performance"][key]) / 2
            merged_v[key] = (first_curve["Value performance"][key] + second_curve["Value performance"][key]) / 2
            merged_overall[key] = (first_curve["Overall performance"][key] + second_curve["Overall performance"][key]) / 2
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            "Overall performance": merged_overall,
            "Identified rate": (first_curve["Identified rate"] + second_curve["Identified rate"]) / 2,
            "Identified recall": (first_curve["Identified recall"] + second_curve["Identified recall"]) / 2
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_multi_subplot(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]
        curves_in_subplot = {
            "1": [
                {"x": (pred_x[0], gt_x[0]), "y": (pred_value[0], gt_value[0]), "err": ([], []), "gt_len": 5, "pred_len": 5},
            ],
            "2": [
                {"x": (pred_x[1], gt_x[1]), "y": (pred_value[1], gt_value[1]), "err": ([], []), "gt_len": 10, "pred_len": 8}
            ]
        }
        first_curve = {
            "X performance": cal_metrics(pred_x[0], gt_x[0]),
            "Value performance": cal_metrics(pred_value[0], gt_value[0]),
            "Overall performance": cal_metrics(pred_x[0] + pred_value[0], gt_x[0] + gt_value[0]), 
            "Identified rate": 1.0,
            "Identified recall": 1.0
        }
        second_curve = {
            "X performance": cal_metrics(pred_x[1], gt_x[1]),
            "Value performance": cal_metrics(pred_value[1], gt_value[1]),
            "Overall performance": cal_metrics(pred_x[1] + pred_value[1], gt_x[1] + gt_value[1]),
            "Identified rate": 0.5,
            "Identified recall": 0.625
        }
        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in first_curve["X performance"].keys():
            merged_x[key] = (first_curve["X performance"][key] + second_curve["X performance"][key]) / 2
            merged_v[key] = (first_curve["Value performance"][key] + second_curve["Value performance"][key]) / 2
            merged_overall[key] = (first_curve["Overall performance"][key] + second_curve["Overall performance"][key]) / 2
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            "Overall performance": merged_overall,
            "Identified rate": (first_curve["Identified rate"] + second_curve["Identified rate"]) / 2,
            "Identified recall": (first_curve["Identified recall"] + second_curve["Identified recall"]) / 2
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

    def test_cal_perf_multi_subplot_multi_curve(self):
        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015], [1, 2, 3, 4]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15], [-0.15, -0.1, -0.05, 0.0]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327], [-0.1957403651115618, -0.3600405679513184, -0.061866125760649, -0.1369168356997971]]
        pred_err = [0.1, 0.08, 0.12, 0.07]
        gt_err = [0.2251521298174442, 0.2718052738336714, 0.1176470588235294, 0.1440162271805274]

        curves_in_subplot = {
            "1": [
                {"x": (pred_x[0], gt_x[0]), "y": (pred_value[0], gt_value[0]), "err": ([], []), "gt_len": 5, "pred_len": 5},
                {"x": (pred_x[1], gt_x[1]), "y": (pred_value[1], gt_value[1]), "err": ([], []), "gt_len": 10, "pred_len": 8}
            ], 
            "2": [
                {"x": ([], []), "y": (pred_value[2], gt_value[2]), "err": (pred_err, gt_err), "gt_len": 16, "pred_len": 20}
            ]
        }

        curves = [{}, {}, {}]
        curves[0]["X performance"] = cal_metrics(pred_x[0], gt_x[0])
        curves[0]["Value performance"] = cal_metrics(pred_value[0], gt_value[0])
        curves[0]["Overall performance"] = cal_metrics(pred_x[0] + pred_value[0], gt_x[0] + gt_value[0])
        curves[0]["Identified rate"] = 1.0
        curves[0]["Identified recall"] = 1.0

        curves[1]["X performance"] = cal_metrics(pred_x[1], gt_x[1])
        curves[1]["Value performance"] = cal_metrics(pred_value[1], gt_value[1])
        curves[1]["Overall performance"] = cal_metrics(pred_x[1] + pred_value[1], gt_x[1] + gt_value[1])
        curves[1]["Identified rate"] = 0.5
        curves[1]["Identified recall"] = 0.625

        curves[2]["Value performance"] = cal_metrics(pred_value[2], gt_value[2])
        curves[2]["Error performance"] = cal_metrics(pred_err, gt_err)
        curves[2]["Overall performance"] = cal_metrics(pred_value[2] + pred_err, gt_value[2] + gt_err)
        curves[2]["Identified rate"] = 4/16
        curves[2]["Identified recall"] = 4/20

        merged_x, merged_v, merged_overall = {}, {}, {}
        for key in curves[0]["X performance"].keys():
            merged_x[key] = (curves[0]["X performance"][key] + curves[1]["X performance"][key]) / 2
            merged_v[key] = ((curves[0]["Value performance"][key] + curves[1]["Value performance"][key])/2 + curves[2]["Value performance"][key]) / 2
            merged_overall[key] = ((curves[0]["Overall performance"][key] + curves[1]["Overall performance"][key])/2 + curves[2]["Overall performance"][key]) / 2
        ans = {
            "X performance": merged_x,
            "Value performance": merged_v,
            "Overall performance": merged_overall,
            "Error performance": curves[2]["Error performance"],
            "Identified rate": ((curves[0]["Identified rate"] + curves[1]["Identified rate"])/2 + curves[2]["Identified rate"]) / 2,
            "Identified recall": ((curves[0]["Identified recall"] + curves[1]["Identified recall"])/2 + curves[2]["Identified recall"]) / 2
        }
        self.assertSameDict(cal_perf(curves_in_subplot), ans)

class TestPlotEvaluatorPairing(TestPlotEvaluator):
    def test_pair_score(self):
        pass

    def test_pair_curves(self):
        pass

    def test_pair_data_points(self):
        pass
    
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
        curves_in_subplot = {
            "1": [
                {
                    "x": ([], []),
                    "y": (pred_value[0], gt_value[0]),
                    "err": (pred_err[0], gt_err[0]),
                    "gt_len": 3,
                    "pred_len": 3
                },
                {
                    "x": ([], []),
                    "y": (pred_value[1], gt_value[1]),
                    "err": (pred_err[1], gt_err[1]),
                    "gt_len": 3,
                    "pred_len": 3
                }
            ]
        }
        self.assertSameDict(perf, cal_perf(curves_in_subplot))
    
    def test_hist_range(self):
        # histogram with Type-1 being range + no subplot value + no type-2
        pred = pd.read_csv("Tests/evaluator/P-10-O1_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-10-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])
        
        pred_x = [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0]
        pred_value = [75, 75, 85, 95, 80, 20, 5, 0, 5, 5]
        gt_x = [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0]
        gt_value = [65.97087378640776, 64.95145631067962, 79.80582524271844, 88.10679611650485, 70.04854368932038, 21.99029126213592, 5.825242718446603, 1.893203883495147, 1.0194174757281471, 3.932038834951456]
        curves_in_subplot = {
            "1": [
                {
                    "x": (pred_x, gt_x),
                    "y": (pred_value, gt_value),
                    "err": ([], []),
                    "gt_len": 10,
                    "pred_len": 10
                }
            ]
        }
        self.assertSameDict(perf, cal_perf(curves_in_subplot))

    def DISABLE_test_cont_simple(self):
        # continuous plot + multiple curves + no subplot valuez
        pred = pd.read_csv("Tests/evaluator/P-20-O1_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-20-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])
        print(self._testMethodName, perf)

    def test_cross_pairing(self):
        # histogram + multiple panels + no type-2
        pred = pd.read_csv("Tests/evaluator/P-14-O2_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-14-O2_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_x = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
        pred_value = [0.05, 0.03, 0.15, 0.08, 0.22, 0.07, 0.13, 0.06, 0.04, 0.02]
        gt_x = [1996.0, 1997.0, 1998.0, 1999.0, 1999.0, 2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2004.0, 2005.0, 2007.0, 2008.0, 2008.0, 2009.0, 2009.0, 2010.0, 2012.0, 2013.0]
        gt_value = [0.0447214076246334, 0.0271260997067449, 0.1590909090909091, 0.0843108504398827, 0.2155425219941349, 0.0659824046920821, 0.1422287390029326, 0.123900293255132, 0.0645161290322581, 0.0344574780058651]

    def test_gt_more_than_pred(self):
        # ground truth has more data then prediction 
        pred = pd.read_csv("Tests/evaluator/P-14-O1_pred.csv")
        gt = pd.read_csv("Tests/evaluator/P-14-O1_gt.csv")
        perf = evaluate_plot([pred], [gt])

        pred_x = [[1995, 2000, 2005, 2010, 2015], [1995, 2000, 2005, 2010, 2015]]
        pred_value = [[85, 80, 75, 70, 65], [25, 22, 20, 18, 15]]
        gt_x = [[1995, 2000, 2005, 2010, 2012], [1995, 2000, 2005, 2010, 2012]]
        gt_value = [[84.06889128094726, 79.1173304628633, 70.18299246501616, 61.14101184068892, 57.91173304628633], [25.188374596340154, 21.95909580193758, 15.931108719052745, 13.024757804090413, 12.917115177610327]]
        curves_in_subplot = {
            "1": [
                {
                    "x": (pred_x[0], gt_x[0]),
                    "y": (pred_value[0], gt_value[0]),
                    "err": ([], []),
                    "gt_len": 18,
                    "pred_len": 5
                },
                {
                    "x": (pred_x[1], gt_x[1]),
                    "y": (pred_value[1], gt_value[1]),
                    "err": ([], []),
                    "gt_len": 18,
                    "pred_len": 5
                }
            ]
        }
        self.assertSameDict(perf, cal_perf(curves_in_subplot))

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
                    "x": (pred_x, gt_x),
                    "y": (pred_value, gt_value),
                    "err": ([], []),
                    "gt_len": 12,
                    "pred_len": 13
                }
            ]
        }
        self.assertSameDict(perf, cal_perf(curves_in_subplot))
    
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
            "2012":[{ "x": (pred_x[0], gt_x[0]), "y": (pred_v[0], gt_v[0]), "err": ([], []), "gt_len": 10, "pred_len": 10 }],
            "2013":[{ "x": (pred_x[1], gt_x[1]), "y": (pred_v[1], gt_v[1]), "err": ([], []), "gt_len": 10, "pred_len": 10 }],
            "2014":[{ "x": (pred_x[2], gt_x[2]), "y": (pred_v[2], gt_v[2]), "err": ([], []), "gt_len": 10, "pred_len": 10 }]
        }
        self.assertSameDict(perf, cal_perf(curves_in_subplot))

    def DISABLE_test_x_str(self):
        pass

    def DISABLE_test_x_mix_str_int(self):
        pass

    def DISABLE_test_missing_subplot(self):
        pass

    def DISABLE_test_missing_type2(self):
        pass

    def DISABLE_test_unexisted_subplot(self):
        pass

    def DISABLE_test_unexisted_type2(self):
        pass

if __name__ == "__main__":
    unittest.main()
