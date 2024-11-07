import os
import cv2
import argparse
import pandas as pd
import logging
import json
import pickle as pkl
import numpy as np

from Baseline.mllm import GPT, Claude, Qwen, Molmo, LLAVA, InternVL
from Baseline.baseline import baseline_prompt
from PlotEvaluator import evaluate_plot, merge_perf, WrongCSVNumberError, FormatError
from TableEvaluator import evaluate_table

MAX_RETRIES = 1
RED = '\033[91m'
BLUE = '\033[94m'
GREEN = '\033[92m'
RESET = '\033[0m'
PLOT_TYPES = ["Dot", "Histogram", "Continuous", "Combo"]

class Dataset:
    def __init__(self, root, types, paper_list=None, read_txt=False):
        self.root = root
        self.read_txt = read_txt
        self.samples = []
        self.metadata = pkl.load(open(os.path.join(root, "metadata.pkl"), "rb"))
        
        for sample, meta in self.metadata.items():
            if meta["Type"] in types:
                if paper_list is None or meta["Paper Index"] in paper_list:
                    self.samples.append(sample)
        logging.warning("Number of samples: " + str(len(self.samples)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        name = self.samples[idx]
        meta = self.metadata[name]
        paper = str(meta["Paper Index"])
        paper_path = os.path.join(self.root, paper)
        if meta["Type"] == "Table":
            img_path = os.path.join(paper_path, name+".png")
        else:
            img_path = os.path.join(paper_path, name+"."+meta["Extension"])
        
        sample_idx = name[name.find("-O")+2:]
        gts = []
        read_gt_files = []
        logging.warning("start loading csvs: " + name)
        if meta["Type"] in PLOT_TYPES:
            gts.append(pd.read_csv(os.path.join(paper_path, name+".csv")))
        else:
            files = sorted(os.listdir(paper_path))
            for file in files:
                if not file.endswith(".csv"):
                    continue
                file_idx = file[file.find("-O")+2:file.rfind(".")]
                if "-" in file_idx:
                    file_idx = file_idx[:file_idx.find("-")]
                if file_idx == sample_idx and name in file:
                    if self.read_txt:
                        with open(os.path.join(paper_path, file), "r") as f:
                            gts.append(f.read())
                    else:
                        read_gt_files.append(file)
                        gts.append(pd.read_csv(os.path.join(paper_path, file)))
        logging.warning("finish reading csvs: " + name)
        return {"image_path": img_path, "gt": gts, "Paper Index": paper, "File name": name, "Type": meta["Type"]}

def stratify_results(perf_per_sample, metadata, group_by, filters=None):
    stratified_results = {}
    for sample, perf in perf_per_sample.items():
        meta = metadata[sample]
        if filters is not None:
            pass_filter = True
            for filter in filters:
                if meta[filter[0]] not in filter[1]:
                    pass_filter = False
                    break
            if not pass_filter:
                continue
        key = []
        for k in group_by:
            key.append(meta[k])
        if len(key) == 1:
            key = key[0]
            if isinstance(key, np.int64):
                key = int(key)
        else:
            key = tuple(key)
        if key not in stratified_results:
            stratified_results[key] = []
        stratified_results[key].append(perf)
    results = {}
    for key, perfs in stratified_results.items():
        results[key] = merge_perf(perfs)
    return results

def main(args):
    if args.model is None:
        mllm = None
        if not args.eval_only:
            raise ValueError("Model is required for evaluation")
    elif "gpt" in args.model.lower():
        mllm = GPT(args.api, args.org, args.model)
    elif "claude" in args.model.lower():
        mllm = Claude(args.api, args.org, args.model)
    elif "llava" in args.model.lower():
        mllm = LLAVA(args.model)
    elif "molmo" in args.model.lower():
        mllm = Molmo(args.model)
    elif "internvl" in args.model.lower():
        mllm = InternVL(args.model)
    elif "qwen" in args.model.lower():
        mllm = Qwen(args.model)
    
    dataset = Dataset(args.root, args.types, args.paper_list)

    perf_per_sample = {}

    for i in range(len(dataset)):
        data = dataset[i]
        img_path, gts, paper, file_name = data["image_path"], data["gt"], data["Paper Index"], data["File name"]
        if not os.path.exists(os.path.join(args.output, str(paper))):
            os.makedirs(os.path.join(args.output, str(paper)))
        perf = None
        logging.warning("===================" + file_name + "===================")
        for retry in range(MAX_RETRIES):
            logging.warning("***"+str(retry+1)+"trial ***")
            if args.eval_only:
                res = []
                read_res = []
                results = os.listdir(os.path.join(args.output, str(paper)))
                if file_name+".csv" in results:
                    res.append(pd.read_csv(os.path.join(args.output, str(paper), file_name+".csv")))
                    read_res.append(file_name+".csv")
                else:
                    cnt = 0
                    while True:
                        if f"{file_name}-{cnt}.csv" in results:
                            res.append(pd.read_csv(os.path.join(args.output, str(paper), f"{file_name}-{cnt}.csv")))
                            read_res.append(f"{file_name}-{cnt}.csv")
                            cnt += 1
                        else:
                            break
                if len(read_res) > 0:
                    if len(read_res) > 1 or file_name+".csv" != read_res[0]:
                        logging.warning("Reading " + file_name + " from ", read_res)
            else:
                res = None
                response, res = mllm.query(baseline_prompt, img_path)
                # try:
                #     response, res = mllm.query(baseline_prompt, img_path)
                #     with open(os.path.join(args.output, str(paper), file_name+".txt"), "w") as f:
                #         f.write(response)
                # except pd.errors.ParserError as e:
                #     logging.warning(e)
                #     perf = {"ParserError": 1, "Success": 0, "WrongCSVNumberError": 0, "FormatError": 0, "Other exception": 0}
                #     continue
                # except Exception as e:
                #     logging.error(RED + str(e) + RESET)
                #     perf = {"Other exception": 1, "Success": 0, "ParserError": 0, "WrongCSVNumberError": 0, "FormatError": 0}
                    # input("Press Enter to continue...")
            
                if res == None:
                    continue
                for i, df in enumerate(res):
                    if len(res) == 1:
                        df.to_csv(os.path.join(args.output, str(paper), file_name+".csv"), index=False)
                    else:
                        df.to_csv(os.path.join(args.output, str(paper), f"{file_name}-{i}.csv"), index=False)
            
            try:
                if data["Type"] in PLOT_TYPES:
                    perf = evaluate_plot(res, gts, data["Type"])
                    
                else:
                    perf = evaluate_table(res, gts)
                perf["Success"] = 1
                perf["ParserError"] = 0
                perf["WrongCSVNumberError"] = 0
                perf["FormatError"] = 0
                perf["Other exception"] = 0
                break
            except (WrongCSVNumberError, FormatError) as e:
                logging.warning(e)
                if type(e) == WrongCSVNumberError:
                    perf = {"WrongCSVNumberError": 1, "Success": 0, "FormatError": 0, "ParserError": 0, "Other exception": 0}
                else:
                    perf = {"FormatError": 1, "Success": 0, "WrongCSVNumberError": 0, "ParserError": 0, "Other exception": 0}
                continue
            except Exception as e:
                logging.error(RED + str(e) + RESET)
                perf = {"Other exception": 1, "Success": 0, "WrongCSVNumberError": 0, "ParserError": 0, "FormatError": 0}
                # input("Press Enter to continue...")
                continue
        if perf is not None:
            if perf["Success"] + perf["ParserError"] + perf["WrongCSVNumberError"] + perf["FormatError"] + perf["Other exception"] != 1:
                logging.warning(perf)
                # input("Press Enter to continue...")
            perf_per_sample[file_name] = perf
        logging.warning(file_name + " performance: " + str(perf))
    
    perfs_per_paper = stratify_results(perf_per_sample, dataset.metadata, ["Paper Index"])
    final_perfs = merge_perf([perfs_per_paper[paper] for paper in perfs_per_paper])

    stratified_results = {}
    stratified_results["Type"] = stratify_results(perf_per_sample, dataset.metadata, ["Type"])

    if any([t in args.types for t in PLOT_TYPES]):
        stratified_results["# Subplot"] = stratify_results(perf_per_sample, dataset.metadata, ["# Subplot"], filters=[("Type", PLOT_TYPES)])
        stratified_results["# Curve"] = stratify_results(perf_per_sample, dataset.metadata, ["# Curve"], filters=[("Type", PLOT_TYPES)])
        stratified_results["Vector/Pixel"] = stratify_results(perf_per_sample, dataset.metadata, ["Vector/Pixel"], filters=[("Type", PLOT_TYPES)])
        stratified_results["Axis"] = stratify_results(perf_per_sample, dataset.metadata, ["Axis"], filters=[("Type", PLOT_TYPES)])
        stratified_results["# Data Points"] = stratify_results(perf_per_sample, dataset.metadata, ["# Data Points"], filters=[("Type", PLOT_TYPES)])
    
    if "Table" in args.types:
        stratified_results["Rotate"] = stratify_results(perf_per_sample, dataset.metadata, ["Rotate"], filters=[("Type", ["Table"])])
        stratified_results["# Panel"] = stratify_results(perf_per_sample, dataset.metadata, ["# Panel"], filters=[("Type", ["Table"])])
        stratified_results["# Row"] = stratify_results(perf_per_sample, dataset.metadata, ["# Row"], filters=[("Type", ["Table"])])
        stratified_results["# Column"] = stratify_results(perf_per_sample, dataset.metadata, ["# Column"], filters=[("Type", ["Table"])])

    logging.warning("================= Final Performance =================")
    logging.warning(str(final_perfs))
    logging.warning("=====================================================")

    with open(os.path.join(args.output, "perfs.json"), "w") as f:
        json.dump(final_perfs, f, indent=4)
    with open(os.path.join(args.output, "perfs_stratified.json"), "w") as f:
        json.dump(stratified_results, f, indent=4)
    with open(os.path.join(args.output, "perf_per_sample.json"), "w") as f:
        json.dump(perf_per_sample, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Digitize a plot or a table from an image')
    parser.add_argument('--root', type=str, help='Path to the image file', required=True)
    parser.add_argument("--output", type=str, help="Path to the output CSV file", default="output")
    parser.add_argument('--types', type=str, nargs="+", help='Types of data to digitize', default=PLOT_TYPES+["Table"])
    parser.add_argument('--api', type=str, help='OpenAI API key')
    parser.add_argument('--org', type=str, help='OpenAI organization')
    parser.add_argument('--model', type=str)
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--paper_list', type=int, nargs="*", help='List of paper indices')
    args = parser.parse_args()

    main(args)