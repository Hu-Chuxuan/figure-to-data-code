import os
import cv2
import argparse
import pandas as pd
import logging
import json

from Baseline.gpt import digitize
from Baseline.baseline import baseline_prompt
from PlotEvaluator import evaluate_plot, merge_perf, WrongCSVNumberError, FormatError
from TableEvaluator import evaluate_table

MAX_RETRIES = 1
RED = '\033[91m'
BLUE = '\033[94m'
GREEN = '\033[92m'
RESET = '\033[0m'

class Dataset:
    def __init__(self, root, types, paper_list=None, read_txt=False):
        self.read_txt = read_txt
        self.samples = []
        self.metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
        self.names = self.metadata["File_name"].tolist()
        if paper_list is None:
            paper_list = os.listdir(root)
        for paper in paper_list:
            if not os.path.isdir(os.path.join(root, paper)):
                continue
            samples = os.listdir(os.path.join(root, paper))
            for sample in samples:
                if sample[0] in types and (sample.endswith(".png") or sample.endswith(".jpeg")):
                    name = sample[:sample.rfind(".")]
                    if name[0] == "T" or name+".csv" in samples:
                        self.samples.append((os.path.join(root, paper, sample), paper, name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, paper, name = self.samples[idx]
        sample_idx = name[name.find("-O")+2:]
        if "P" == name[0]:
            meta = self.metadata.loc[self.names.index(name)].to_dict()
            meta["Paper_idx"] = paper
        else:
            meta = {"Type": "Table", "Paper_idx": paper}
        img = cv2.imread(path)
        gts = []
        path = path[:path.rfind("/")]
        files = sorted(os.listdir(path))
        for file in files:
            if not file.endswith(".csv"):
                continue
            file_idx = file[file.find("-O")+2:file.rfind(".")]
            if "-" in file_idx:
                file_idx = file_idx[:file_idx.find("-")]
            if file_idx == sample_idx and name in file:
                if self.read_txt:
                    with open(os.path.join(path, file), "r") as f:
                        gts.append(f.read())
                else:
                    gts.append(pd.read_csv(os.path.join(path, file)))
        return {"image": img, "gt": gts, "metadata": meta}

def main(args):
    train_paper = ["2", "8", "10", "12", "14", "16", "20", "26", "32", "34", "38", "40", "41", "42", "43", "47", "49", "52", "55", "58", "61", "66", "68", "74", "86", "90", "92", "100"]
    valid_paper = ["4", "6", "18", "22", "24", "28", "30", "36", "46", "48", "56", "62", "72", "84", "88", "94", "96", "100"]
    dataset = Dataset(args.root, ["P"], valid_paper)

    total_csv = [0, 0]
    valid_csv = [0, 0]
    invalid_format = [0, 0]
    invalid_num = [0, 0]
    invalid_col = [0, 0]
    perf_per_sample = {}

    for i in range(len(dataset)):
        data = dataset[i]
        img, gts, meta = data["image"], data["gt"], data["metadata"]
        file_name = meta["File_name"]
        paper = meta["Paper_idx"]
        if not os.path.exists(os.path.join(args.output, str(paper))):
            os.makedirs(os.path.join(args.output, str(paper)))
        perf = None
        print("===================", file_name, meta["Type"], "===================")
        if file_name[0] == "P":
            idx = 0
        else:
            idx = 1
        total_csv[idx] += 1
        for retry in range(MAX_RETRIES):
            print("***", retry+1, "trial ***")
            if args.eval_only:
                res = []
                results = os.listdir(args.output)
                sample_idx = file_name[file_name.find("-O")+2:]
                if "-" in sample_idx:
                    sample_idx = sample_idx[:sample_idx.find("-")]
                for paper_idx in results:
                    if os.path.isdir(os.path.join(args.output, paper_idx)):
                        files = os.listdir(os.path.join(args.output, paper_idx))
                        for file in files:
                            if file.endswith(".csv"):
                                file_idx = file[file.find("-O")+2:file.rfind(".")]
                                if "-" in file_idx:
                                    file_idx = file_idx[:file_idx.find("-")]
                                if file_idx == sample_idx and file_name in file:
                                    res.append(pd.read_csv(os.path.join(args.output, paper_idx, file)))
            else:
                try:
                    res, response = digitize(baseline_prompt, img, args.api, args.org, args.model)
                    with open(os.path.join(args.output, str(paper), file_name+".txt"), "w") as f:
                        f.write(response)
                except pd.errors.ParserError as e:
                    print(e)
                    invalid_format[idx] += 1
                    continue
                except Exception as e:
                    logging.error(RED + str(e) + RESET)
                    input("Press Enter to continue...")
            
                if res == None:
                    continue
                for i, df in enumerate(res):
                    if len(res) == 1:
                        df.to_csv(os.path.join(args.output, str(paper), file_name+".csv"), index=False)
                    else:
                        df.to_csv(os.path.join(args.output, str(paper), f"{file_name}-{i}.csv"), index=False)
            
            try:
                if file_name[0] == "P":
                    perf, pair_trace = evaluate_plot(res, gts)
                    print(GREEN + pair_trace + RESET)
                else:
                    perf, pair_trace = evaluate_table(res, gts)
                valid_csv[idx] += 1
                break
            except (WrongCSVNumberError, FormatError) as e:
                if type(e) == WrongCSVNumberError:
                    invalid_num[idx] += 1
                else:
                    invalid_col[idx] += 1
                logging.warning(e)
                continue
            except Exception as e:
                logging.error(RED + str(e) + RESET)
                input("Press Enter to continue...")
                continue
        if perf is not None:
            if paper not in perf_per_sample:
                perf_per_sample[paper] = {}
            perf["Success retry"] = retry+1
            perf_per_sample[paper][file_name] = perf
        print(file_name, "performance:", perf)
    final_perfs = {}
    perfs_per_paper = {}
    plot_perfs, table_perfs = [], []
    for paper in perf_per_sample:
        plots, tables = [], []
        for key in perf_per_sample[paper]:
            if key[0] == "P":
                plots.append(perf_per_sample[paper][key])
            else:
                tables.append(perf_per_sample[paper][key])
        perfs_per_paper[paper] = {"plots": merge_perf(plots), "tables": merge_perf(tables)}
        plot_perfs.append(perfs_per_paper[paper]["plots"])
        table_perfs.append(perfs_per_paper[paper]["tables"])
    final_perfs["plots"] = merge_perf(plot_perfs)
    final_perfs["tables"] = merge_perf(table_perfs)

    print("====================================\n")
    if total_csv[0] > 0:
        final_perfs["plots"]["Total"] = total_csv[0]
        final_perfs["plots"]["Valid"] = valid_csv[0]
        final_perfs["plots"]["Invalid CSV format"] = invalid_format[0]
        final_perfs["plots"]["Invalid CSV number"] = invalid_num[0]
        final_perfs["plots"]["Invalid column"] = invalid_col[0]
        print(f"Plot: total {total_csv[0]}, valid {valid_csv[0]}, invalid format {invalid_format[0]}, invalid number {invalid_num[0]}, invalid column {invalid_col[0]}")
        print("Final performance:", )
        for key, value in final_perfs["plots"].items():
            print("\t", key, ":", value)
    if total_csv[1] > 0:
        final_perfs["tables"]["Total"] = total_csv[1]
        final_perfs["tables"]["Valid"] = valid_csv[1]
        final_perfs["tables"]["Invalid CSV format"] = invalid_format[1]
        final_perfs["tables"]["Invalid CSV number"] = invalid_num[1]
        final_perfs["tables"]["Invalid column"] = invalid_col[1]
        print(f"Table: total {total_csv[1]}, valid {valid_csv[1]}, invalid format {invalid_format[1]}, invalid number {invalid_num[1]}, invalid column {invalid_col[1]}")
        print("Final performance:", final_perfs["tables"])
    print("\n====================================")

    with open(os.path.join(args.output, "no-P-30-O2/perfs_dot.json"), "w") as f:
        json.dump(final_perfs, f, indent=4)
    with open(os.path.join(args.output, "no-P-30-O2/perfs_per_paper_dot.json"), "w") as f:
        json.dump(perfs_per_paper, f, indent=4)
    with open(os.path.join(args.output, "no-P-30-O2/perf_per_sample_dot.json"), "w") as f:
        json.dump(perf_per_sample, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Digitize a plot or a table from an image')
    parser.add_argument('--root', type=str, help='Path to the image file')
    parser.add_argument("--output", type=str, help="Path to the output CSV file")
    parser.add_argument('--api', type=str, help='OpenAI API key')
    parser.add_argument('--org', type=str, help='OpenAI organization')
    parser.add_argument('--model', type=str)
    parser.add_argument('--eval_only', action="store_true")
    args = parser.parse_args()

    main(args)