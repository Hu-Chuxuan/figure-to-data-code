import os
import cv2
import argparse
import pandas as pd
import logging

from Baseline.gpt import digitize
from Evaluator import evaluate_plot, evaluate_table, WrongCSVNumberError, FormatError

parser = argparse.ArgumentParser(description='Digitize a plot or a table from an image')
parser.add_argument('--root', type=str, help='Path to the image file')
parser.add_argument("--output", type=str, help="Path to the output CSV file")
parser.add_argument('--api', type=str, help='OpenAI API key')
parser.add_argument('--org', type=str, help='OpenAI organization')
args = parser.parse_args()

MAX_RETRIES = 1
RED = '\033[91m'
RESET = '\033[0m'

class Dataset:
    def __init__(self, root, types, paper_list=None):
        self.samples = []
        if paper_list is None:
            paper_list = os.listdir(root)
        for paper in paper_list:
            samples = os.listdir(os.path.join(root, paper))
            for sample in samples:
                if sample[0] in types and (sample.endswith(".png") or sample.endswith(".jpeg")):
                    name = sample[:sample.rfind(".")]
                    if name[0] == "T" or name+".csv" in samples:
                        self.samples.append((os.path.join(root, paper, sample), name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, name = self.samples[idx]
        img = cv2.imread(path)
        gts = []
        path = path[:path.rfind("/")]
        files = sorted(os.listdir(path))
        for file in files:
            if name in file and file.endswith(".csv"):
                gts.append(pd.read_csv(os.path.join(path, file)))
        return img, name, gts

dataset = Dataset(args.root, ["P", "T"])
for i in range(len(dataset)):
    img, file_name, gts = dataset[i]
    perf = None
    print("===================", file_name, "===================")
    for _ in range(MAX_RETRIES):
        res, response = digitize(img, args.api, args.org)
        with open(os.path.join(args.output, file_name+".txt"), "w") as f:
            f.write(response)
        
        if res == None:
            continue
        for i, df in enumerate(res):
            if len(res) == 1:
                df.to_csv(os.path.join(args.output, file_name+".csv"), index=False)
            else:
                df.to_csv(os.path.join(args.output, f"{file_name}-{i}.csv"), index=False)
        
        try:
            if file_name[0] == "P":
                perf = evaluate_plot(res, gts)
            else:
                perf = evaluate_table(res, gts)
            break
        except (WrongCSVNumberError, FormatError) as e:
            logging.warning(e)
            continue
        except Exception as e:
            logging.error(RED + str(e) + RESET)
            continue
    print(file_name, "performance:", perf)
    input("Press Enter to continue...")