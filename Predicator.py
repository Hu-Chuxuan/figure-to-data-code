import os
import cv2
import argparse
import pandas as pd

from Baseline.gpt import digitize
from Evaluator import evaluate_discrete_plot, evaluate_continuous_plot, evaluate_table

parser = argparse.ArgumentParser(description='Digitize a plot or a table from an image')
parser.add_argument('--root', type=str, help='Path to the image file')
parser.add_argument("--output", type=str, help="Path to the output CSV file")
parser.add_argument('--api', type=str, help='OpenAI API key')
parser.add_argument('--org', type=str, help='OpenAI organization')
args = parser.parse_args()

class Dataset:
    def __init__(self, root, types, paper_list=None):
        self.samples = []
        if paper_list is None:
            paper_list = os.listdir(root)
        for paper in paper_list:
            samples = os.listdir(os.path.join(root, paper))
            for sample in samples:
                if sample[0] in types and (sample.endswith(".png") or sample.endswith(".jpeg")):
                    self.samples.append((os.path.join(root, paper, sample), sample[:sample.rfind(".")]))
    
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

# val_papers = ["2", "10", "14", "16", "17", "20", "79", "104"]
val_papers = ["2"]
dataset = Dataset(args.root, ["P", "T"], val_papers)
for i in range(len(dataset)):
    img, file_name, gts = dataset[i]
    print("===================", file_name, "===================")
    res, response = digitize(img, args.api, args.org)
    for i, df in enumerate(res):
        if len(res) == 1:
            df.to_csv(os.path.join(args.output, file_name+".csv"), index=False)
        else:
            df.to_csv(os.path.join(args.output, f"{file_name}-{i}.csv"), index=False)
    with open(os.path.join(args.output, file_name+".txt"), "w") as f:
        f.write(response)
    if file_name[0] == "P":
        perf = evaluate_discrete_plot(res, gts)
    else:
        perf = evaluate_table(res, gts)
    print(file_name, perf)
    input("Press Enter to continue...")