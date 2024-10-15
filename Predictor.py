import os
import cv2
import argparse
import pandas as pd

from Baseline.gpt import digitize
from Evaluator import evaluate_plot

parser = argparse.ArgumentParser(description='Digitize a plot or a table from an image')
parser.add_argument('--root', type=str, help='Path to the image file')
parser.add_argument("--output", type=str, help="Path to the output CSV file")
parser.add_argument('--api', type=str, help='OpenAI API key')
parser.add_argument('--org', type=str, help='OpenAI organization')
args = parser.parse_args()

class Dataset:
    def __init__(self, root, types):
        self.samples = []
        cur_path = root
        queue = [cur_path]
        while len(queue) > 0:
            cur_path = queue.pop(0)
            if os.path.isfile(cur_path):
                if not cur_path.endswith(".csv") and cur_path[cur_path.rfind("/")+1] in types:
                    self.samples.append(cur_path)
            else:
                files = os.listdir(cur_path)
                for file in files:
                    queue.append(os.path.join(cur_path, file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        img = cv2.imread(path)
        gts = []
        name = path[path.rfind("/")+1:path.rfind(".")]
        path = path[:path.rfind("/")]
        files = os.listdir(path)
        for file in files:
            if name in file and file.endswith(".csv"):
                gts.append(pd.read_csv(os.path.join(path, file)))
        return img, name, gts

dataset = Dataset(args.root, ["P"])
for i in range(len(dataset)):
    img, file_name, gts = dataset[i]
    res, response = digitize(img, args.api, args.org)
    for i, df in enumerate(res):
        if len(res) == 1:
            df.to_csv(os.path.join(args.output, file_name+".csv"), index=False)
        else:
            df.to_csv(os.path.join(args.output, f"{file_name}-{i}.csv"), index=False)
    with open(os.path.join(args.output, file_name+".txt"), "w") as f:
        f.write(response)
    perf = evaluate_plot(res, gts)
    with open(os.path.join(args.output, file_name+".json"), "w") as f:
        f.write(perf)