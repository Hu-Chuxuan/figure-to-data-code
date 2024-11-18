#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=0-qwen-7b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:1
#SBATCH --time=4:00:00
#SBATCH --mem=21G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/Qwen-7B-0-Shot-COT
python Predictor.py --root ../figure-to-data --output ../output/Qwen-7B-0-Shot-COT --model Qwen/Qwen2-VL-7B-Instruct --Prompt cot --shot 0 --types Table --paper_list 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122