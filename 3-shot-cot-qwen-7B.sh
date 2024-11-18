#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=3-qwen-7b
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

mkdir ../output/Qwen-7B-3-Shot-COT
python Predictor.py --root ../figure-to-data --output ../output/Qwen-7B-3-Shot-COT --model Qwen/Qwen2-VL-7B-Instruct --Prompt cot --shot 3 --paper_list 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122