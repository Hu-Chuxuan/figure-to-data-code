#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=2-$i-qwen-72b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:3
#SBATCH --time=04:00:00
#SBATCH --mem=240G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir -p ../output/Qwen-72B-2-Shot-COT
python Predictor.py --root ../figure-to-data --output ../output/Qwen-72B-2-Shot-COT --model Qwen/Qwen2-VL-72B-Instruct --Prompt cot --shot 2 --types Table --paper_list 20 21 22 39 40 41 42