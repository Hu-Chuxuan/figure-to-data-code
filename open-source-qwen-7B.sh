#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=qwen-7b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=ddkang
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:1
#SBATCH --time=10:00:00
#SBATCH --mem=21G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/Qwen-7B
python Predictor.py --root ../figure-to-data --output ../output/Qwen-7B --model Qwen/Qwen2-VL-7B-Instruct