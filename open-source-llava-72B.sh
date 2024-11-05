#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=llava-72b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=ddkang
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:3
#SBATCH --time=48:00:00
#SBATCH --mem=240G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/Llava-72B
python Predictor.py --root ../figure-to-data --output ../output/Llava-72B --model lmms-lab/llava-onevision-qwen2-72b-si