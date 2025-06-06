#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=0-molmo-72b-ddkang
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=ddkang
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:3
#SBATCH --time=04:00:00
#SBATCH --mem=240G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/Molmo-72B-0-Shot-COT
python Predictor.py --root ../figure-to-data --output ../output/Molmo-72B-0-Shot-COT --model allenai/Molmo-72B-0924 --Prompt cot --shot 0 --types Table --paper_list 113 114 115 116 117 118 119 120 121 122