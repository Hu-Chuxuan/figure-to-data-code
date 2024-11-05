#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=molmo-7b-o
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --mem=21G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/Molmo-7B-O
python Predictor.py --root ../figure-to-data --output ../output/Molmo-7B-O --model allenai/Molmo-7B-O-0924

python Predictor.py --root ../figure-to-data --output ../output/Molmo-7B-O --eval_only