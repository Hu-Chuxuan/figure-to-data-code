#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=molmo-7b-d
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --mem=21G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/Molmo-7B-D
python Predictor.py --root ../figure-to-data --output ../output/Molmo-7B-D --model allenai/Molmo-7B-D-0924 --paper_list 1 2 3 4 5